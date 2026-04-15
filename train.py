#!/usr/bin/env python3
"""
Unified Training Script for BiomedCLIP Fine-tuning
==================================================
Supports multiple loss functions: CLIP, SigLIP, HNL
Usage:
    python train.py --loss clip --train-dir /path/to/train --output-dir checkpoints/clip_exp
"""
import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("[WARNING] TensorBoard not available. Logging to console only.")

from config import DEFAULT_BIOMEDCLIP_CHECKPOINT, LOSS_CONFIGS, DEFAULT_TRAINING_CONFIG
from models import load_biomedclip, get_biomedclip_features, get_biomedclip_features_mgca
from data import TumorDataset
from losses import CLIPLoss, SigLIPLoss, HardNegativeLoss, MGCALoss, GLoRIALoss


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BiomedCLIP with various losses")
    
    # Data
    parser.add_argument('--train-dir', type=str, required=True, 
                        help='Training images directory')
    parser.add_argument('--val-dir', type=str, default=None,
                        help='Validation images directory (optional)')
    
    # Model
    parser.add_argument('--model-checkpoint', type=str, 
                        default=DEFAULT_BIOMEDCLIP_CHECKPOINT,
                        help='BiomedCLIP checkpoint')
    
    # Loss function
    parser.add_argument('--loss', type=str, default='clip',
                        choices=['clip', 'siglip', 'hnl', 'mgca', 'gloria'],
                        help='Loss function to use')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=DEFAULT_TRAINING_CONFIG['batch_size'])
    parser.add_argument('--epochs', type=int, default=DEFAULT_TRAINING_CONFIG['epochs'])
    parser.add_argument('--lr', type=float, default=DEFAULT_TRAINING_CONFIG['lr'])
    parser.add_argument('--weight-decay', type=float, default=DEFAULT_TRAINING_CONFIG['weight_decay'])
    parser.add_argument('--warmup-epochs', type=int, default=DEFAULT_TRAINING_CONFIG['warmup_epochs'])
    parser.add_argument('--num-workers', type=int, default=DEFAULT_TRAINING_CONFIG['num_workers'])
    
    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for checkpoints')
    parser.add_argument('--save-freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Other
    parser.add_argument('--seed', type=int, default=DEFAULT_TRAINING_CONFIG['seed'])
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_loss_function(loss_name):
    """Create loss function based on name"""
    config = LOSS_CONFIGS[loss_name]
    
    if loss_name == 'clip':
        return CLIPLoss(temperature=config['temperature'])
    elif loss_name == 'siglip':
        return SigLIPLoss(temperature=config['temperature'], bias=config['bias'])
    elif loss_name == 'hnl':
        return HardNegativeLoss(
            temperature=config['temperature'],
            beta1=config['beta1'],
            beta2=config['beta2'],
            alpha=config['alpha']
        )
    elif loss_name == 'mgca':
        return MGCALoss(
            input_dim=768,
            emb_dim=config['emb_dim'],
            hidden_dim=config['hidden_dim'],
            num_prototypes=config['num_prototypes'],
            softmax_temperature=config['softmax_temperature'],
            local_temperature=config['local_temperature'],
            proto_temperature=config['proto_temperature'],
            lambda_1=config['lambda_1'],
            lambda_2=config['lambda_2'],
            lambda_3=config['lambda_3'],
            epsilon=config['epsilon'],
            sinkhorn_iterations=config['sinkhorn_iterations'],
            bidirectional=config['bidirectional'],
            use_local_atten=config['use_local_atten'],
            num_heads=config['num_heads']
        )
    elif loss_name == 'gloria':
        return GLoRIALoss(
            temp_global=config['temp_global'],
            temp_local1=config['temp_local1'],
            temp_local2=config['temp_local2'],
            temp_local3=config['temp_local3'],
            lambda_global=config['lambda_global'],
            lambda_local=config['lambda_local'],
            local_agg=config['local_agg']
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, writer=None, use_local_features=False):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        imgs = batch['imgs'].to(device)
        caption_ids = batch['caption_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        batch_on_device = {
            'imgs': imgs,
            'caption_ids': caption_ids,
            'attention_mask': attention_mask
        }
        
        # Forward pass
        if use_local_features:
            # MGCA and GLoRIA require global and local features
            image_features_dict, text_features_dict = get_biomedclip_features_mgca(model, batch_on_device)
            loss, loss_dict = criterion(image_features_dict, text_features_dict)
        else:
            # Simple losses only need global features
            image_features, text_features = get_biomedclip_features(model, batch_on_device)
            loss = criterion(image_features, text_features)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Logging
        total_loss += loss.item()
        if use_local_features:
            postfix = {
                'loss': f'{loss.item():.4f}',
                'global': f"{loss_dict['loss_global']:.4f}",
                'local': f"{loss_dict['loss_local']:.4f}",
            }
            # Add proto loss only if it exists (MGCA has it, GLoRIA doesn't)
            if 'loss_proto' in loss_dict:
                postfix['proto'] = f"{loss_dict['loss_proto']:.4f}"
            progress_bar.set_postfix(postfix)
        else:
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # TensorBoard logging
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss_step', loss.item(), global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            if use_local_features:
                writer.add_scalar('train/loss_global', loss_dict['loss_global'], global_step)
                writer.add_scalar('train/loss_local', loss_dict['loss_local'], global_step)
                if 'loss_proto' in loss_dict:
                    writer.add_scalar('train/loss_proto', loss_dict['loss_proto'], global_step)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            imgs = batch['imgs'].to(device)
            caption_ids = batch['caption_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            batch_on_device = {
                'imgs': imgs,
                'caption_ids': caption_ids,
                'attention_mask': attention_mask
            }
            
            image_features, text_features = get_biomedclip_features(model, batch_on_device)
            loss = criterion(image_features, text_features)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    config['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Setup TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE:
        log_dir = output_dir / 'logs'
        writer = SummaryWriter(log_dir)
        print(f"[INFO] TensorBoard logs: {log_dir}")
    
    # Load model
    model, tokenizer, processor = load_biomedclip(args.model_checkpoint, device)
    
    # Create datasets
    train_dataset = TumorDataset(args.train_dir, processor, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if args.val_dir:
        val_dataset = TumorDataset(args.val_dir, processor, tokenizer)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Create loss function
    criterion = create_loss_function(args.loss).to(device)
    print(f"[INFO] Using loss function: {args.loss}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print(f"\n[INFO] Starting training for {args.epochs} epochs...")
    print(f"[INFO] Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    best_val_loss = float('inf')
    use_local_features = (args.loss in ['mgca', 'gloria'])  # Both need local features
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, writer, use_local_features)
        print(f"\n[Epoch {epoch}/{args.epochs}] Train Loss: {train_loss:.4f}")
        
        if writer:
            writer.add_scalar('train/loss_epoch', train_loss, epoch)
        
        # Validate
        if val_loader:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"[Epoch {epoch}/{args.epochs}] Val Loss: {val_loss:.4f}")
            
            if writer:
                writer.add_scalar('val/loss', val_loss, epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, output_dir / 'best_model.pt')
                print(f"[INFO] Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}_biomedclip.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[INFO] Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\n[INFO] Training complete! Final model saved to: {final_path}")
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
