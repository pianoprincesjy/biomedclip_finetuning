#!/usr/bin/env python3
"""
BiomedCLIP Model Wrapper
========================
"""
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor


def load_biomedclip(checkpoint="chuhac/BiomedCLIP-vit-bert-hf", device="cuda"):
    """
    Load BiomedCLIP model, tokenizer, and processor.
    
    Args:
        checkpoint (str): HuggingFace checkpoint name
        device (str): Device to load model on
        
    Returns:
        model: BiomedCLIP model
        tokenizer: Text tokenizer
        processor: Image processor
    """
    print(f"[INFO] Loading BiomedCLIP from {checkpoint}...")
    
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    
    # Get image processor
    if hasattr(processor, 'image_processor'):
        image_processor = processor.image_processor
    else:
        image_processor = processor
    
    model.to(device)
    print(f"[INFO] BiomedCLIP loaded successfully on {device}")
    
    return model, tokenizer, image_processor


def get_biomedclip_features(model, batch):
    """
    Extract image and text features from BiomedCLIP.
    
    Args:
        model: BiomedCLIP model
        batch: Dictionary containing 'imgs', 'caption_ids', 'attention_mask'
        
    Returns:
        image_features: Image embeddings [batch_size, embed_dim]
        text_features: Text embeddings [batch_size, embed_dim]
    """
    # Image features
    image_outputs = model.vision_model(batch['imgs'])
    if hasattr(image_outputs, 'pooler_output'):
        image_features = image_outputs.pooler_output
    else:
        # Use CLS token
        image_features = image_outputs.last_hidden_state[:, 0]
    
    # Text features
    text_outputs = model.text_model(
        input_ids=batch['caption_ids'],
        attention_mask=batch['attention_mask']
    )
    if hasattr(text_outputs, 'pooler_output'):
        text_features = text_outputs.pooler_output
    else:
        # Use CLS token
        text_features = text_outputs.last_hidden_state[:, 0]
    
    return image_features, text_features


def get_biomedclip_features_mgca(model, batch):
    """
    Extract global and local features from BiomedCLIP for MGCA loss.
    
    Args:
        model: BiomedCLIP model
        batch: Dictionary containing 'imgs', 'caption_ids', 'attention_mask'
        
    Returns:
        image_features_dict: Dict with 'global' [B, D] and 'local' [B, N, D]
        text_features_dict: Dict with 'global' [B, D], 'local' [B, L, D], 'attention_mask' [B, L]
    """
    # Image features (global + local)
    image_outputs = model.vision_model(batch['imgs'], output_hidden_states=True)
    
    if hasattr(image_outputs, 'last_hidden_state'):
        hidden_states = image_outputs.last_hidden_state  # [B, 1+N, D]
    else:
        hidden_states = image_outputs[0]
    
    img_global = hidden_states[:, 0]  # CLS token [B, D]
    img_local = hidden_states[:, 1:]  # Patch tokens [B, N, D]
    
    # Text features (global + local)
    text_outputs = model.text_model(
        input_ids=batch['caption_ids'],
        attention_mask=batch['attention_mask'],
        output_hidden_states=True
    )
    
    if hasattr(text_outputs, 'last_hidden_state'):
        text_hidden = text_outputs.last_hidden_state  # [B, L, D]
    else:
        text_hidden = text_outputs[0]
    
    text_global = text_hidden[:, 0]  # CLS token [B, D]
    text_local = text_hidden  # All tokens [B, L, D]
    
    return {
        'global': img_global,
        'local': img_local
    }, {
        'global': text_global,
        'local': text_local,
        'attention_mask': batch['attention_mask']
    }
