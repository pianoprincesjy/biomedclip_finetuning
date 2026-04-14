#!/bin/bash
# Test fine-tuned BiomedCLIP model

CHECKPOINT="checkpoints/clip_20260414_120000/best_model.pt"
IMAGE_DIR="/home/jaey00ns/MedCLIP-SAMv2-main/data/breast_tumors/ttrain"
GPU_ID=0

echo "========================================================================"
echo "Testing Fine-tuned BiomedCLIP"
echo "========================================================================"
echo "Checkpoint:   $CHECKPOINT"
echo "Image dir:    $IMAGE_DIR"
echo "GPU:          $GPU_ID"
echo "========================================================================"

CUDA_VISIBLE_DEVICES=$GPU_ID python test.py \
    --checkpoint "$CHECKPOINT" \
    --image-dir "$IMAGE_DIR" \
    --pattern "*.png" \
    --gpu 0

echo ""
echo "========================================================================"
echo "Testing complete!"
echo "========================================================================"
