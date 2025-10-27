#!/bin/bash
# Script to compare ENGRF inference with different stage configurations

# Default paths - modify these as needed
CONFIG=${1:-"configs/config_swinir_hourglass.yaml"}
CKPT=${2:-"outputs/engrf_swinir_hourglass/ckpts_stage1/best_stage1_ep006.pt"}
OUTPUT_DIR=${3:-"inference_comparison"}

echo "=========================================="
echo "ENGRF Stage Comparison Script"
echo "=========================================="
echo "Config: $CONFIG"
echo "Checkpoint: $CKPT"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run comparison mode
python infer.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --save_dir "$OUTPUT_DIR" \
    --stages both \
    --steps 50 \
    --batch_size 4 \
    --num_workers 4 \
    --save_panels \
    --device cuda

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

