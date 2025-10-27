# ENGRF: Endpoint-Neutral Gauge Rectified Flow

A PyTorch implementation of Endpoint-Neutral Gauge Rectified Flow (ENGRF) for image restoration and reconstruction tasks, particularly designed for MRI reconstruction and general image enhancement.

## Overview

ENGRF is a three-stage training approach that combines:
- **Stage 0**: Posterior Mean (PM) training using SwinIR or UNet
- **Stage 1**: Rectified Flow (RF) training 
- **Stage 2**: Gauge Field training for endpoint-neutral flow

The method is particularly effective for tasks like MRI reconstruction where you need to recover high-quality images from degraded measurements.

## Project Structure

```
ENGRF/
├── configs/                 # Configuration files
│   ├── config.yaml         # Main configuration
│   ├── config_swinir_hourglass.yaml
│   ├── config_swinir_hdit.yaml
│   └── config_unet_baseline.yaml
├── data/                   # Dataset implementations
│   ├── dataset.py          # FastMRI masked dataset
│   └── LFHF_dataset.py     # Low-frequency/High-frequency pair dataset
├── models/                 # Model implementations
│   ├── engrf.py           # Main ENGRF model
│   ├── posterior_mean.py  # Posterior mean networks
│   ├── rectified_flow.py  # Rectified flow networks
│   ├── gauge.py          # Gauge field implementation
│   ├── swinir.py         # SwinIR implementation
│   └── unet.py           # UNet implementation
├── training/              # Training scripts
│   ├── stage0_pm.py       # Stage 0 training
│   ├── stage1.py          # Stage 1 training
│   ├── stage2.py          # Stage 2 training
│   ├── trainer.py         # Training utilities
│   └── losses.py          # Loss functions
├── util/                  # Utility functions
│   ├── checkpoint.py      # Checkpoint management
│   └── seed.py           # Random seed utilities
├── train.py              # Main training script
├── infer.py              # Inference script
└── validate_pmrf.py      # Validation script
```

## Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Python Dependencies
Install the required packages:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch>=2.0.0` - PyTorch framework
- `torchvision>=0.15.0` - Computer vision utilities
- `numpy>=1.21.0` - Numerical computing
- `pillow>=8.0.0` - Image processing
- `pyyaml>=6.0` - Configuration file parsing
- `tqdm>=4.60.0` - Progress bars

## Quick Start

### 1. Data Preparation

Prepare your data according to one of the supported formats:

**FastMRI Format:**
- Place training data in `data/train/HF/`
- Place validation data in `data/val/HF/`
- Update paths in `configs/config.yaml`

**LFHF Pair Format:**
- Place low-frequency images in `data/train/LF/` and `data/val/LF/`
- Place high-frequency images in `data/train/HF/` and `data/val/HF/`
- Update paths in configuration file

### 2. Configuration

Edit `configs/config.yaml` to match your setup:

```yaml
data:
  kind: fastmri_mask  # or lfhf_pair
  train_root: /path/to/train/data
  val_root: /path/to/val/data
  img_size: [256, 256]

model:
  posterior_mean: swinir  # or unet
  pmrf_only: false  # Set to true for stages 0+1 only

train:
  batch_size: 1
  epochs_stage0: 100
  epochs_stage1: 100
  epochs_stage2: 100
```

### 3. Training

Train the model in stages:

```bash
# Stage 0: Train posterior mean
python train.py --config configs/config.yaml --stage 0

# Stage 1: Train rectified flow
python train.py --config configs/config.yaml --stage 1 --ckpt outputs/engrf_abs/stage0_pm.pt

# Stage 2: Train gauge field
python train.py --config configs/config.yaml --stage 2 --ckpt outputs/engrf_abs/stage1.pt
```

### 4. Inference

Run inference on your data:

```bash
# Basic inference
python infer.py --config configs/config.yaml --ckpt outputs/engrf_abs/stage2.pt

# Inference with visualization
python infer.py --config configs/config.yaml --ckpt outputs/engrf_abs/stage2.pt --save_panels

# Compare different stages
python infer.py --config configs/config.yaml --ckpt outputs/engrf_abs/stage2.pt --stages both
```

## Configuration Options

### Model Architecture
- `posterior_mean`: Choose between `swinir` or `unet`
- `rf_unet.arch`: Choose between `unet`, `hdit`, or `hourglass`
- `pmrf_only`: Set to `true` to use only stages 0+1 (PM+RF)

### Training Parameters
- `batch_size`: Batch size for training
- `lr_pm`, `lr_stage1`, `lr_stage2`: Learning rates for each stage
- `epochs_stage0`, `epochs_stage1`, `epochs_stage2`: Number of epochs per stage
- `weight_decay`: L2 regularization
- `grad_clip`: Gradient clipping threshold

### Data Configuration
- `img_size`: Target image size [height, width]
- `resize_to`: Alternative resize configuration
- `resize_mode`: Interpolation mode (`bilinear`, `nearest`, `bicubic`)
- `center_fractions_tr/va`: Center fractions for FastMRI masking
- `accelerations_tr/va`: Acceleration factors for FastMRI masking

## Advanced Usage

### Custom Datasets

To use your own dataset, implement a custom dataset class following the pattern in `data/dataset.py`:

```python
class CustomDataset(Dataset):
    def __init__(self, root, **kwargs):
        # Your initialization code
        pass
    
    def __getitem__(self, idx):
        # Return dict with 'x' (clean) and 'y' (degraded) keys
        return {'x': clean_image, 'y': degraded_image}
```

### Model Customization

You can customize the model architecture by modifying the configuration:

```yaml
model:
  pm_swinir:
    embed_dim: 60
    depths: [3, 3, 3, 3, 3]
    num_heads: [3, 3, 3, 3, 3]
  
  rf_unet:
    arch: hourglass
    mapping:
      depth: 2
      width: 256
    levels:
      - depth: 4
        width: 256
        self_attn_type: neighborhood
```

## Performance Tips

1. **GPU Memory**: Reduce `batch_size` if you encounter out-of-memory errors
2. **Training Speed**: Use `amp: true` in config for automatic mixed precision
3. **Data Loading**: Adjust `num_workers` based on your CPU cores
4. **Inference Speed**: Use `steps=50` for good quality-speed tradeoff

## Troubleshooting

### Common Issues

**Out of Memory Error:**
- Reduce `batch_size` in config
- Use gradient checkpointing
- Reduce image size with `img_size`

**Slow Training:**
- Increase `num_workers` for data loading
- Use `persistent_workers: true`
- Enable mixed precision with `amp: true`

**Poor Results:**
- Check data preprocessing
- Verify configuration parameters
- Ensure sufficient training epochs

### Getting Help

If you encounter issues:
1. Check the configuration file format
2. Verify data paths and formats
3. Ensure all dependencies are installed
4. Check GPU memory usage

## Citation

If you use this code in your research, please cite the original ENGRF paper:

```bibtex
@article{engrf2024,
  title={Endpoint-Neutral Gauge Rectified Flow},
  author={Your Name},
  journal={Conference/Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
