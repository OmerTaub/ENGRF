# inference.py
from __future__ import annotations
import os, csv, argparse, yaml, math, time, logging
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw

from models.engrf import ENGRFAbs
from data.dataset import FastMRIMaskedAbsDataset
import tqdm

logger = logging.getLogger(__name__)

def setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

# ----------------------------- I/O helpers ----------------------------- #

def _to_uint8(img: torch.Tensor) -> np.ndarray:
    """
    img: (C,H,W) or (H,W), any range. Returns uint8 HxW or HxWx3.
    """
    if img.dim() == 3 and img.size(0) in (1, 3):
        c, h, w = img.shape
        x = img
    elif img.dim() == 2:
        h, w = img.shape
        c, x = 1, img.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported image shape: {tuple(img.shape)}")

    x = x.float()
    x = x - x.min()
    denom = (x.max() - x.min()).clamp_min(1e-12)
    x = x / denom

    if c == 1:
        arr = (x.squeeze(0).clamp(0,1).cpu().numpy() * 255.0).round().astype(np.uint8)
    else:
        arr = (x.clamp(0,1).permute(1,2,0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return arr


# --- add this new helper next to save_triptych() ---
def save_quad_panel(path: str,
                    y: torch.Tensor,
                    x_pm: torch.Tensor,
                    x_pred: torch.Tensor,
                    x_gt: torch.Tensor,
                    labels=("LF", "Posterior Mean", "ENGRF Output", "GT"),
                    add_titles: bool = False) -> None:
    yi = _to_uint8(y)
    pmi = _to_uint8(x_pm)
    pri = _to_uint8(x_pred)
    gi = _to_uint8(x_gt)

    to_rgb = (yi.ndim == 3) or (pmi.ndim == 3) or (pri.ndim == 3) or (gi.ndim == 3)
    if to_rgb:
        if yi.ndim == 2:  yi  = np.stack([yi]*3,  axis=-1)
        if pmi.ndim == 2: pmi = np.stack([pmi]*3, axis=-1)
        if pri.ndim == 2: pri = np.stack([pri]*3, axis=-1)
        if gi.ndim == 2:  gi  = np.stack([gi]*3,  axis=-1)

    strip = np.concatenate([yi, pmi, pri, gi], axis=1)

    if add_titles:
        h, w = strip.shape[:2]
        tiles = 4
        tile_w = w // tiles
        band_h = 24
        canvas = np.ones((h + band_h, w, 3), dtype=np.uint8) * 255
        canvas[band_h:, :, :] = strip if strip.ndim == 3 else np.stack([strip]*3, axis=-1)
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        for i, text in enumerate(labels):
            draw.text((i * tile_w + 5, 4), text, fill=(0, 0, 0))
        img.save(path)
    else:
        Image.fromarray(strip).save(path)



# ----------------------------- Metrics (Torch) ----------------------------- #

def mse_per_image(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x,y: (B,C,H,W) in [0,1] (or consistent data_range). Returns (B,)
    """
    return ((x - y) ** 2).flatten(1).mean(dim=1)


def mae_per_image(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).abs().flatten(1).mean(dim=1)


def psnr_per_image(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    """
    PSNR = 10 * log10( MAX^2 / MSE ). Returns (B,)
    """
    mse = mse_per_image(x, y).clamp_min(eps)
    return 10.0 * torch.log10((data_range ** 2) / mse)


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    window_1d = g.unsqueeze(0)  # (1, W)
    window_2d = (window_1d.t() @ window_1d).unsqueeze(0).unsqueeze(0)  # (1,1,W,W)
    return window_2d


def ssim_per_image(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    K: Tuple[float, float] = (0.01, 0.03),
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Multi-channel SSIM averaged over channels, computed with depthwise conv.
    x,y: (B,C,H,W). Returns (B,)
    """
    assert x.shape == y.shape and x.dim() == 4
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    # constants
    k1, k2 = K
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # window
    win = _gaussian_window(window_size, sigma, device, dtype)  # (1,1,ws,ws)
    pad = window_size // 2
    # depthwise conv weight: (C,1,ws,ws)
    weight = win.expand(C, 1, window_size, window_size)

    # per-channel means and variances
    mu_x = F.conv2d(x, weight, padding=pad, groups=C)
    mu_y = F.conv2d(y, weight, padding=pad, groups=C)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, weight, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, weight, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, weight, padding=pad, groups=C) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + eps)
    # average over spatial dims then channels
    ssim_c = ssim_map.flatten(2).mean(dim=2)          # (B,C)
    ssim_b = ssim_c.mean(dim=1)                       # (B,)
    return ssim_b


# ----------------------------- Loading ----------------------------- #

def load_pretrained(ckpt_path: str, fallback_cfg: dict, device: str) -> ENGRFAbs:
    """
    Load an ENGRFAbs model from checkpoint.
    The checkpoint should contain either:
      - {"state_dict", "config"} as saved by your trainers, or
      - a raw state_dict.
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("config", ckpt.get("cfg", fallback_cfg))
    model = ENGRFAbs(ckpt_cfg)
    state = ckpt.get("state_dict", ckpt)  # tolerate raw state_dict
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.info(f"[load_pretrained] non-strict load (missing={len(missing)}, unexpected={len(unexpected)})")
    model.to(device)
    model.eval()
    return model


# ----------------------------- Runner ----------------------------- #

def run_inference(
    cfg: Dict,
    ckpt_path: str,
    device: str = "cuda",
    batch_size: int = 4,
    num_workers: int = 4,
    steps: int = 50,
    save_dir: Optional[str] = None,
    save_panels: bool = False,
    data_range: float = 1.0,
    use_pmrf_only: bool = False,
    override_resize: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """
    Runs ENGRF sampling over the validation set and computes PSNR/SSIM/MSE/MAE.
    Returns dict of averaged metrics; also writes a CSV with per-image metrics.
    
    Args:
        cfg: Configuration dictionary
        ckpt_path: Path to checkpoint file
        device: Device to run on ('cuda' or 'cpu')
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        steps: Number of ODE integration steps
        save_dir: Directory to save results
        save_panels: Whether to save visualization panels
        data_range: Data range for metrics computation
        use_pmrf_only: If True, uses only stages 0+1 (PM+RF). If False, uses full stages 0+1+2 (ENGRF with gauge)
        override_resize: Optional tuple (H, W) to override config resize settings. If None, uses config settings
    
    Returns:
        Dictionary with average metrics (PSNR, SSIM, MSE, MAE)
    """
    os.makedirs(save_dir or "inference_outputs", exist_ok=True)
    out_dir = save_dir or "inference_outputs"
    csv_path = os.path.join(out_dir, "metrics.csv")

    # Dataset / loader (mirror training val split usage)
    # Get resize configuration - support both img_size and resize_to (same as training)
    if override_resize is not None:
        resize_to = tuple(override_resize)
        logger.info(f"[Inference] Using override resize: {resize_to}")
    else:
        resize_to = cfg["data"].get("resize_to", None) or cfg["data"].get("img_size", None)
        if resize_to is not None:
            resize_to = tuple(resize_to)
            logger.info(f"[Inference] Using config resize: {resize_to}")
        else:
            logger.info("[Inference] Using original image sizes (no resizing)")
    resize_mode = cfg["data"].get("resize_mode", "bilinear")
    
    val_ds = FastMRIMaskedAbsDataset(
        cfg["data"]["train_root"], #         cfg["data"]["val_root"], # overfitting
        center_fractions=tuple(cfg["data"].get("center_fractions_va", (0.04,))),
        accelerations=tuple(cfg["data"].get("accelerations_va", (4,))),
        seed=int(cfg["data"].get("seed_va", 123)),
        resize_to=resize_to,
        resize_mode=resize_mode,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # Model
    model = load_pretrained(ckpt_path, cfg, device)
    
    # Set inference mode (stages 0+1 only vs full 0+1+2)
    model.pmrf_only = use_pmrf_only
    mode_str = "Stages 0+1 (PM+RF)" if use_pmrf_only else "Stages 0+1+2 (Full ENGRF)"
    logger.info(f"=== Running inference with: {mode_str} ===")

    # Metrics accumulators
    all_psnr = []
    all_ssim = []
    all_mse  = []
    all_mae  = []

    # CSV header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "psnr", "ssim", "mse", "mae"])

    t0 = time.time()
    idx_base = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader):
            x = batch["x"].to(device)  # GT
            y = batch["y"].to(device)  # measurement/degraded

            # Posterior mean (for visualization only)
            x_pm = model.pm(y).clamp(0, 1)

            # ENGRF deterministic sample
            x_pred = model.sample(y, steps=steps)

            # Clamp for fair metrics if your data live in [0,1]
            x_clamp     = x.clamp(0, 1)
            x_pred_post = x_pred.clamp(0, 1)

            # Compute metrics per-image
            psnr_b = psnr_per_image(x_pred_post, x_clamp, data_range=data_range)    # (B,)
            ssim_b = ssim_per_image(x_pred_post, x_clamp, data_range=data_range)    # (B,)
            mse_b  = mse_per_image(x_pred_post, x_clamp)                             # (B,)
            mae_b  = mae_per_image(x_pred_post, x_clamp)                             # (B,)

            # Accumulate
            all_psnr.append(psnr_b.cpu())
            all_ssim.append(ssim_b.cpu())
            all_mse.append(mse_b.cpu())
            all_mae.append(mae_b.cpu())

            # Save metrics to CSV row-wise
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for i in range(psnr_b.numel()):
                    writer.writerow([
                        idx_base + i,
                        float(psnr_b[i]),
                        float(ssim_b[i]),
                        float(mse_b[i]),
                        float(mae_b[i]),
                    ])

            # Optional panels
            if save_panels:
                B = x.size(0)
                for i in range(B):
                    out_path = os.path.join(out_dir, f"val_idx{idx_base + i:06d}.png")
                    save_quad_panel(
                        out_path,
                        y[i].cpu(),
                        x_pm[i].cpu(),
                        x_pred_post[i].cpu(),
                        x_clamp[i].cpu(),
                        labels=("LF", "Posterior Mean", "ENGRF Output", "GT"),
                        add_titles=False,
                    )

            idx_base += x.size(0)

    # Final averages
    psnr_avg = torch.cat(all_psnr).mean().item() if all_psnr else float("nan")
    ssim_avg = torch.cat(all_ssim).mean().item() if all_ssim else float("nan")
    mse_avg  = torch.cat(all_mse).mean().item() if all_mse else float("nan")
    mae_avg  = torch.cat(all_mae).mean().item() if all_mae else float("nan")
    elapsed  = time.time() - t0

    # Log summary and also write a small text file
    summary = {
        "PSNR": psnr_avg,
        "SSIM": ssim_avg,
        "MSE": mse_avg,
        "MAE": mae_avg,
        "images": idx_base,
        "seconds": elapsed,
    }
    logger.info("ENGRF Inference (Validation)")
    for k, v in summary.items():
        logger.info(f"{k:>8}: {v:.6f}" if isinstance(v, float) else f"{k:>8}: {v}")

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    return summary


# ----------------------------- CLI ----------------------------- #

def main():
    setup_logging(verbosity=1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Path to YAML config (same format as training).",default =f"/home/omertaub/projects/ENGRF/configs/config.yaml")
    ap.add_argument("--ckpt", help="Path to a trained checkpoint (Stage-2 recommended).", default= f"/home/omertaub/projects/ENGRF/outputs_1/engrf_abs/ckpts_stage2/best_stage2_ep006.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50, help="ODE steps for ENGRF sampler.")
    ap.add_argument("--save_dir", default="inference_outputs")
    ap.add_argument("--save_panels", action="store_true", help="Save triptych [LF|Pred|GT] panels.",default = True)
    ap.add_argument("--data_range", type=float, default=1.0, help="Max intensity for PSNR/SSIM constants.")
    ap.add_argument("--stages", type=str, choices=["01", "012", "both"], default="012",
                    help="Which stages to use: '01' (PM+RF), '012' (Full ENGRF), 'both' (run both and compare)")
    ap.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"), default=None,
                    help="Override image resize (H W). If not specified, uses config settings or original size")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.stages == "both":
        # Run both modes and save comparison
        logger.info("COMPARISON MODE: Running inference with both stage configurations")
        
        # Run with stages 0+1 only
        save_dir_01 = os.path.join(args.save_dir, "stages_0+1")
        os.makedirs(save_dir_01, exist_ok=True)
        results_01 = run_inference(
            cfg=cfg,
            ckpt_path=args.ckpt,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            steps=args.steps,
            save_dir=save_dir_01,
            save_panels=args.save_panels,
            data_range=args.data_range,
            use_pmrf_only=True,
            override_resize=tuple(args.resize) if args.resize else None,
        )
        
        # Run with stages 0+1+2
        save_dir_012 = os.path.join(args.save_dir, "stages_0+1+2")
        os.makedirs(save_dir_012, exist_ok=True)
        results_012 = run_inference(
            cfg=cfg,
            ckpt_path=args.ckpt,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            steps=args.steps,
            save_dir=save_dir_012,
            save_panels=args.save_panels,
            data_range=args.data_range,
            use_pmrf_only=False,
            override_resize=tuple(args.resize) if args.resize else None,
        )
        
        # Print comparison
        logger.info("COMPARISON RESULTS")
        logger.info(f"{'Metric':<15} {'Stages 0+1 (PM+RF)':<25} {'Stages 0+1+2 (ENGRF)':<25} {'Improvement':<15}")
        
        for metric in ["PSNR", "SSIM", "MSE", "MAE"]:
            val_01 = results_01[metric]
            val_012 = results_012[metric]
            
            # Calculate improvement (higher is better for PSNR/SSIM, lower is better for MSE/MAE)
            if metric in ["PSNR", "SSIM"]:
                improvement = val_012 - val_01
                sign = "+" if improvement > 0 else ""
                perc = (improvement / val_01 * 100) if val_01 != 0 else 0
                imp_str = f"{sign}{improvement:.4f} ({sign}{perc:.2f}%)"
            else:  # MSE, MAE
                improvement = val_01 - val_012
                sign = "+" if improvement > 0 else ""
                perc = (improvement / val_01 * 100) if val_01 != 0 else 0
                imp_str = f"{sign}{improvement:.6f} ({sign}{perc:.2f}%)"
            
            logger.info(f"{metric:<15} {val_01:<25.6f} {val_012:<25.6f} {imp_str:<15}")
        
        logger.info("Results saved to:")
        logger.info(f"  Stages 0+1:   {save_dir_01}")
        logger.info(f"  Stages 0+1+2: {save_dir_012}")
        
        # Save comparison to file
        comparison_path = os.path.join(args.save_dir, "comparison.txt")
        with open(comparison_path, "w") as f:
            f.write("="*80 + "\n")
            f.write("ENGRF STAGE COMPARISON\n")
            f.write("="*80 + "\n\n")
            f.write(f"{'Metric':<15} {'Stages 0+1 (PM+RF)':<25} {'Stages 0+1+2 (ENGRF)':<25} {'Improvement':<15}\n")
            f.write("-"*80 + "\n")
            
            for metric in ["PSNR", "SSIM", "MSE", "MAE"]:
                val_01 = results_01[metric]
                val_012 = results_012[metric]
                
                if metric in ["PSNR", "SSIM"]:
                    improvement = val_012 - val_01
                    sign = "+" if improvement > 0 else ""
                    perc = (improvement / val_01 * 100) if val_01 != 0 else 0
                    imp_str = f"{sign}{improvement:.4f} ({sign}{perc:.2f}%)"
                else:
                    improvement = val_01 - val_012
                    sign = "+" if improvement > 0 else ""
                    perc = (improvement / val_01 * 100) if val_01 != 0 else 0
                    imp_str = f"{sign}{improvement:.6f} ({sign}{perc:.2f}%)"
                
                f.write(f"{metric:<15} {val_01:<25.6f} {val_012:<25.6f} {imp_str:<15}\n")
            
            f.write("-"*80 + "\n")
            f.write(f"\nCheckpoint: {args.ckpt}\n")
            f.write(f"Sampling steps: {args.steps}\n")
        
        logger.info(f"Comparison summary saved to: {comparison_path}")
        
    else:
        # Run single mode
        use_pmrf = (args.stages == "01")
        os.makedirs(args.save_dir, exist_ok=True)
        
        run_inference(
            cfg=cfg,
            ckpt_path=args.ckpt,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            steps=args.steps,
            save_dir=args.save_dir,
            save_panels=args.save_panels,
            data_range=args.data_range,
            use_pmrf_only=use_pmrf,
            override_resize=tuple(args.resize) if args.resize else None,
        )


if __name__ == "__main__":
    # CUDNN settings for reproducible yet performant inference
    torch.backends.cudnn.benchmark = True
    main()
