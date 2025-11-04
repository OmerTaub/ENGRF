# training/stage0_pm.py
from __future__ import annotations
import os, math, logging, argparse
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
from util.checkpoint import load_latest_checkpoint, save_ckpt
from models.engrf import ENGRFAbs

try:
    import wandb
    WANDB = True
except Exception:
    WANDB = False

logger = logging.getLogger(__name__)

# --------------------------- small helpers --------------------------- #

def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(a, b)

@torch.no_grad()
def _count_params(module: torch.nn.Module, trainable_only: bool = True) -> int:
    ps = (p for p in module.parameters() if (p.requires_grad or not trainable_only))
    return sum(p.numel() for p in ps)

@torch.no_grad()
def _psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute PSNR per image in batch. Returns (B,) tensor."""
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    return 10 * torch.log10(data_range ** 2 / (mse + 1e-8))

@torch.no_grad()
def _ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute SSIM per image in batch. Returns (B,) tensor."""
    from torch.nn.functional import avg_pool2d
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu_x = avg_pool2d(pred, 7, 1, 3)
    mu_y = avg_pool2d(target, 7, 1, 3)
    sigma_x = avg_pool2d(pred * pred, 7, 1, 3) - mu_x ** 2
    sigma_y = avg_pool2d(target * target, 7, 1, 3) - mu_y ** 2
    sigma_xy = avg_pool2d(pred * target, 7, 1, 3) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean(dim=[1, 2, 3])

@torch.no_grad()
def _nmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Normalized MSE per image in batch. Returns (B,) tensor."""
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    target_power = torch.mean(target ** 2, dim=[1, 2, 3])
    return mse / (target_power + 1e-8)

# Optional tiny viz utils (same API as stage1)
import numpy as np
from PIL import Image

def _fft2c(xc):  # centered 2D FFT, xc: complex tensor HxW
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(xc, dim=(-2, -1)),
                                             norm="ortho"), dim=(-2, -1))

def _to_kspace_mag(img_1hw: torch.Tensor) -> np.ndarray:
    """
    img_1hw: torch.Tensor (1,H,W) in [0,1] (real)
    returns: numpy HxW log-magnitude of centered k-space
    """
    x = img_1hw.squeeze(0).contiguous()          # (H,W)
    z = torch.complex(x, torch.zeros_like(x))    # real -> complex
    K = _fft2c(z)                                # (H,W), complex
    mag = torch.abs(K)
    mag = torch.log1p(mag)                       # log visualization
    mag = mag.cpu().float().numpy()
    mag /= (mag.max() + 1e-12)                   # normalize 0..1 for display
    return mag

def _to_uint8(img: torch.Tensor) -> np.ndarray:
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

def _to_kspace_mag_masked(img_1hw: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """
    img_1hw: torch.Tensor (1,H,W) in [0,1] (real)
    mask: torch.Tensor (W,) boolean mask
    returns: numpy HxW log-magnitude of original undersampled k-space
    """
    x = img_1hw.squeeze(0).contiguous()          # (H,W)
    z = torch.complex(x, torch.zeros_like(x))    # real -> complex
    K_full = _fft2c(z)                           # (H,W), complex - full k-space
    
    # Apply the original undersampling mask
    mask_2d = mask.unsqueeze(0).expand(K_full.shape[-2], -1)  # (H,W)
    K_masked = torch.where(mask_2d, K_full, torch.zeros_like(K_full))
    
    mag = torch.abs(K_masked)
    mag = torch.log1p(mag)                       # log visualization
    mag = mag.cpu().float().numpy()
    mag /= (mag.max() + 1e-12)                   # normalize 0..1 for display
    return mag

def _save_triptych(y: torch.Tensor, x_pred: torch.Tensor, x_gt: torch.Tensor, mask: torch.Tensor, path: str):
    """
    Save visualization with 2 rows x 3 columns:
    Row 1: [Undersampled | Posterior Mean | GT]
    Row 2: [Original Undersampled k-space | Posterior Mean k-space | GT k-space]
    With title band on top.
    """
    from PIL import ImageDraw
    
    # Convert to numpy and normalize together to preserve relative relationships
    y_np = y.squeeze(0).cpu().numpy() if y.dim() == 3 else y.cpu().numpy()
    pred_np = x_pred.squeeze(0).cpu().numpy() if x_pred.dim() == 3 else x_pred.cpu().numpy()
    gt_np = x_gt.squeeze(0).cpu().numpy() if x_gt.dim() == 3 else x_gt.cpu().numpy()
    
    # Stack all images and normalize together
    all_images = np.stack([y_np, pred_np, gt_np], axis=0)
    all_min = all_images.min()
    all_max = all_images.max()
    
    if all_max > all_min:
        all_images = (all_images - all_min) / (all_max - all_min)
    else:
        all_images = np.zeros_like(all_images)
    
    # Convert to uint8
    yi = (all_images[0] * 255.0).round().astype(np.uint8)
    pi = (all_images[1] * 255.0).round().astype(np.uint8)
    gi = (all_images[2] * 255.0).round().astype(np.uint8)
    
    # Convert to RGB if needed
    if yi.ndim == 2: yi = np.stack([yi]*3, axis=-1)
    if pi.ndim == 2: pi = np.stack([pi]*3, axis=-1)
    if gi.ndim == 2: gi = np.stack([gi]*3, axis=-1)
    
    # Generate k-space visualizations
    # For undersampled: use original masked k-space
    y_kspace = _to_kspace_mag_masked(y.unsqueeze(0) if y.dim() == 2 else y, mask)
    # For posterior mean and GT: compute k-space normally
    pred_kspace = _to_kspace_mag(x_pred.unsqueeze(0) if x_pred.dim() == 2 else x_pred)
    gt_kspace = _to_kspace_mag(x_gt.unsqueeze(0) if x_gt.dim() == 2 else x_gt)
    
    # Convert k-space to uint8 RGB
    y_k = (y_kspace * 255.0).round().astype(np.uint8)
    pred_k = (pred_kspace * 255.0).round().astype(np.uint8)
    gt_k = (gt_kspace * 255.0).round().astype(np.uint8)
    
    if y_k.ndim == 2: y_k = np.stack([y_k]*3, axis=-1)
    if pred_k.ndim == 2: pred_k = np.stack([pred_k]*3, axis=-1)
    if gt_k.ndim == 2: gt_k = np.stack([gt_k]*3, axis=-1)
    
    # Create image row and k-space row
    img_row = np.concatenate([yi, pi, gi], axis=1)
    kspace_row = np.concatenate([y_k, pred_k, gt_k], axis=1)
    
    # Stack rows vertically
    combined = np.concatenate([img_row, kspace_row], axis=0)
    
    # Add title band
    h, w = combined.shape[:2]
    tiles = 3
    tile_w = w // tiles
    band_h = 24
    canvas = np.ones((h + band_h, w, 3), dtype=np.uint8) * 255
    canvas[band_h:, :, :] = combined
    
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    labels = ("Undersampled", "Posterior Mean", "GT")
    for i, text in enumerate(labels):
        draw.text((i*tile_w + 5, 4), text, fill=(0,0,0))
    
    img.save(path)

# ----------------------------- core step ----------------------------- #

def _pm_step(model: ENGRFAbs, batch: Dict[str, torch.Tensor], amp: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Stage-0 supervised PM training:
        x_star = f_psi(y)
        loss = || x_star - x ||^2
    """
    device = next(model.parameters()).device
    x = batch["x"].to(device)  # (B,C,H,W)
    y = batch["y"].to(device)  # (B,C,H,W) or (B,1,H,W)
    with autocast(enabled=amp):
        x_star = model.pm(y)
        loss = _mse(x_star, x)
    return loss, {"pm_loss": float(loss.detach().cpu())}

# ---------------------------- public API ----------------------------- #

def train_stage0_pm(
    cfg: Dict,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    device: str = "cuda",
    pretrained: Optional[ENGRFAbs] = None,
    args: argparse.Namespace = None,
) -> ENGRFAbs:
    """
    Train posterior-mean backbone f_psi (Stage-0).
    Returns ENGRFAbs with trained PM, RF/Gauge untouched (frozen here).
    """
    trn = cfg.get("train", {})
    exp = cfg.get("experiment", {})

    out_dir = trn["save_dir"] # exp/runX
    ckpt_dir = os.path.join(out_dir, "ckpts_stage0")
    viz_dir  = os.path.join(out_dir, "viz_stage0")
    viz_val_dir = os.path.join(viz_dir, "val")
    viz_train_dir = os.path.join(viz_dir, "train")
    os.makedirs(viz_val_dir, exist_ok=True)
    os.makedirs(viz_train_dir, exist_ok=True)
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    amp          = bool(trn.get("amp", True))
    epochs       = int(trn.get("epochs_stage0", 50))
    lr           = float(trn.get("lr_pm", 1e-4))
    wd           = float(trn.get("weight_decay", 1e-4))
    grad_clip    = float(trn.get("grad_clip", 1.0))
    log_interval = int(trn.get("log_interval", 50))
    val_every    = int(trn.get("val_every", 1))
    vis_every    = int(trn.get("vis_every", val_every))
    vis_n        = int(trn.get("vis_n", 0))

    # Build or warm-start model
    model = pretrained if pretrained is not None else ENGRFAbs(cfg)
    model = model.to(device)

    # Freeze everything except PM
    for p in model.pm.parameters():      p.requires_grad = True
    for p in model.rf.parameters():      p.requires_grad = False
    for p in model.W.parameters():       p.requires_grad = False
    for p in model.hflow.parameters():   p.requires_grad = False

    n_trainable = _count_params(model.pm, trainable_only=True)
    logger.info(f"[Stage0] Trainable PM params: {n_trainable:,}")

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    scaler = GradScaler(enabled=amp)

    # Resume from latest checkpoint
    if args.resume or args.resume_dir is not None:
        ckpt_resume_dir = os.path.join(args.resume_dir, "ckpts_stage0") if args.resume_dir is not None else ckpt_dir
        logger.info(f"[Stage0] Trying to resume from directory: {ckpt_resume_dir}")
        latest_ckpt, model, opt, global_step, start_epoch = load_latest_checkpoint(ckpt_resume_dir, model, opt, device)
        if global_step == 0 and start_epoch > 1:
            global_step = (start_epoch - 1) * len(train_loader)
        logger.info(f"[Stage0] Resuming from {latest_ckpt} at epoch {start_epoch} (global_step={global_step})")
    else:
        global_step = 0
        start_epoch = 1

    best_val = math.inf

    for ep in range(start_epoch, epochs + 1):
        # ------------------------------ train ------------------------------ #
        model.train()
        pbar = tqdm(train_loader, desc=f"[Stage0-PM] Epoch {ep}/{epochs}", leave=False)
        run_loss, run_cnt = 0.0, 0
        all_train_psnr = []
        all_train_ssim = []

        for it, batch in enumerate(pbar, 1):
            opt.zero_grad(set_to_none=True)

            loss, logs = _pm_step(model, batch, amp=amp)

            if amp:
                scaler.scale(loss).backward()
                # Unscale before clipping so clipping threshold isn't applied to scaled grads
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.pm.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.pm.parameters(), grad_clip)
                opt.step()

            # Compute metrics for display
            with torch.no_grad():
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                x_pm = model.pm(y)
                x_clamp = x.clamp(0, 1)
                x_pm_clamp = x_pm.clamp(0, 1)
                psnr_b = _psnr(x_pm_clamp, x_clamp)
                ssim_b = _ssim(x_pm_clamp, x_clamp)
                all_train_psnr.append(psnr_b.cpu())
                all_train_ssim.append(ssim_b.cpu())

            bs = batch["x"].size(0)
            run_loss += float(loss.detach().cpu()) * bs
            run_cnt  += bs
            global_step += 1

            # Compute running averages
            avg_psnr = torch.cat(all_train_psnr).mean().item() if all_train_psnr else 0
            avg_ssim = torch.cat(all_train_ssim).mean().item() if all_train_ssim else 0

            if (it % log_interval) == 0:
                logger.info(f"[Stage0][Ep {ep}] it={it} pm_loss={(run_loss/max(run_cnt,1)):.6f}")
                if WANDB:
                    wandb.log({
                        "train/pm_loss": (run_loss/max(run_cnt,1)),
                        "epoch": ep,
                        "iter": it,
                        "step": global_step,
                    })
            pbar.set_postfix(loss=f"{(run_loss/max(run_cnt,1)):.4f}", PSNR=f"{avg_psnr:.2f}", SSIM=f"{avg_ssim:.3f}")

        # ----------------------------- validate ---------------------------- #
        if val_loader is not None and (ep % val_every == 0):
            model.eval()
            tot, n = 0.0, 0
            all_psnr = []
            all_ssim = []
            all_nmse = []
            
            with torch.no_grad():
                vbar = tqdm(val_loader, desc=f"[Stage0-PM][Val] Epoch {ep}", leave=False)
                for vb in vbar:
                    vloss, _ = _pm_step(model, vb, amp=False)
                    bs = vb["x"].size(0)
                    tot += float(vloss.detach().cpu()) * bs
                    n   += bs

                    # Calculate PSNR, SSIM, NMSE
                    x = vb["x"].to(device)
                    y = vb["y"].to(device)
                    x_pm = model.pm(y)
                    # Clamp to [0,1] for fair metrics if needed
                    x_clamp = x.clamp(0, 1)
                    x_pm_clamp = x_pm.clamp(0, 1)
                    
                    psnr_b = _psnr(x_pm_clamp, x_clamp)
                    ssim_b = _ssim(x_pm_clamp, x_clamp)
                    nmse_b = _nmse(x_pm_clamp, x_clamp)
                    all_psnr.append(psnr_b.cpu())
                    all_ssim.append(ssim_b.cpu())
                    all_nmse.append(nmse_b.cpu())

                    # Compute running averages for display
                    avg_psnr = torch.cat(all_psnr).mean().item()
                    avg_ssim = torch.cat(all_ssim).mean().item()
                    vbar.set_postfix(loss=f"{(tot/max(n,1)):.4f}", PSNR=f"{avg_psnr:.2f}", SSIM=f"{avg_ssim:.3f}")

            val_avg = tot / max(n, 1)
            psnr_avg = torch.cat(all_psnr).mean().item() if all_psnr else float("nan")
            ssim_avg = torch.cat(all_ssim).mean().item() if all_ssim else float("nan")
            nmse_avg = torch.cat(all_nmse).mean().item() if all_nmse else float("nan")
            logger.info(f"[Stage0-PM][Val] Epoch {ep} Summary:")
            logger.info(f"  Loss: {val_avg:.6f}")
            logger.info(f"  PSNR: {psnr_avg:.3f} dB")
            logger.info(f"  SSIM: {ssim_avg:.4f}")
            logger.info(f"  NMSE: {nmse_avg:.6f}")
            if WANDB:
                wandb.log({
                    "val/pm_loss": val_avg,
                    "val/PSNR": psnr_avg,
                    "val/SSIM": ssim_avg,
                    "val/NMSE": nmse_avg,
                    "epoch": ep,
                })
            

            # Save best
            if val_avg < best_val:
                best_val = val_avg
                path = os.path.join(ckpt_dir, f"best_stage0_ep{ep:03d}.pt")
                save_ckpt(
                    path,
                    model.state_dict(),
                    opt.state_dict(),
                    global_step=global_step,
                    epoch=ep,
                    config=cfg,
                    val_pm_loss=best_val
                )
                logger.info(f"[Stage0] Saved best checkpoint to: {path}")

            # Visualization panel(s)
            if (ep % vis_every == 0) and vis_n > 0:
                model.eval()
                _sample_and_viz(val_loader.dataset, val_loader.batch_size, device, model, vis_n, viz_val_dir, ep, "val")
                _sample_and_viz(train_loader.dataset, train_loader.batch_size, device, model, vis_n, viz_train_dir, ep, "train")


        # ----------------------------- last save --------------------------- #
        if ep == epochs:
            path = os.path.join(ckpt_dir, f"last_stage0_ep{ep:03d}.pt")
            save_ckpt(
                path,
                model.state_dict(),
                opt.state_dict(),
                global_step=global_step,
                epoch=ep,
                config=cfg,
            )
            logger.info(f"[Stage0] Saved last checkpoint to: {path}")

    return model


def _sample_and_viz(dataset, batch_size, device, model, vis_n, out_dir, ep, split):
    # pick up to vis_n random indices across the whole dataset
    n = len(dataset)
    k = min(vis_n, n)
    idxs = random.sample(range(n), k)

    # use a single DataLoader over those indices
    sampler = torch.utils.data.SubsetRandomSampler(idxs)
    tmp_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, sampler=sampler
    )
    saved = 0
    with torch.no_grad():
        for vb in tmp_loader:
            x = vb["x"].to(device)
            y = vb["y"].to(device)
            mask = vb["mask"]  # Get the mask from dataset
            x_star = model.pm(y)
            for i in range(x.size(0)):
                if saved >= vis_n: break
                _save_triptych(
                    y[i].cpu(), x_star[i].cpu(), x[i].cpu(), mask[i].cpu(),
                    os.path.join(out_dir, f"{split}_ep{ep:03d}_idx{saved:02d}.png")
                )
                if WANDB:
                    wandb.log({
                        f"viz/stage0_{split}": wandb.Image(os.path.join(out_dir, f"{split}_ep{ep:03d}_idx{saved:02d}.png"))
                    }, commit=False)
                saved += 1
            if saved >= vis_n: break
