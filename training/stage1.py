# training/stage1.py
from __future__ import annotations
import os, math, logging, argparse
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from util.checkpoint import load_latest_checkpoint, save_ckpt
from models.engrf import ENGRFAbs
try:
    import wandb
    WANDB = True
except Exception:
    WANDB = False

logger = logging.getLogger(__name__)


# ------------------------------- Utils -------------------------------- #

def _sample_t(b: int, device: torch.device, eps: float = 0.0) -> torch.Tensor:
    """
    t ~ U[eps, 1-eps], returned as (B,1,1,1).
    A small eps (e.g., 1e-3) avoids degenerate endpoints if desired.
    """
    if eps < 0.0 or eps >= 0.5:
        raise ValueError("eps must be in [0, 0.5).")
    return torch.rand(b, 1, 1, 1, device=device) * (1.0 - 2 * eps) + eps

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


# ------------------------- Tiny visualization utils ------------------------- #
import numpy as np
from PIL import Image, ImageDraw

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
    else:  # c == 3
        arr = (x.clamp(0,1).permute(1,2,0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return arr

def save_quad_panel(
    path: str,
    lf: torch.Tensor,
    pm: torch.Tensor,
    rf: torch.Tensor,
    gt: torch.Tensor,
    labels: tuple[str, str, str, str] = ("Undersampled", "Posterior Mean", "Rectified Flow", "GT"),
    add_titles: bool = True,
) -> None:
    """
    Save visualization with 2 rows x 4 columns:
    Row 1: [Undersampled | Posterior Mean | Rectified Flow | GT]
    Row 2: [k-space | k-space | k-space | k-space]
    With title band on top.
    """
    # Convert to numpy and normalize together to preserve relative relationships
    lf_np = lf.squeeze(0).cpu().numpy() if lf.dim() == 3 else lf.cpu().numpy()
    pm_np = pm.squeeze(0).cpu().numpy() if pm.dim() == 3 else pm.cpu().numpy()
    rf_np = rf.squeeze(0).cpu().numpy() if rf.dim() == 3 else rf.cpu().numpy()
    gt_np = gt.squeeze(0).cpu().numpy() if gt.dim() == 3 else gt.cpu().numpy()
    
    # Stack all images and normalize together
    all_images = np.stack([lf_np, pm_np, rf_np, gt_np], axis=0)
    all_min = all_images.min()
    all_max = all_images.max()
    
    if all_max > all_min:
        all_images = (all_images - all_min) / (all_max - all_min)
    else:
        all_images = np.zeros_like(all_images)
    
    # Convert to uint8
    lf_u8 = (all_images[0] * 255.0).round().astype(np.uint8)
    pm_u8 = (all_images[1] * 255.0).round().astype(np.uint8)
    rf_u8 = (all_images[2] * 255.0).round().astype(np.uint8)
    gt_u8 = (all_images[3] * 255.0).round().astype(np.uint8)

    # harmonize to RGB if any is RGB
    to_rgb = (lf_u8.ndim == 3) or (pm_u8.ndim == 3) or (rf_u8.ndim == 3) or (gt_u8.ndim == 3)
    if to_rgb:
        if lf_u8.ndim == 2: lf_u8 = np.stack([lf_u8]*3, axis=-1)
        if pm_u8.ndim == 2: pm_u8 = np.stack([pm_u8]*3, axis=-1)
        if rf_u8.ndim == 2: rf_u8 = np.stack([rf_u8]*3, axis=-1)
        if gt_u8.ndim == 2: gt_u8 = np.stack([gt_u8]*3, axis=-1)
    
    # Generate k-space visualizations
    lf_kspace = _to_kspace_mag(lf.unsqueeze(0) if lf.dim() == 2 else lf)
    pm_kspace = _to_kspace_mag(pm.unsqueeze(0) if pm.dim() == 2 else pm)
    rf_kspace = _to_kspace_mag(rf.unsqueeze(0) if rf.dim() == 2 else rf)
    gt_kspace = _to_kspace_mag(gt.unsqueeze(0) if gt.dim() == 2 else gt)
    
    # Convert k-space to uint8 RGB
    lf_k = (lf_kspace * 255.0).round().astype(np.uint8)
    pm_k = (pm_kspace * 255.0).round().astype(np.uint8)
    rf_k = (rf_kspace * 255.0).round().astype(np.uint8)
    gt_k = (gt_kspace * 255.0).round().astype(np.uint8)
    
    if lf_k.ndim == 2: lf_k = np.stack([lf_k]*3, axis=-1)
    if pm_k.ndim == 2: pm_k = np.stack([pm_k]*3, axis=-1)
    if rf_k.ndim == 2: rf_k = np.stack([rf_k]*3, axis=-1)
    if gt_k.ndim == 2: gt_k = np.stack([gt_k]*3, axis=-1)
    
    # Create image row and k-space row
    img_row = np.concatenate([lf_u8, pm_u8, rf_u8, gt_u8], axis=1)
    kspace_row = np.concatenate([lf_k, pm_k, rf_k, gt_k], axis=1)

    # Ensure both rows are 3D before vertical stacking
    if img_row.ndim == 2:
        img_row = np.stack([img_row] * 3, axis=-1)
    if kspace_row.ndim == 2:
        kspace_row = np.stack([kspace_row] * 3, axis=-1)
    
    # Stack rows vertically
    combined = np.concatenate([img_row, kspace_row], axis=0)

    if add_titles:
        h, w = combined.shape[:2]
        tiles = 4
        tile_w = w // tiles
        band_h = 24
        canvas = np.ones((h + band_h, w, 3), dtype=np.uint8) * 255
        canvas[band_h:, :, :] = combined if combined.ndim == 3 else np.stack([combined]*3, axis=-1)
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        for i, text in enumerate(labels):
            draw.text((i*tile_w + 5, 4), text, fill=(0,0,0))
        img.save(path)
    else:
        Image.fromarray(combined).save(path)


# ----------------------------- Core FM step ----------------------------- #

def _fm_step(
    model: ENGRFAbs,
    batch: Dict[str, torch.Tensor],
    amp: bool = True,
    eps_t: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    One training/validation step for Stage-1 FM.

    Math (rectified flow FM, unconditional):
        X* = f_psi(Y)           (posterior mean, psi frozen; no grad)
        Z0 = X* + σ_s * ε       (stochastic starting point, ε ~ N(0,I))
        Δ  = X - Z0             (velocity target)
        Z_t = (1 - t)·Z0 + t·X  (linear interpolation path)
        loss = E || v_theta(Z_t, t) - Δ ||^2
    
    The noise σ_s prevents the model from learning a deterministic mapping
    and enables the flow to capture the conditional distribution p(X|Y).
    """
    device = next(model.parameters()).device
    x = batch["x"].to(device)  # (B,C,H,W) clean target
    y = batch["y"].to(device)  # (B,C,H,W) degraded input
    B = x.size(0)
    t = _sample_t(B, device=device, eps=eps_t)  # (B,1,1,1) ~ U[eps, 1-eps]

    with torch.no_grad():
        model.pm.eval()
        x_star = model.pm(y)  # Posterior mean X* = f_psi(Y)
        
        # Create stochastic starting point Z0 = X* + σ_s·ε
        if model.pmrf_sigma_s > 0:
            Z0 = x_star + torch.randn_like(x_star) * model.pmrf_sigma_s
        else:
            Z0 = x_star
        
        # Velocity target: Δ = X - Z0
        delta = x - Z0

    # Interpolation path: Z_t = (1-t)·Z0 + t·X
    # At t=0: Z_t = Z0 (starting point)
    # At t=1: Z_t = X  (target point)
    Z_t = (1.0 - t) * Z0 + t * x

    with autocast(enabled=amp):
        v_pred = model.rf(Z_t, t)  # Predict velocity v_theta(Z_t, t)
        loss = _mse(v_pred, delta)

    logs = {"fm_loss": float(loss.detach().cpu())}
    return loss, logs


# ----------------------------- RF sampler ----------------------------- #

@torch.no_grad()
def _rf_sample_euler(model: ENGRFAbs, y: torch.Tensor, steps: int = 50) -> torch.Tensor:
    """
    Deterministic Euler integration from t=0->1 using the learned unconditional RF.
    Start at x*(y); integrate dx/dt = v_theta(x_t, t) with dt=1/steps.

    This is purely for visualization during Stage-1 (no gauge).
    """
    device = next(model.parameters()).device
    dtype = y.dtype
    model.eval()
    model.pm.eval()

    x_star = model.pm(y.to(device)).clamp(0, 1).to(dtype)
    # Use the built-in sample_euler method which should be more robust
    x_rf = model.rf.sample_euler(x_star, steps=steps)
    return x_rf.clamp(0, 1)


# ----------------------------- Train API ----------------------------- #

def train_stage1(
    cfg: Dict,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    device: str = "cuda",
    pretrained: Optional[ENGRFAbs] = None,
    args: argparse.Namespace = None,
) -> ENGRFAbs:
    """
    Stage-1 trainer: learns the unconditional Rectified Flow v_theta via flow matching,
    using your ENGRFAbs container so it reuses the Stage-0 PM.

    Signature matches train.py (accepts `pretrained`).
    Returns: ENGRFAbs with trained RF (PM frozen; gauge unused here).
    """
    # ------------------------ Setup & Model ------------------------ #
    trn = cfg.get("train", {})
    exp = cfg.get("experiment", {})

    out_dir = trn["save_dir"]
    ckpt_dir      = os.path.join(out_dir, "ckpts_stage1")
    viz_dir       = os.path.join(out_dir, "viz_stage1")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    viz_val_dir   = os.path.join(viz_dir, "stage1/val")
    viz_train_dir = os.path.join(viz_dir, "stage1/train")
    os.makedirs(viz_val_dir, exist_ok=True)
    os.makedirs(viz_train_dir, exist_ok=True)

    amp           = bool(trn.get("amp", True))
    epochs        = int(trn.get("epochs_stage1", 50))
    lr            = float(trn.get("lr_stage1", 1e-4))
    wd            = float(trn.get("weight_decay", 1e-4))
    grad_clip     = float(trn.get("grad_clip", 1.0))
    log_interval  = int(trn.get("log_interval", 500))
    val_every     = int(trn.get("val_every", 1))
    vis_every     = int(trn.get("vis_every", val_every))
    vis_n         = int(trn.get("vis_n", 2))              # number of panels per val viz
    sample_steps  = int(trn.get("sample_steps_stage1", 50))# RF Euler steps for viz
    eps_t         = float(trn.get("t_eps", 0.0))           # avoid endpoints if you wish
    save_every    = float(trn.get("save_every", 0.5)) # save every 50% of epoch
    save_every    = int(save_every * len(train_loader))
    # Build or warm-start ENGRFAbs; RF will be trained, PM frozen
    model = pretrained if pretrained is not None else ENGRFAbs(cfg)
    model = model.to(device)

    # --------------------- Freeze/Unfreeze ------------------------- #
    for p in model.pm.parameters():      p.requires_grad = False
    for p in model.W.parameters():       p.requires_grad = False
    for p in model.hflow.parameters():   p.requires_grad = False
    for p in model.rf.parameters():      p.requires_grad = True

    n_trainable = _count_params(model.rf, trainable_only=True)
    logger.info(f"[Stage1] Trainable RF params: {n_trainable:,}")
    logger.info(f"[Stage1] Total model params: {_count_params(model):,}")
    total_rf_tensors = sum(1 for _, p in model.rf.named_parameters() if p.requires_grad)
    logger.info(f"[Stage1] Trainable RF tensors: {total_rf_tensors}")
    logger.info(f"[Stage1] PMRF sigma s: {model.pmrf_sigma_s}")
    logger.info(f"[Stage1] PMRF only: {model.pmrf_only}")


    opt = torch.optim.AdamW(model.rf.parameters(), lr=lr, weight_decay=wd)
    scaler = GradScaler(enabled=amp)

    # Resume from latest checkpoint
    if args.resume or args.resume_dir is not None:
        ckpt_resume_dir = os.path.join(args.resume_dir, "ckpts_stage1") if args.resume_dir is not None else ckpt_dir
        logger.info(f"[Stage1] Trying to resuming from directory: {ckpt_resume_dir}")
        latest_ckpt, model, opt, global_step, start_epoch = load_latest_checkpoint(ckpt_resume_dir, model, opt, device)
        for g in opt.param_groups:
            g["lr"] = lr
        logger.info(f"[Stage1] Found checkpoint: {latest_ckpt} global_step: {global_step} start_epoch: {start_epoch}")
        if global_step == 0 and start_epoch > 1:
            global_step = (start_epoch - 1) * len(train_loader)
        logger.info(f"[Stage1] Resuming from {latest_ckpt} at epoch {start_epoch} (global_step={global_step})")
    else:
        global_step = 0
        start_epoch = 1

    best_val = math.inf

    # ------------------------- Epoch Loop -------------------------- #
    for ep in range(start_epoch, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Stage1-RF] Epoch {ep}/{epochs}", leave=False)
        run_loss, run_cnt = 0.0, 0
        all_train_psnr = []
        all_train_ssim = []

        for it, batch in enumerate(pbar, 1):
            opt.zero_grad(set_to_none=True)

            # For parameter-change inspection: capture weights only on logging iterations
            pre_step_weights = None
            params_changed = None
            param_mean_abs_diff = None
            param_max_abs_diff = None
            capture_for_log = (it % log_interval) == 0
            if capture_for_log:
                with torch.no_grad():
                    pre_step_weights = {
                        name: p.detach().clone()
                        for name, p in model.rf.named_parameters()
                        if p.requires_grad
                    }

            loss, logs = _fm_step(model, batch, amp=amp, eps_t=eps_t)

            if amp:
                scaler.scale(loss).backward()
                # Unscale gradients before clipping so the threshold is meaningful
                scaler.unscale_(opt)
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.rf.parameters(), grad_clip)
                if grad_total_norm == float("inf"):
                    logger.warning(f"[Stage1] Grad total norm is infinite, skipping step {it} of epoch {ep}")
                    # save the batch that caused the infinite grad
                    torch.save(batch, os.path.join(out_dir, f"infinite_grad_batch_ep{ep:03d}_it{it:05d}.pt"))
                    continue
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.rf.parameters(), grad_clip)
                if grad_total_norm == float("inf"):
                    logger.warning(f"[Stage1] Grad total norm is infinite, skipping step {it} of epoch {ep}")
                    # save the batch that caused the infinite grad
                    torch.save(batch, os.path.join(out_dir, f"infinite_grad_batch_ep{ep:03d}_it{it:05d}.pt"))
                    continue
                opt.step()

            # If we captured pre-step weights for this iteration, compute which changed
            if pre_step_weights is not None:
                with torch.no_grad():
                    changed_count = 0
                    total_abs_diff = 0.0
                    total_elems = 0
                    max_abs_diff = 0.0
                    for name, p in model.rf.named_parameters():
                        if not p.requires_grad:
                            continue
                        prev = pre_step_weights.get(name, None)
                        if prev is None:
                            continue
                        diff = (p.detach() - prev).abs()
                        if diff.max().item() > 0.0:
                            changed_count += 1
                        total_abs_diff += diff.sum().item()
                        total_elems += diff.numel()
                        max_abs_diff = max(max_abs_diff, diff.max().item())
                    params_changed = changed_count
                    param_mean_abs_diff = (total_abs_diff / max(total_elems, 1)) if total_elems > 0 else float("nan")
                    param_max_abs_diff = max_abs_diff

            # Compute metrics for display (using RF sampler)
            with torch.no_grad():
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                x_rf = _rf_sample_euler(model, y, steps=sample_steps)
                x_clamp = x.clamp(0, 1)
                x_rf_clamp = x_rf.clamp(0, 1)
                psnr_b = _psnr(x_rf_clamp, x_clamp)
                ssim_b = _ssim(x_rf_clamp, x_clamp)
                all_train_psnr.append(psnr_b.cpu())
                all_train_ssim.append(ssim_b.cpu())

            bs = batch["x"].size(0)
            run_loss += float(loss.detach().cpu()) * bs
            run_cnt  += bs
            global_step += 1

            # Compute running averages
            avg_psnr = torch.cat(all_train_psnr).mean().item() if all_train_psnr else 0
            avg_ssim = torch.cat(all_train_ssim).mean().item() if all_train_ssim else 0
            gn = float(grad_total_norm.detach().float().cpu()) if torch.isfinite(grad_total_norm) else float("inf")

            if (it % log_interval) == 0:
                avg = run_loss / max(run_cnt, 1)
                # Parameter change stats (computed this iteration only)
                pc = params_changed if params_changed is not None else float("nan")
                pmean = param_mean_abs_diff if param_mean_abs_diff is not None else float("nan")
                pmax = param_max_abs_diff if param_max_abs_diff is not None else float("nan")
                logger.info(
                    f"[Stage1][Ep {ep}] it={it} fm_loss={avg:.6f} grad_norm={gn:.4f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.3f} "
                    f"params_changed={pc} mean_param_abs_diff={pmean:.6e} max_param_abs_diff={pmax:.6e}"
                )
                if WANDB:
                    wandb.log({
                        "train/fm_loss": avg,
                        "train/grad_norm": gn,
                        "train/params_changed": pc,
                        "train/param_mean_abs_diff": pmean,
                        "train/param_max_abs_diff": pmax,
                        "train/psnr": avg_psnr,
                        "train/ssim": avg_ssim,
                        "epoch": ep,
                        "iter": it,
                        "step": global_step,
                    })

            pbar.set_postfix(loss=f"{(run_loss/max(run_cnt,1)):.5f}", PSNR=f"{avg_psnr:.2f}", SSIM=f"{avg_ssim:.3f}", grad_norm=f"{gn:.4f}")

            if (it % save_every == 0):
                path = os.path.join(ckpt_dir, f"stage1_ep{ep:03d}_it{it:05d}.pt")
                save_ckpt(
                    path,
                    model.state_dict(),
                    opt.state_dict(),
                    global_step=global_step,
                    epoch=ep,
                    config=cfg,
                )
                logger.info(f"[Stage1] Saved periodic checkpoint to: {path}")

        # ------------------------- Validation ------------------------- #
        if val_loader is not None and (ep % val_every == 0):
            model.eval()
            tot, n = 0.0, 0
            all_psnr = []
            all_ssim = []
            all_nmse = []
            with torch.no_grad():
                vbar = tqdm(val_loader, desc=f"[Stage1-RF][Val] Epoch {ep}", leave=False)
                for vb in vbar:
                    vloss, _ = _fm_step(model, vb, amp=False, eps_t=eps_t)
                    bs = vb["x"].size(0)
                    tot += float(vloss.detach().cpu()) * bs
                    n   += bs
                    
                    # Calculate PSNR, SSIM, NMSE
                    x = vb["x"].to(device)
                    y = vb["y"].to(device)
                    x_rf = _rf_sample_euler(model, y, steps=sample_steps)
                    x_clamp = x.clamp(0, 1)
                    x_rf_clamp = x_rf.clamp(0, 1)
                    
                    psnr_b = _psnr(x_rf_clamp, x_clamp)
                    ssim_b = _ssim(x_rf_clamp, x_clamp)
                    nmse_b = _nmse(x_rf_clamp, x_clamp)
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
            logger.info(f"[Stage1-RF][Val] Epoch {ep} Summary:")
            logger.info(f"  Loss: {val_avg:.6f}")
            logger.info(f"  PSNR: {psnr_avg:.3f} dB")
            logger.info(f"  SSIM: {ssim_avg:.4f}")
            logger.info(f"  NMSE: {nmse_avg:.6f}")
            if WANDB:
                wandb.log({
                    "val/fm_loss": val_avg,
                    "val/PSNR": psnr_avg,
                    "val/SSIM": ssim_avg,
                    "val/NMSE": nmse_avg,
                    "epoch": ep,
                })

            # Save best (so your loader in train.py can warm-start Stage-2)
            if val_avg < best_val:
                best_val = val_avg
                path = os.path.join(ckpt_dir, f"best_stage1_ep{ep:03d}.pt")
                save_ckpt(
                    path,
                    model.state_dict(),
                    opt.state_dict(),
                    global_step=global_step,
                    epoch=ep,
                    config=cfg,
                    val_fm_loss=best_val
                )
                logger.info(f"[Stage1] Saved best checkpoint to: {path}")

            # ---------------------- Visualization ---------------------- #
            if (ep % vis_every == 0) and vis_n > 0:
                _viz_random_quads(
                    model=model,
                    dataset=val_loader.dataset,
                    batch_size=val_loader.batch_size,
                    device=device,
                    out_dir=viz_val_dir,
                    ep=ep,
                    vis_n=vis_n,
                    sample_steps=sample_steps,
                    split_label="val",
                )
                _viz_random_quads(
                    model=model,
                    dataset=train_loader.dataset,
                    batch_size=train_loader.batch_size,
                    device=device,
                    out_dir=viz_train_dir,
                    ep=ep,
                    vis_n=vis_n,
                    sample_steps=sample_steps,
                    split_label="train",
                )

        # ------------------------- Periodic Save ----------------------- #
        if ep == epochs:
            path = os.path.join(ckpt_dir, f"last_stage1_ep{ep:03d}.pt")
            save_ckpt(
                path,
                model.state_dict(),
                opt.state_dict(),
                global_step=global_step,
                epoch=ep,
                config=cfg,
            )
            logger.info(f"[Stage1] Saved last checkpoint to: {path}")

    return model


# --- add below save_quad_panel() ---
from torch.utils.data import DataLoader, SubsetRandomSampler
import random

@torch.no_grad()
def _viz_random_quads(
    model: ENGRFAbs,
    dataset,
    batch_size: int,
    device: str,
    out_dir: str,
    ep: int,
    vis_n: int,
    sample_steps: int,
    split_label: str,  # "train" or "val"
):
    """
    Randomly sample 'vis_n' items (batched) from 'dataset' and save
    [Input | Posterior Mean | Rectified Flow | GT] panels into out_dir.
    """
    n = len(dataset)
    if n == 0 or vis_n <= 0:
        return
    k = min(vis_n, n)
    idxs = random.sample(range(n), k)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=SubsetRandomSampler(idxs),
    )

    model.eval()
    model.pm.eval()

    saved = 0
    for vb in loader:
        x = vb["x"].to(device)
        y = vb["y"].to(device)

        # 1) Input (LF or raw input—here we use y)
        lf = y

        # 2) Posterior mean
        x_star = model.pm(y).clamp(0, 1)

        # 3) RF output (Euler steps, no gauge)
        x_rf = _rf_sample_euler(model, y, steps=sample_steps)

        # 4) GT
        gt = x

        B = x.size(0)
        for i in range(B):
            if saved >= vis_n:
                break
            out_path = os.path.join(out_dir, f"{split_label}_ep{ep:03d}_idx{saved:02d}.png")
            save_quad_panel(
                path=out_path,
                lf=lf[i].cpu(),
                pm=x_star[i].cpu(),
                rf=x_rf[i].cpu(),
                gt=gt[i].cpu(),
                labels=("Undersampled", "Posterior Mean", "Rectified Flow", "GT"),
                add_titles=True,
            )
            if WANDB:
                wandb.log({
                    f"viz/stage1_{split_label}": wandb.Image(out_path)
                }, commit=False)
            saved += 1
        if saved >= vis_n:
            break