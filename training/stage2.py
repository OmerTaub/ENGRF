# training/stage2.py
from __future__ import annotations
import os, math, logging
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from models.engrf import ENGRFAbs
try:
    import wandb
    WANDB = True
except Exception:
    WANDB = False

logger = logging.getLogger(__name__)

# --------------------------- small helpers --------------------------- #

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

# (tiny viz utils – identical to Stage0/1 for consistency)
import numpy as np
from PIL import Image

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

def _save_triptych(y: torch.Tensor, x_pred: torch.Tensor, x_gt: torch.Tensor, path: str):
    # Convert to numpy and normalize together to preserve relative relationships
    y_np = y.squeeze(0).cpu().numpy() if y.dim() == 3 else y.cpu().numpy()
    pred_np = x_pred.squeeze(0).cpu().numpy() if x_pred.dim() == 3 else x_pred.cpu().numpy()
    gt_np = x_gt.squeeze(0).cpu().numpy() if x_gt.dim() == 3 else x_gt.cpu().numpy()
    
    # Stack all images and normalize together
    all_images = np.stack([y_np, gt_np, pred_np], axis=0)
    all_min = all_images.min()
    all_max = all_images.max()
    
    if all_max > all_min:
        all_images = (all_images - all_min) / (all_max - all_min)
    else:
        all_images = np.zeros_like(all_images)
    
    # Convert to uint8
    yi = (all_images[0] * 255.0).round().astype(np.uint8)
    gi = (all_images[1] * 255.0).round().astype(np.uint8)
    pi = (all_images[2] * 255.0).round().astype(np.uint8)
    
    # Convert to RGB if needed
    if yi.ndim == 2: yi = np.stack([yi]*3, axis=-1)
    if gi.ndim == 2: gi = np.stack([gi]*3, axis=-1)
    if pi.ndim == 2: pi = np.stack([pi]*3, axis=-1)
    
    # Order: input (y), ground truth (x_gt), prediction (x_pred)
    trip = np.concatenate([yi, gi, pi], axis=1)
    Image.fromarray(trip).save(path)

# ----------------------------- core step ----------------------------- #

def _gfm_step(
    model: ENGRFAbs,
    batch: Dict[str, torch.Tensor],
    lambda_conjugacy: float,
    amp: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    One training/validation step for Stage-2 (gauge-FM with tied conjugacy).
    The tied form guarantees endpoint neutrality; 'lambda_conjugacy' is kept for API symmetry.
    """
    with autocast(enabled=amp):
        loss, logs = model.compute_stage2(batch, lambda_conjugacy=lambda_conjugacy)
    return loss, logs

# ----------------------------- Train API ----------------------------- #

def train_stage2(
    cfg: Dict,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: str = "cuda",
    pretrained: Optional[ENGRFAbs] = None
) -> ENGRFAbs:
    """
    Stage-2 trainer: learns the gauge flow h_t^Y (and optionally fine-tunes RF) via gauged FM.

    Theoretical requirements (ENGRF):
      - Uses the gauged FM target: ∂_t h_t^Y(Z_t) + (D h_t^Y)(Z_t) (X - X*)
      - Prediction is the tied conjugacy: ∂_t h_t^Y(tilde Z_t) + (D h_t^Y)(tilde Z_t) v(h^{-1}(tilde Z_t), t)
      - Enforce h_0^Y = h_1^Y = Id (checked once if model exposes 'check_h_identity')
    """
    trn = cfg.get("train", {})
    exp = cfg.get("experiment", {})

    out_dir  = trn.get("out_dir", exp.get("out_dir", "./outputs"))
    ckpt_dir = os.path.join(out_dir, "ckpts_stage2")
    viz_dir  = os.path.join(out_dir, "viz_stage2")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    viz_val_dir   = os.path.join(viz_dir, "val")
    viz_train_dir = os.path.join(viz_dir, "train")
    os.makedirs(viz_val_dir, exist_ok=True)
    os.makedirs(viz_train_dir, exist_ok=True)

    amp           = bool(trn.get("amp", True))
    epochs        = int(trn.get("epochs_stage2", 50))
    lr            = float(trn.get("lr_stage2", 1e-4))
    wd            = float(trn.get("weight_decay", 1e-4))
    grad_clip     = float(trn.get("grad_clip", 1.0))
    log_interval  = int(trn.get("log_interval", 50))
    val_every     = int(trn.get("val_every", 1))
    vis_every     = int(trn.get("vis_every", val_every))
    vis_n         = int(trn.get("vis_n", 0))
    sample_steps  = int(trn.get("sample_steps_stage2", 50))
    lam_resid     = float(trn.get("lambda_conjugacy", 1.0))
    ft_rf         = bool(trn.get("ft_rf_in_stage2", False))

    # Build/warm-start model
    model = pretrained if pretrained is not None else ENGRFAbs(cfg)
    model = model.to(device)

    # Freeze/Unfreeze according to Stage-2 spec
    for p in model.pm.parameters():    p.requires_grad = False
    for p in model.W.parameters():     p.requires_grad = True
    for p in model.hflow.parameters(): p.requires_grad = True
    for p in model.rf.parameters():    p.requires_grad = ft_rf

    n_gauge = _count_params(model.W, True) + _count_params(model.hflow, True)
    n_rf_ft = _count_params(model.rf, True)
    logger.info(f"[Stage2] Trainable params — gauge: {n_gauge:,} | rf(ft={ft_rf}): {n_rf_ft:,}")

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    scaler = GradScaler(enabled=amp)

    # ---- one-time endpoint neutrality sanity check ----
    if hasattr(model, "check_h_identity"):
        try:
            model.check_h_identity(device=device)  # assert ||h(y,0)-y|| and ||h(y,1)-y|| < eps
        except Exception as e:
            logger.warning(f"[Stage2] Warning: check_h_identity failed: {e}")

    best_val = math.inf

    # ------------------------- Epoch Loop -------------------------- #
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Stage2-Gauge] Epoch {ep}/{epochs}", leave=False)
        run_loss, run_cnt = 0.0, 0
        all_train_psnr = []
        all_train_ssim = []

        for it, batch in enumerate(pbar, 1):
            opt.zero_grad(set_to_none=True)

            loss, logs = _gfm_step(model, batch, lambda_conjugacy=lam_resid, amp=amp)

            if amp:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip)
                opt.step()

            # Compute metrics for display (using full gauge+RF sampler)
            with torch.no_grad():
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                x_out = model.sample(y, steps=sample_steps)
                x_clamp = x.clamp(0, 1)
                x_out_clamp = x_out.clamp(0, 1)
                psnr_b = _psnr(x_out_clamp, x_clamp)
                ssim_b = _ssim(x_out_clamp, x_clamp)
                all_train_psnr.append(psnr_b.cpu())
                all_train_ssim.append(ssim_b.cpu())

            bs = batch["x"].size(0)
            run_loss += float(loss.detach().cpu()) * bs
            run_cnt  += bs

            # Compute running averages
            avg_psnr = torch.cat(all_train_psnr).mean().item() if all_train_psnr else 0
            avg_ssim = torch.cat(all_train_ssim).mean().item() if all_train_ssim else 0

            if (it % log_interval) == 0:
                logger.info(f"[Stage2][Ep {ep}] it={it} gfm={logs.get('gfm_loss', float('nan')):.6f} "
                            f"resid={logs.get('conj_resid', float('nan')):.6f}")
                if WANDB:
                    wandb.log({
                        "train/gfm_loss": logs.get("gfm_loss"),
                        "train/conj_resid": logs.get("conj_resid"),
                        "epoch": ep,
                        "iter": it,
                    })
            pbar.set_postfix(loss=f"{(run_loss/max(run_cnt,1)):.4f}", PSNR=f"{avg_psnr:.2f}", SSIM=f"{avg_ssim:.3f}")

        # --------------------------- Validation --------------------------- #
        if val_loader is not None and (ep % val_every == 0):
            model.eval()
            tot, n = 0.0, 0
            all_psnr = []
            all_ssim = []
            all_nmse = []
            with torch.no_grad():
                vbar = tqdm(val_loader, desc=f"[Stage2-Gauge][Val] Epoch {ep}", leave=False)
                for vb in vbar:
                    vloss, vlogs = _gfm_step(model, vb, lambda_conjugacy=lam_resid, amp=False)
                    bs = vb["x"].size(0)
                    tot += float(vloss.detach().cpu()) * bs
                    n   += bs
                    
                    # Calculate PSNR, SSIM, NMSE
                    x = vb["x"].to(device)
                    y = vb["y"].to(device)
                    x_out = model.sample(y, steps=sample_steps)
                    x_clamp = x.clamp(0, 1)
                    x_out_clamp = x_out.clamp(0, 1)
                    
                    psnr_b = _psnr(x_out_clamp, x_clamp)
                    ssim_b = _ssim(x_out_clamp, x_clamp)
                    nmse_b = _nmse(x_out_clamp, x_clamp)
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
            logger.info(f"[Stage2-Gauge][Val] Epoch {ep} Summary:")
            logger.info(f"  Loss: {val_avg:.6f}")
            logger.info(f"  PSNR: {psnr_avg:.3f} dB")
            logger.info(f"  SSIM: {ssim_avg:.4f}")
            logger.info(f"  NMSE: {nmse_avg:.6f}")
            if WANDB:
                wandb.log({
                    "val/gfm_loss": val_avg,
                    "val/PSNR": psnr_avg,
                    "val/SSIM": ssim_avg,
                    "val/NMSE": nmse_avg,
                    "epoch": ep,
                })

            # Save best
            if val_avg < best_val:
                best_val = val_avg
                path = os.path.join(ckpt_dir, f"best_stage2_ep{ep:03d}.pt")
                torch.save({"state_dict": model.state_dict(), "config": cfg, "val_gfm_loss": best_val}, path)
                logger.info(f"[Stage2] Saved best checkpoint to: {path}")

            # ------------------------- Visualization ------------------------- #
            if (ep % vis_every == 0) and vis_n > 0:
                saved = 0
                with torch.no_grad():
                    for vb in val_loader:
                        x = vb["x"].to(device)
                        y = vb["y"].to(device)

                        # stage predictions
                        x_out = model.sample(y, steps=sample_steps)

                        for i in range(x.size(0)):
                            if saved >= vis_n: break
                            x_pm, x_rf = _infer_pm_rf(model, y, sample_steps)

                            _save_pentaptych(
                            y[i].cpu(), x_pm[i].cpu(), x_rf[i].cpu(), x_out[i].cpu(), x[i].cpu(),
                            os.path.join(viz_val_dir, f"ep{ep:03d}_idx{saved:02d}.png")  # <-- was viz_train_dir
                            )
                            if WANDB:
                                wandb.log({
                                    "viz/stage2_val": wandb.Image(os.path.join(viz_val_dir, f"ep{ep:03d}_idx{saved:02d}.png"))
                                }, commit=False)
                            saved += 1
                        if saved >= vis_n: break
            vis_n_train = int(trn.get("vis_n_train", 0))

            if (ep % vis_every == 0) and vis_n_train > 0:
                model.eval()
                saved = 0
                with torch.no_grad():
                    for tb in train_loader:
                        x = tb["x"].to(device)
                        y = tb["y"].to(device)

                        x_out = model.sample(y, steps=sample_steps)

                        for i in range(x.size(0)):
                            if saved >= vis_n_train: break
                            x_pm, x_rf = _infer_pm_rf(model, y, sample_steps)
                            _save_pentaptych(
                            y[i].cpu(), x_pm[i].cpu(), x_rf[i].cpu(), x_out[i].cpu(), x[i].cpu(),
                            os.path.join(viz_train_dir, f"ep{ep:03d}_idx{saved:02d}.png")
                        )
                            if WANDB:
                                wandb.log({
                                    "viz/stage2_train": wandb.Image(os.path.join(viz_train_dir, f"ep{ep:03d}_idx{saved:02d}.png"))
                                }, commit=False)
                            saved += 1
                        if saved >= vis_n_train: break
                model.train()

        # ---------------------------- Last save ---------------------------- #
        if ep == epochs:
            path = os.path.join(ckpt_dir, f"last_stage2_ep{ep:03d}.pt")
            torch.save({"state_dict": model.state_dict(), "config": cfg}, path)
            logger.info(f"[Stage2] Saved last checkpoint to: {path}")

    return model


def _save_pentaptych(y: torch.Tensor,
                     x_pm: torch.Tensor,
                     x_rf: torch.Tensor,
                     x_out: torch.Tensor,
                     x_gt: torch.Tensor,
                     path: str):
    # Convert to numpy and normalize together to preserve relative relationships
    y_np = y.squeeze(0).cpu().numpy() if y.dim() == 3 else y.cpu().numpy()
    pm_np = x_pm.squeeze(0).cpu().numpy() if x_pm.dim() == 3 else x_pm.cpu().numpy()
    rf_np = x_rf.squeeze(0).cpu().numpy() if x_rf.dim() == 3 else x_rf.cpu().numpy()
    out_np = x_out.squeeze(0).cpu().numpy() if x_out.dim() == 3 else x_out.cpu().numpy()
    gt_np = x_gt.squeeze(0).cpu().numpy() if x_gt.dim() == 3 else x_gt.cpu().numpy()
    
    # Stack all images and normalize together
    all_images = np.stack([y_np, pm_np, rf_np, out_np, gt_np], axis=0)
    all_min = all_images.min()
    all_max = all_images.max()
    
    if all_max > all_min:
        all_images = (all_images - all_min) / (all_max - all_min)
    else:
        all_images = np.zeros_like(all_images)
    
    # Convert to uint8
    yi = (all_images[0] * 255.0).round().astype(np.uint8)
    pmi = (all_images[1] * 255.0).round().astype(np.uint8)
    rfi = (all_images[2] * 255.0).round().astype(np.uint8)
    oi = (all_images[3] * 255.0).round().astype(np.uint8)
    gi = (all_images[4] * 255.0).round().astype(np.uint8)

    # promote to 3ch if any is 3ch
    if any(arr.ndim == 3 for arr in (yi, pmi, rfi, oi, gi)):
        if yi.ndim  == 2: yi  = np.stack([yi]*3,  axis=-1)
        if pmi.ndim == 2: pmi = np.stack([pmi]*3, axis=-1)
        if rfi.ndim == 2: rfi = np.stack([rfi]*3, axis=-1)
        if oi.ndim  == 2: oi  = np.stack([oi]*3,  axis=-1)
        if gi.ndim  == 2: gi  = np.stack([gi]*3,  axis=-1)

    panel = np.concatenate([yi, pmi, rfi, oi, gi], axis=1)
    Image.fromarray(panel).save(path)


@torch.no_grad()
def _rf_sample_euler(model: ENGRFAbs, y: torch.Tensor, steps: int = 50) -> torch.Tensor:
    """Deterministic Euler integration of the unconditional RF from x* to t=1."""
    device, dtype = y.device, y.dtype
    model.eval(); model.pm.eval()
    x_star = model.pm(y).clamp(0, 1).to(dtype)
    x_t = x_star.clone()
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x_t.size(0), 1, 1, 1), (i + 0.5) * dt, device=device, dtype=dtype)
        v = model.rf(x_t, t).to(dtype)  # dx/dt = v_theta(x_t, t)
        x_t = x_t + v * dt
    return x_t.clamp(0, 1)

@torch.no_grad()
def _infer_pm_rf(model, y: torch.Tensor, steps: int):
    """Returns (x_pm, x_rf) on y.device. Falls back to Euler over model.rf."""
    device = y.device
    model.eval(); model.pm.eval()

    # Posterior mean
    if hasattr(model, "posterior_mean") and callable(getattr(model, "posterior_mean")):
        x_pm = model.posterior_mean(y)
    elif hasattr(model, "pm"):
        pm_mod = getattr(model, "pm")
        x_pm = pm_mod(y) if callable(pm_mod) else pm_mod.forward(y)
    else:
        raise AttributeError("Model lacks 'posterior_mean' or 'pm' for Stage-2 viz.")
    x_pm = x_pm.clamp(0, 1).to(device)

    # Rectified flow (try model.rectified_flow first, else Euler over model.rf)
    x_rf = None
    if hasattr(model, "rectified_flow") and callable(getattr(model, "rectified_flow")):
        try:
            x_rf = model.rectified_flow(x_pm, steps=steps)
        except TypeError:
            x_rf = model.rectified_flow(x_pm)
    if x_rf is None:
        x_rf = _rf_sample_euler(model, y, steps=steps)
    return x_pm, x_rf