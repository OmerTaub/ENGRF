from __future__ import annotations
import argparse, os, yaml, torch
from torch.utils.data import DataLoader

from training.stage0_pm import train_stage0_pm
from training.stage1 import train_stage1
from training.stage2 import train_stage2
from data.dataset import FastMRIMaskedAbsDataset
from data.LFHF_dataset import LFHFAbsPairDataset  
from models.engrf import ENGRFAbs


def load_pretrained(ckpt_path: str, fallback_cfg: dict, device: str):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("config", ckpt.get("cfg", fallback_cfg))
    model = ENGRFAbs(ckpt_cfg)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load_pretrained] Loaded with non-strict state_dict "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("Success loaded model")
    return model


def make_datasets(cfg):
    kind = cfg["data"].get("kind", "fastmri_mask")
    if kind == "fastmri_mask":
        # K-space padding/cropping (preferred) - use img_size or pad_to
        pad_to = cfg["data"].get("pad_to", None) or cfg["data"].get("img_size", None)
        if pad_to is not None:
            pad_to = tuple(pad_to)
        
        # Legacy image-space resize (discouraged)
        resize_to = cfg["data"].get("resize_to", None)
        if resize_to is not None:
            resize_to = tuple(resize_to)
        resize_mode = cfg["data"].get("resize_mode", "bilinear")
        
        train_ds = FastMRIMaskedAbsDataset(
            cfg["data"]["train_root"],
            center_fractions=tuple(cfg["data"].get("center_fractions_tr", (0.04, 0.04))),
            accelerations=tuple(cfg["data"].get("accelerations_tr", (4, 8))),
            seed=int(cfg["data"].get("seed_tr", 42)),
            pad_to=pad_to,
            deterministic=False,
            resize_to=resize_to,
            resize_mode=resize_mode,
        )
        val_ds = FastMRIMaskedAbsDataset(
            cfg["data"]["val_root"],
            center_fractions=tuple(cfg["data"].get("center_fractions_va", (0.04,0.04))),
            accelerations=tuple(cfg["data"].get("accelerations_va", (4, 4))),
            seed=int(cfg["data"].get("seed_va", 123)),
            pad_to=pad_to,
            deterministic=True,  # Stable masks for validation
            resize_to=resize_to,
            resize_mode=resize_mode,
        )
    elif kind == "lfhf_pair":
        print("LFHF_DATASET")
        train_ds = LFHFAbsPairDataset(
            lf_root=cfg["data"]["train_lf_root"],
            hf_root=cfg["data"]["train_hf_root"],
            recursive=True,
            resize_to=tuple(cfg["data"].get("resize_to", ())) or None,
            require_same_shape=bool(cfg["data"].get("require_same_shape", True)),
        )
        val_ds = LFHFAbsPairDataset(
            lf_root=cfg["data"]["val_lf_root"],
            hf_root=cfg["data"]["val_hf_root"],
            recursive=True,
            resize_to=tuple(cfg["data"].get("resize_to", ())) or None,
            require_same_shape=bool(cfg["data"].get("require_same_shape", True)),
        )
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")
    return train_ds, val_ds # overfitting check val_ds no include


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_swinir_hourglass.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stage", type=int, choices=[0, 1, 2], default=2)
    ap.add_argument("--ckpt", 
    #  default = None,
    default= f"/home/omertaub/projects/ENGRF/outputs/engrf_swinir_hourglass/ckpts_stage0/best_stage0_ep010.pt"
    # default=f"/home/omertaub/projects/ENGRF/outputs/engrf_swinir_hourglass/ckpts_stage1/best_stage1_ep034.pt"
    # default= f"/home/omertaub/projects/ENGRF/outputs/engrf_abs/ckpts_stage1/best_stage1_ep020.pt"
    )
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # dataset selection
    train_ds, val_ds = make_datasets(cfg)

    dl_tr = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        persistent_workers=bool(cfg["train"].get("persistent_workers", True)),
    )
    dl_va = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        persistent_workers=bool(cfg["train"].get("persistent_workers", True)),
    )

    # Optional warm start
    pretrained = load_pretrained(args.ckpt, cfg, args.device)

    out_dir = cfg.get("experiment", {}).get("out_dir", "runs")
    os.makedirs(out_dir, exist_ok=True)

    if args.stage == 0:
        model = train_stage0_pm(cfg, dl_tr, dl_va, device=args.device, pretrained=pretrained)
        torch.save({"state_dict": model.state_dict(), "config": cfg},
                   os.path.join(out_dir, "stage0_pm.pt"))

    elif args.stage == 1:
        if pretrained is None:
            print("[Stage-1] Warning: no checkpoint provided.")
        model = train_stage1(cfg, dl_tr, dl_va, device=args.device, pretrained=pretrained)
        torch.save({"state_dict": model.state_dict(), "config": cfg},
                   os.path.join(out_dir, "stage1.pt"))

    else:
        if pretrained is None:
            print("[Stage-2] Warning: no checkpoint provided.")
        model = train_stage2(cfg, dl_tr, dl_va, device=args.device, pretrained=pretrained)
        torch.save({"state_dict": model.state_dict(), "config": cfg},
                   os.path.join(out_dir, "stage2.pt"))


if __name__ == "__main__":
    main()
