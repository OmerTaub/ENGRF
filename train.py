from __future__ import annotations
import argparse, os, yaml, torch, logging
import wandb
from torch.utils.data import DataLoader

from training.stage0_pm import train_stage0_pm
from training.stage1 import train_stage1
from training.stage2 import train_stage2
from data.dataset import FastMRIMaskedAbsDataset
from data.LFHF_dataset import LFHFAbsPairDataset  
from models.engrf import ENGRFAbs
from util.checkpoint import get_outdir
logger = logging.getLogger(__name__)


def setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def load_pretrained(ckpt_path: str, fallback_cfg: dict, device: str):
    if not ckpt_path or not os.path.exists(ckpt_path):
        logger.warning(f"[load_pretrained] Checkpoint not found: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("config", ckpt.get("cfg", fallback_cfg))
    model = ENGRFAbs(ckpt_cfg)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.info(f"[load_pretrained] Loaded with non-strict state_dict (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        logger.info(f"[load_pretrained] Success loaded model from {ckpt_path}")
    return model


def make_datasets(cfg):
    kind = cfg["data"].get("kind", "fastmri_mask")
    if kind == "fastmri_mask":
        logger.info("FASTMRI_MASK_DATASET")
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
            center_fractions = tuple(cfg["data"]["center_fractions_tr"]),
            accelerations = tuple(cfg["data"]["accelerations_tr"]),

            seed=int(cfg["data"].get("seed_tr", 42)),
            pad_to=pad_to,
            deterministic=False,
            resize_to=resize_to,
            resize_mode=resize_mode,
        )
        val_ds = FastMRIMaskedAbsDataset(
            cfg["data"]["val_root"],
            center_fractions = tuple(cfg["data"]["center_fractions_va"]),
            accelerations = tuple(cfg["data"]["accelerations_va"]),
            seed=int(cfg["data"].get("seed_va", 123)),
            pad_to=pad_to,
            deterministic=True,  # Stable masks for validation
            resize_to=resize_to,
            resize_mode=resize_mode,
        )
    elif kind == "lfhf_pair":
        logger.info("LFHF_DATASET")
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

def to_wandb_id(out_dir: str) -> str:
    # take out_dir remove "/"
    return out_dir.replace("/", "")



def main():
    setup_logging(verbosity=1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_swinir_hourglass.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stage", type=int, choices=[0, 1, 2], default=2)
    ap.add_argument("--ckpt", 
     default = None,
    # default= f"/home/omertaub/projects/ENGRF/outputs/engrf_swinir_hourglass/ckpts_stage0/best_stage0_ep010.pt"
    # default=f"/home/omertaub/projects/ENGRF/outputs/engrf_swinir_hourglass/ckpts_stage1/best_stage1_ep034.pt"
    # default= f"/home/omertaub/projects/ENGRF/outputs/engrf_abs/ckpts_stage1/best_stage1_ep020.pt"
    )
    ap.add_argument("--resume", action="store_true", help="Automatically load latest checkpoint in latest run directory. This uses the latest run with highest X for run_X as the output directory.")
    ap.add_argument("--resume_dir", type=str, default=None, help="Specify which run directory to load latest checkpoint from. This uses the specified run directory as the output directory.")
    ap.add_argument("--wandb_id", type=str, default=None, help="Specify which wandb id to use.")
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

    if args.resume_dir is not None:
        out_dir = args.resume_dir
    else:
        out_dir = get_outdir(cfg, resume=args.resume)
    os.makedirs(out_dir, exist_ok=True)

    cfg["train"]["save_dir"] = out_dir
    save_dir = cfg["train"]["save_dir"]

    # Add file handler to write logs to out_dir/log.txt
    log_path = os.path.join(save_dir, "log.txt")
    root_logger = logging.getLogger()
    # Avoid duplicating file handlers if main() is called multiple times
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == os.path.abspath(log_path) for h in root_logger.handlers):
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(root_logger.level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S"))
        root_logger.addHandler(file_handler)

    # Initialize Weights & Biases
    wandb_id = to_wandb_id(out_dir) if args.wandb_id is None else args.wandb_id
    run = wandb.init(
        project="engrf",
        name=out_dir,
        id=wandb_id,
        config=cfg,
        dir=save_dir,
        resume="must" if args.resume_dir is not None else "allow",
    )
    wandb.define_metric("epoch")
    wandb.define_metric("iter")
    logger.info(f"[wandb] Initialized with project={run.project}, name={out_dir}, dir={save_dir}")

    # Optional warm start
    pretrained = load_pretrained(args.ckpt, cfg, args.device)

    if args.stage == 0:
        model = train_stage0_pm(cfg, dl_tr, dl_va, device=args.device, pretrained=pretrained, args=args)
        save_ckpt(os.path.join(out_dir, "stage0_pm.pt"), model.state_dict())

    elif args.stage == 1:
        if pretrained is None:
            logger.warning("[Stage-1] Warning: no checkpoint provided.")
        model = train_stage1(cfg, dl_tr, dl_va, device=args.device, pretrained=pretrained, args=args)
        save_ckpt(os.path.join(out_dir, "stage1.pt"), model.state_dict())

    else:
        if pretrained is None:
            logger.warning("[Stage-2] Warning: no checkpoint provided.")
        model = train_stage2(cfg, dl_tr, dl_va, device=args.device, pretrained=pretrained, args=args)
        save_ckpt(os.path.join(out_dir, "stage2.pt"), model.state_dict())


if __name__ == "__main__":
    main()
