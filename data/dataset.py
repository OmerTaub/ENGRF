import os, glob, math, numpy as np, torch
from torch.utils.data import Dataset

# Optional deps
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    _HAVE_PYDICOM = True
except Exception:
    _HAVE_PYDICOM = False

try:
    import nibabel as nib
    _HAVE_NIB = True
except Exception:
    _HAVE_NIB = False

from PIL import Image

def _read_npy(path):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, dict) or hasattr(arr, "item"):
        d = arr.item()
        x = d["x"] if "x" in d else d.get("image", None)
        if x is None:
            raise ValueError(f"{path}: dict .npy missing key 'x' or 'image'")
        return np.asarray(x)
    return np.asarray(arr)

def _read_img_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        im = Image.open(path).convert("F")        # float grayscale
        return np.array(im)
    if ext in {".dcm", ".dicom"}:
        if not _HAVE_PYDICOM:
            raise ImportError("Install pydicom to read DICOM.")
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        # Apply VOI LUT if present
        try:
            arr = apply_voi_lut(arr, ds).astype(np.float32)
        except Exception:
            pass
        # Rescale if slope/intercept exist
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + inter
        return arr
    if ext in {".nii", ".gz"} or path.endswith(".nii.gz"):
        if not _HAVE_NIB:
            raise ImportError("Install nibabel to read NIfTI.")
        data = np.asanyarray(nib.load(path).dataobj).astype(np.float32)
        if data.ndim == 3:
            z = data.shape[2] // 2  # middle slice
            data = data[..., z]
        if data.ndim != 2:
            raise ValueError(f"{path}: expected 2D slice, got shape {data.shape}")
        return data
    if ext == ".npy":
        return _read_npy(path)
    raise ValueError(f"Unsupported file type: {path}")

def _norm01(a):
    a = np.asarray(a, dtype=np.float32)
    amin, amax = np.min(a), np.max(a)
    if not np.isfinite(amin) or not np.isfinite(amax):
        a = np.nan_to_num(a, copy=False)
        amin, amax = np.min(a), np.max(a)
    if amax > amin:
        a = (a - amin) / (amax - amin)
    else:
        a = np.zeros_like(a, dtype=np.float32)
    return a

def _fft2c(xc):  # centered 2D FFT, xc: complex tensor HxW
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(xc, dim=(-2, -1)),
                                             norm="ortho"), dim=(-2, -1))

def _ifft2c(Xc): # centered 2D IFFT
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(Xc, dim=(-2, -1)),
                                              norm="ortho"), dim=(-2, -1))

def _center_pad_crop_complex(K: torch.Tensor, target_hw: tuple[int,int]) -> torch.Tensor:
    """
    Center pad/crop a centered k-space tensor K (H,W) complex to target_hw.
    Assumes K is fftshifted (centered). Returns a new complex tensor (Ht,Wt).
    """
    H, W = K.shape[-2], K.shape[-1]
    Ht, Wt = int(target_hw[0]), int(target_hw[1])

    # If already target size, return as-is
    if H == Ht and W == Wt:
        return K

    out = torch.zeros((*K.shape[:-2], Ht, Wt), dtype=K.dtype, device=K.device)

    # compute source crop region
    hs = min(H, Ht); ws = min(W, Wt)
    h_src0 = (H - hs) // 2; w_src0 = (W - ws) // 2
    h_tgt0 = (Ht - hs) // 2; w_tgt0 = (Wt - ws) // 2

    out[..., h_tgt0:h_tgt0+hs, w_tgt0:w_tgt0+ws] = K[..., h_src0:h_src0+hs, w_src0:w_src0+ws]
    return out


def _center_pad_crop_mask_1d(mask_1d: np.ndarray, target_w: int) -> np.ndarray:
    """
    Center pad/crop a boolean 1-D mask (W,) to length target_w.
    """
    W = mask_1d.shape[0]
    if W == target_w:
        return mask_1d
    out = np.zeros((target_w,), dtype=bool)
    ws = min(W, target_w)
    w_src0 = (W - ws) // 2
    w_tgt0 = (target_w - ws) // 2
    out[w_tgt0:w_tgt0+ws] = mask_1d[w_src0:w_src0+ws]
    return out


def _fastmri_random_mask_1d(num_cols:int, center_fraction:float, acceleration:int, rng:np.random.Generator):
    """
    FastMRI-style random line mask along columns (phase-encode).
    Keeps a centered low-frequency band; picks remaining columns at prob that yields the target accel on expectation.
    See formula used in open-source fastMRI-style implementations. :contentReference[oaicite:2]{index=2}
    """
    if center_fraction >= 1.0 and float(center_fraction).is_integer():
        num_low = int(center_fraction)
    else:
        num_low = int(round(num_cols * float(center_fraction)))
    num_low = max(1, min(num_low, num_cols))
    mask = np.zeros(num_cols, dtype=bool)

    # center band
    pad = (num_cols - num_low) // 2
    mask[pad:pad + num_low] = True

    # probability for the rest
    target = num_cols / float(acceleration)
    prob = max(0.0, min(1.0, (target - num_low) / max(1, (num_cols - num_low))))
    if prob > 0:
        outside = np.r_[np.arange(0, pad), np.arange(pad + num_low, num_cols)]
        draw = rng.uniform(size=outside.shape[0]) < prob
        mask[outside] = draw
    return mask  # (W,)

class FastMRIMaskedAbsDataset(Dataset):
    """
    Loads 2D magnitude 'x' from common formats, converts to complex by treating as real part,
    standardizes k-space to a fixed grid via center pad/crop, applies a fastMRI-style 1-D PE mask,
    and returns GT and zero-filled recon on the SAME grid.

    Returns:
      {'x': (1,Ht,Wt), 'y': (1,Ht,Wt), 'mask': (Wt,), 'path': str}
    """
    def __init__(self, root:str,
                 center_fractions=(0.08,), accelerations=(4,),
                 seed:int|None=None, recursive=True,
                 pad_to:tuple[int,int]|None=(320,320),
                 deterministic:bool=False,
                 # legacy args kept but discouraged for single-coil; only used if not None
                 resize_to=None, resize_mode="bilinear"):
        self.paths = []
        patterns = ["*.npy","*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.dcm","*.dicom","*.nii","*.nii.gz"]
        for p in patterns:
            globber = glob.glob(os.path.join(root, "**", p) if recursive else os.path.join(root, p),
                                recursive=recursive)
            self.paths.extend(globber)
        self.paths = sorted(self.paths)
        if not self.paths:
            raise FileNotFoundError(f"No image files found in {root}")

        self.center_fractions = tuple(center_fractions)
        self.accelerations   = tuple(accelerations)
        self.global_seed     = seed if seed is not None else 0
        self.rng             = np.random.default_rng(self.global_seed)
        self.pad_to          = pad_to
        self.deterministic   = deterministic

        # Kept for backward-compatibility; prefer pad_to over resize_to.
        self.resize_to = resize_to
        self.resize_mode = resize_mode
        print("Number of slices: "+ str(len(self.paths)))

    def __len__(self):
        return int(int(len(self.paths)) / 1)

    def _rng_for_idx(self, idx:int) -> np.random.Generator:
        if not self.deterministic:
            return self.rng
        # Stable per-item RNG (same mask every epoch for val/test)
        return np.random.default_rng(self.global_seed ^ (idx * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFF))

    def __getitem__(self, idx):
        path = self.paths[idx]

        # 1) load & normalize magnitude image to [0,1]
        x_np = _read_img_any(path)
        x_np = _norm01(x_np)  # (H,W), float32

        if x_np.ndim != 2:
            raise ValueError(f"{path}: expected 2D array, got {x_np.shape}")
        H, W = x_np.shape

        # 2) Build native k-space, then standardize to target grid via center pad/crop
        x_t = torch.from_numpy(x_np).float()                      # (H,W)
        zc  = torch.complex(x_t, torch.zeros_like(x_t))           # real->complex
        K   = _fft2c(zc)                                          # (H,W) complex

        if self.pad_to is not None:
            Ht, Wt = int(self.pad_to[0]), int(self.pad_to[1])
            K_std  = _center_pad_crop_complex(K, (Ht, Wt))        # (Ht,Wt) complex
        else:
            Ht, Wt = H, W
            K_std  = K

        # 3) Choose (cf, R) + RNG
        rng = self._rng_for_idx(idx)
        cf = self.center_fractions[rng.integers(len(self.center_fractions))]
        R  = int(self.accelerations[rng.integers(len(self.accelerations))])

        # 4) Make 1-D PE mask on the standardized width (Wt)
        col_mask = _fastmri_random_mask_1d(Wt, cf, R, rng)        # (Wt,)
        mask2d = torch.from_numpy(np.broadcast_to(col_mask[None, :], (Ht, Wt)).copy()).to(torch.bool)

        # 5) Zero-filled recon and GT on the SAME grid
        x_full = _ifft2c(K_std).abs()                              # (Ht,Wt)
        K_masked = torch.where(mask2d, K_std, torch.zeros_like(K_std))
        y_zf = _ifft2c(K_masked).abs()                             # (Ht,Wt)

        # 6) Normalize both to [0,1] using the GT scale (keeps comparisons fair)
        scale = max(x_full.max().item(), 1e-6)
        x = (x_full / scale).clamp(0, 1).unsqueeze(0)              # (1,Ht,Wt)
        y = (y_zf   / scale).clamp(0, 1).unsqueeze(0)              # (1,Ht,Wt)

        # 7) (Discouraged) image-domain resize only if explicitly requested
        if self.resize_to is not None:
            Ht2, Wt2 = int(self.resize_to[0]), int(self.resize_to[1])
            x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(Ht2, Wt2), mode=self.resize_mode, align_corners=False).squeeze(0)
            y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(Ht2, Wt2), mode=self.resize_mode, align_corners=False).squeeze(0)
            col_mask = _center_pad_crop_mask_1d(col_mask, Wt2)

        return {"x": x, "y": y, "mask": torch.from_numpy(col_mask), "path": path}



# viz_fastmri_sample.py
import argparse, numpy as np, torch, matplotlib.pyplot as plt


def to_kspace_mag(img_1hw: torch.Tensor) -> np.ndarray:
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

def imshow_gray(ax, img, title):
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="Dataset root directory", default=f"/home/omertaub/data/ULF_EnC_slices/training/HF")
    ap.add_argument("--idx", type=int, default=0, help="Sample index to visualize")
    ap.add_argument("--seed", type=int, default=0, help="Dataset RNG seed (reproducible masks)")
    ap.add_argument("--center_fractions", type=float, nargs="+", default=[0.08], help="e.g. 0.04 0.08")
    ap.add_argument("--accelerations", type=int, nargs="+", default=[8], help="e.g. 4 8")
    args = ap.parse_args()

    ds = FastMRIMaskedAbsDataset(
        root=args.root,
        center_fractions=tuple(args.center_fractions),
        accelerations=tuple(args.accelerations),
        seed=args.seed,
        recursive=True,
    )

    if not (0 <= args.idx < len(ds)):
        raise IndexError(f"idx {args.idx} out of range [0,{len(ds)-1}]")

    sample = ds[args.idx]
    x = sample["x"]           # (1,H,W)
    y = sample["y"]           # (1,H,W)
    mask = sample["mask"]     # (W,)
    path = sample["path"]

    # Prepare images for display
    x_np = x.squeeze(0).cpu().numpy()
    y_np = y.squeeze(0).cpu().numpy()
    Kx = to_kspace_mag(x)
    Ky = to_kspace_mag(y)

    # Plot
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1,1,0.25])

    ax1 = fig.add_subplot(gs[0,0]); imshow_gray(ax1, x_np, "x (target)")
    ax2 = fig.add_subplot(gs[0,1]); imshow_gray(ax2, Kx,  "k-space |F{x}| (log)")

    ax3 = fig.add_subplot(gs[1,0]); imshow_gray(ax3, y_np, "y (ZF recon)")
    ax4 = fig.add_subplot(gs[1,1]); imshow_gray(ax4, Ky,   "k-space |F{y}| (log)")

    # Optional: show the 1-D phase-encode mask used (as an image strip)
    ax5 = fig.add_subplot(gs[1,2])
    ax5.imshow(mask[None, :].cpu().numpy(), cmap="gray", aspect="auto", interpolation="nearest")
    ax5.set_title("PE mask (columns)")
    ax5.set_yticks([]); ax5.set_xticks([])

    ax6 = fig.add_subplot(gs[0,2], frameon=False)  
    ax6.axis('off')
    fig.suptitle(f"Sample from: {path}", fontsize=10)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
