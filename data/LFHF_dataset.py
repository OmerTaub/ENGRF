import os, glob, math, numpy as np, torch
from torch.utils.data import Dataset

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

class LFHFAbsPairDataset(Dataset):
    """
    Loads paired magnitude images where:
      - y is read from a low-frequency (LF) root
      - x is read from a high-frequency (HF) root
    No synthetic degradation is applied.

    Pairing rule: match by filename stem (without extension), optionally across
    nested folders (recursive=True). Only stems present in BOTH roots are kept.

    Returns:
      dict {
        'x': (1,H,W) float32 in [0,1],   # target / HF
        'y': (1,H,W) float32 in [0,1],   # input  / LF
        'path_x': str,
        'path_y': str,
        'stem': str
      }

    Args:
      lf_root (str): directory containing LF images (y).
      hf_root (str): directory containing HF images (x).
      recursive (bool): search subfolders too.
      resize_to (tuple[int,int] | None): if provided (H,W), both x and y are
        resized to this size with bilinear interpolation.
      require_same_shape (bool): if True and shapes differ (and no resize_to),
        raises ValueError. If False, the larger one is center-cropped to the
        smaller one's size (no resize).
      transform (callable | None): optional fn(dict) -> dict applied to the
        output sample (after resizing/cropping).
    """
    def __init__(self,
                 lf_root: str,
                 hf_root: str,
                 recursive: bool = True,
                 resize_to = None,
                 require_same_shape: bool = True,
                 transform = None):
        super().__init__()
        self.lf_root = lf_root
        self.hf_root = hf_root
        self.recursive = recursive
        self.resize_to = resize_to
        self.require_same_shape = require_same_shape
        self.transform = transform

        patterns = ["*.npy","*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.dcm","*.dicom","*.nii","*.nii.gz"]

        def _scan(root):
            paths = []
            for p in patterns:
                globber = glob.glob(os.path.join(root, "**", p) if recursive else os.path.join(root, p),
                                    recursive=recursive)
                paths.extend(globber)
            return sorted(paths)

        lf_paths = _scan(lf_root)
        hf_paths = _scan(hf_root)
        if not lf_paths:
            raise FileNotFoundError(f"No images found under LF root: {lf_root}")
        if not hf_paths:
            raise FileNotFoundError(f"No images found under HF root: {hf_root}")

        def _stem(path: str) -> str:
            # handle .nii.gz as one extension
            base = os.path.basename(path)
            if base.endswith(".nii.gz"):
                return base[:-7]
            return os.path.splitext(base)[0]

        lf_map = {}
        for p in lf_paths:
            lf_map.setdefault(_stem(p), []).append(p)
        hf_map = {}
        for p in hf_paths:
            hf_map.setdefault(_stem(p), []).append(p)

        # Keep only stems present in both; if multiple matches per stem, pick first (deterministic)
        common = sorted(set(lf_map.keys()) & set(hf_map.keys()))
        if not common:
            raise FileNotFoundError("No matching stems found between LF and HF roots.")
        self.pairs = [(hf_map[s][0], lf_map[s][0], s) for s in common]  # (x_path, y_path, stem)

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _to_chw01(a: np.ndarray) -> torch.Tensor:
        a = _norm01(a)
        if a.ndim != 2:
            raise ValueError(f"Expected 2D grayscale array, got shape {a.shape}")
        t = torch.from_numpy(a).float().unsqueeze(0)  # (1,H,W)
        return t

    @staticmethod
    def _center_crop_to(t: torch.Tensor, H: int, W: int) -> torch.Tensor:
        _, h, w = t.shape
        top = max(0, (h - H) // 2)
        left = max(0, (w - W) // 2)
        return t[:, top:top + H, left:left + W]

    def __getitem__(self, idx: int):
        x_path, y_path, stem = self.pairs[idx]

        x_np = _read_img_any(x_path)
        y_np = _read_img_any(y_path)

        x = self._to_chw01(x_np)  # (1,Hx,Wx)
        y = self._to_chw01(y_np)  # (1,Hy,Wy)

        Hx, Wx = x.shape[-2], x.shape[-1]
        Hy, Wy = y.shape[-2], y.shape[-1]

        if self.resize_to is not None:
            Ht, Wt = int(self.resize_to[0]), int(self.resize_to[1])
            x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(Ht, Wt), mode="bilinear", align_corners=False).squeeze(0)
            y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(Ht, Wt), mode="bilinear", align_corners=False).squeeze(0)
        else:
            if (Hx, Wx) != (Hy, Wy):
                if self.require_same_shape:
                    raise ValueError(
                        f"Shape mismatch for stem '{stem}': x={x_path} -> {(Hx,Wx)}, y={y_path} -> {(Hy,Wy)}. "
                        f"Set resize_to=(H,W) or require_same_shape=False to handle."
                    )
                # center-crop larger to smaller
                Ht, Wt = min(Hx, Hy), min(Wx, Wy)
                x = self._center_crop_to(x, Ht, Wt)
                y = self._center_crop_to(y, Ht, Wt)

        sample = {
            "x": x,            # HF / target
            "y": y,            # LF / input
            "path_x": x_path,
            "path_y": y_path,
            "stem": stem,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
