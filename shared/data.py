import numpy as np
from sklearn.model_selection import train_test_split


def mad_normalize_hsi(patches: np.ndarray) -> np.ndarray:
    if patches.ndim == 4:
        x_data = patches[..., None]
    else:
        x_data = patches

    x_squeezed = x_data.squeeze(-1)
    band_median = np.median(x_squeezed, axis=(0, 2, 3))
    band_mad = np.median(np.abs(x_squeezed - band_median[None, :, None, None]), axis=(0, 2, 3)) + 1e-6
    x_norm = (x_squeezed - band_median[None, :, None, None]) / band_mad[None, :, None, None]
    x_norm = np.clip(x_norm, -5, 5) / 5.0
    return x_norm.astype(np.float32)


def split_train_val(idxs: np.ndarray, val_ratio: float = 0.15, seed: int = 42):
    train_idx, val_idx = train_test_split(idxs, test_size=val_ratio, random_state=seed, shuffle=True)
    return train_idx, val_idx
