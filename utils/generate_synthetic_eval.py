"""Build a labeled hyperspectral evaluation set that does not reuse training patches.

The FedAvg clients load every patch in the seven ``data/*.npy`` files and split
each file 85/15 into train and validation. Those 1,151 patches are therefore
not a held-out test set. This script fits a spectral PCA on a pixel sample,
then builds new 32x32 patches by blending several donors inside 8x8 blocks and
adding score-space noise. A candidate is kept only when its spatial pattern is
below a correlation threshold against every training patch and the eight
dihedral copies of that patch (rotations and mirrors).

Anomalies are injected only into accepted synthetic backgrounds. The result is
evaluation data for the global ConvAE3D model: raw float32 arrays in the same
``(N, 131, 32, 32)`` layout as the client files. Run ``mad_normalize_hsi`` and
add a channel axis before the forward pass, matching ``client/train.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def mad_normalize_hsi(patches: np.ndarray) -> np.ndarray:
    """Same per-band median/MAD scaling as ``shared.data.mad_normalize_hsi``."""
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

CLIENT_FILES = [
    "AEZ_6.npy",
    "AEZ_8.npy",
    "AEZ_9.npy",
    "AEZ_10.npy",
    "Split_1_ImageOnly.npy",
    "Split_2_ImageOnly.npy",
    "Split_3_ImageOnly.npy",
]
BANDS = np.array([30, 60, 90, 120], dtype=np.int64)
N_COMPONENTS = 16
N_DONORS = 6
BLOCK = 8
NOISE_SCALE = 0.20
MAX_SPATIAL_CORR = 0.75
N_NORMAL = 1152
N_PER_SCENARIO = 384
ANOMALY_FRACTION = 0.03
SCENARIOS = ("spectral_spike", "spectral_substitution", "band_absorption")
PIXELS_PER_FILE = 6000
BATCH = 32
MAX_ATTEMPT_FACTOR = 25


def load_training(data_dir: Path) -> tuple[np.ndarray, list[dict]]:
    parts = []
    inventory = []
    for name in CLIENT_FILES:
        path = data_dir / name
        array = np.load(path)
        if array.ndim != 4 or array.shape[1:] != (131, 32, 32):
            raise ValueError(f"{name} has shape {array.shape}, expected (N, 131, 32, 32)")
        parts.append(array.astype(np.float32, copy=False))
        inventory.append({"file": name, "patches": int(array.shape[0]), "dtype": str(array.dtype)})
    train = np.concatenate(parts, axis=0)
    return train, inventory


def fit_pca(train: np.ndarray, ranges: list[tuple[int, int]], rng: np.random.Generator):
    samples = []
    offset = 0
    for count in (end - start for start, end in ranges):
        take = min(PIXELS_PER_FILE, count * 32 * 32)
        patch_ids = rng.integers(offset, offset + count, size=take)
        ys = rng.integers(0, 32, size=take)
        xs = rng.integers(0, 32, size=take)
        samples.append(train[patch_ids, :, ys, xs])
        offset += count
    pixels = np.concatenate(samples, axis=0).astype(np.float64)
    mean = pixels.mean(axis=0)
    centered = pixels - mean
    covariance = (centered.T @ centered) / (pixels.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:N_COMPONENTS]
    components = eigenvectors[:, order].T
    score_std = np.sqrt(np.maximum(eigenvalues[order], 1e-8))
    return mean.astype(np.float64), components.astype(np.float64), score_std.astype(np.float32)


def pack_maps(patches: np.ndarray) -> np.ndarray:
    images = patches[:, BANDS].astype(np.float32, copy=False)
    flat = np.ascontiguousarray(images.reshape(images.shape[0], len(BANDS), -1))
    flat -= flat.mean(axis=2, keepdims=True)
    flat /= np.linalg.norm(flat, axis=2, keepdims=True) + 1e-8
    return flat


def dihedral_bank(identity: np.ndarray) -> np.ndarray:
    images = identity.reshape(identity.shape[0], len(BANDS), 32, 32)
    variants = []
    for turns in range(4):
        rotated = np.rot90(images, turns, axes=(2, 3))
        variants.append(np.ascontiguousarray(rotated.reshape(identity.shape[0], len(BANDS), -1)))
        mirrored = np.flip(rotated, axis=3)
        variants.append(np.ascontiguousarray(mirrored.reshape(identity.shape[0], len(BANDS), -1)))
    return np.concatenate(variants, axis=0)


def max_spatial_corr(patches: np.ndarray, bank: np.ndarray) -> np.ndarray:
    maps = pack_maps(patches)
    band_max = np.empty((patches.shape[0], len(BANDS)), dtype=np.float32)
    for band in range(len(BANDS)):
        corr = maps[:, band] @ bank[:, band].T
        band_max[:, band] = corr.max(axis=1)
    return band_max.max(axis=1)


def synthesize_batch(
    train: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    score_std: np.ndarray,
    rng: np.random.Generator,
    batch: int,
) -> np.ndarray:
    donors = np.empty((batch, N_DONORS), dtype=np.int32)
    for row in range(batch):
        donors[row] = rng.choice(train.shape[0], size=N_DONORS, replace=False)

    raw = train[donors]
    flat = raw.transpose(0, 1, 3, 4, 2).reshape(batch * N_DONORS, 32 * 32, 131)
    scores = (flat - mean) @ components.T
    scores = scores.reshape(batch, N_DONORS, 32, 32, N_COMPONENTS).astype(np.float32)
    mixed = np.empty((batch, 32, 32, N_COMPONENTS), dtype=np.float32)
    noise_std = (NOISE_SCALE * score_std).astype(np.float32)

    for by in range(0, 32, BLOCK):
        for bx in range(0, 32, BLOCK):
            pick = np.empty((batch, 2), dtype=np.int32)
            for row in range(batch):
                pick[row] = rng.choice(N_DONORS, size=2, replace=False)
            weight = rng.uniform(0.40, 0.60, size=(batch, 1, 1, 1)).astype(np.float32)
            rows = np.arange(batch)
            left = scores[rows, pick[:, 0], by : by + BLOCK, bx : bx + BLOCK]
            right = scores[rows, pick[:, 1], by : by + BLOCK, bx : bx + BLOCK]
            block = np.array(weight * left + (1.0 - weight) * right, copy=True)
            flip_y = rng.random(batch) < 0.5
            flip_x = rng.random(batch) < 0.5
            for row in np.flatnonzero(flip_y):
                block[row] = np.flip(block[row], axis=0)
            for row in np.flatnonzero(flip_x):
                block[row] = np.flip(block[row], axis=1)
            noise = rng.normal(0.0, 1.0, size=block.shape).astype(np.float32) * noise_std
            mixed[:, by : by + BLOCK, bx : bx + BLOCK] = block + noise

    restored = mixed.reshape(batch, 32 * 32, N_COMPONENTS) @ components + mean
    restored = restored.reshape(batch, 32, 32, 131).transpose(0, 3, 1, 2)
    np.clip(restored, 0.0, 32767.0, out=restored)
    return restored.astype(np.float32)


def generate_backgrounds(
    train: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    score_std: np.ndarray,
    bank: np.ndarray,
    rng: np.random.Generator,
    count: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    accepted = np.empty((count, 131, 32, 32), dtype=np.float32)
    correlations = np.empty(count, dtype=np.float32)
    filled = 0
    attempts = 0
    attempt_limit = count * MAX_ATTEMPT_FACTOR
    while filled < count:
        if attempts >= attempt_limit:
            raise RuntimeError(
                f"Only accepted {filled} of {count} patches after {attempts} candidates. "
                "Raise MAX_SPATIAL_CORR or NOISE_SCALE."
            )
        batch = min(BATCH, count - filled + BATCH)
        candidates = synthesize_batch(train, mean, components, score_std, rng, batch)
        nearest = max_spatial_corr(candidates, bank)
        attempts += batch
        for index in range(batch):
            if nearest[index] <= threshold and filled < count:
                accepted[filled] = candidates[index]
                correlations[filled] = nearest[index]
                filled += 1
        print(f"accepted {filled}/{count}  candidates {attempts}", flush=True)
    return accepted, correlations, attempts


def ellipse_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    ry = int(rng.integers(2, 4))
    rx = int(rng.integers(2, 5))
    cy = int(rng.integers(ry, height - ry))
    cx = int(rng.integers(rx, width - rx))
    yy, xx = np.ogrid[:height, :width]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1


def target_mask(height: int, width: int, target: int, rng: np.random.Generator) -> np.ndarray:
    union = np.zeros(height * width, dtype=bool)
    while int(union.sum()) < target:
        union |= ellipse_mask(height, width, rng).reshape(-1)
    chosen = rng.choice(np.flatnonzero(union), size=target, replace=False)
    mask = np.zeros(height * width, dtype=bool)
    mask[chosen] = True
    return mask


def inject(base: np.ndarray, mean: np.ndarray, std: np.ndarray, kind: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cube = base.copy()
    count, bands, height, width = cube.shape
    labels = np.zeros((count, height, width), dtype=np.uint8)
    wavelengths = np.linspace(0.0, 1.0, bands)
    target = max(1, int(round(ANOMALY_FRACTION * height * width)))
    for index in range(count):
        pixels = cube[index].transpose(1, 2, 0).reshape(-1, bands)
        mask = target_mask(height, width, target, rng)
        if kind == "spectral_spike":
            centers = rng.uniform(0.12, 0.88, target)
            widths = rng.uniform(0.018, 0.045, target)
            amplitudes = rng.uniform(5.0, 8.0, target)
            response = np.exp(-0.5 * ((wavelengths[None, :] - centers[:, None]) / widths[:, None]) ** 2)
            response += 0.35 * np.exp(
                -0.5 * ((wavelengths[None, :] - (centers[:, None] + 0.10)) / (widths[:, None] * 1.6)) ** 2
            )
            pixels[mask] += response * (amplitudes[:, None] * std[None, :])
        elif kind == "spectral_substitution":
            centers = rng.uniform(0.15, 0.85, target)
            widths = rng.uniform(0.04, 0.10, target)
            sign = rng.choice(np.array([-1.0, 1.0]), size=target)
            shape = np.exp(-0.5 * ((wavelengths[None, :] - centers[:, None]) / widths[:, None]) ** 2)
            shape += 0.45 * np.exp(
                -0.5
                * ((wavelengths[None, :] - (centers[:, None] + sign[:, None] * 0.18)) / (widths[:, None] * 1.7))
                ** 2
            )
            contrast = rng.uniform(3.0, 5.0, target)
            replacement = mean[None, :] + sign[:, None] * contrast[:, None] * shape * std[None, :]
            pixels[mask] = np.clip(replacement, 0.0, 32767.0)
        elif kind == "band_absorption":
            starts = rng.integers(8, bands - 25, target)
            widths = rng.integers(12, 25, target)
            attenuation = rng.uniform(0.08, 0.25, target)
            selected = pixels[mask].copy()
            for row, (start, width_bands, factor) in enumerate(zip(starts, widths, attenuation)):
                stop = min(bands, int(start + width_bands))
                selected[row, int(start) : stop] *= factor
            pixels[mask] = selected
        else:
            raise ValueError(kind)
        labels[index] = mask.reshape(height, width)
    np.clip(cube, 0.0, 32767.0, out=cube)
    return cube.astype(np.float32), labels


def spectral_angle(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-8))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def training_pair_reference(identity: np.ndarray, rng: np.random.Generator, n_pairs: int = 3000) -> np.ndarray:
    count = identity.shape[0]
    left = rng.integers(0, count, n_pairs)
    right = rng.integers(0, count, n_pairs)
    keep = left != right
    left, right = left[keep], right[keep]
    corrs = [np.sum(identity[left, band] * identity[right, band], axis=1) for band in range(identity.shape[1])]
    return np.max(np.stack(corrs, axis=0), axis=0)


def file_ranges(inventory: list[dict]) -> list[tuple[int, int]]:
    ranges = []
    start = 0
    for item in inventory:
        stop = start + item["patches"]
        ranges.append((start, stop))
        start = stop
    return ranges


def percentile_summary(values: np.ndarray) -> dict:
    qs = np.percentile(values, [50, 90, 95, 99, 100])
    return {
        "median": float(qs[0]),
        "p90": float(qs[1]),
        "p95": float(qs[2]),
        "p99": float(qs[3]),
        "max": float(qs[4]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate leakage-screened HSI evaluation patches.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "synthetic_eval")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--threshold", type=float, default=MAX_SPATIAL_CORR)
    parser.add_argument("--limit", type=int, default=0, help="Generate this many backgrounds and skip saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, inventory = load_training(args.data_dir)
    print(f"loaded {train.shape[0]} training patches from {len(inventory)} clients", flush=True)
    ranges = file_ranges(inventory)
    pca_rng = np.random.default_rng(args.seed)
    mean, components, score_std = fit_pca(train, ranges, pca_rng)

    identity = pack_maps(train)
    bank = dihedral_bank(identity)
    probe = max_spatial_corr(train[:2], bank)
    flipped = max_spatial_corr(np.ascontiguousarray(train[:2, :, :, ::-1]), bank)
    if np.any(probe < 0.99) or np.any(flipped < 0.99):
        raise RuntimeError(f"Leakage detector missed training copies: identity={probe}, flipped={flipped}")

    pair_rng = np.random.default_rng(args.seed + 7)
    pair_corr = training_pair_reference(identity, pair_rng)
    n_backgrounds = args.limit if args.limit > 0 else N_NORMAL + N_PER_SCENARIO * len(SCENARIOS)
    synth_rng = np.random.default_rng(args.seed + 11)
    backgrounds, nearest, attempts = generate_backgrounds(
        train, mean, components, score_std, bank, synth_rng, n_backgrounds, args.threshold
    )

    train_mean = train.mean(axis=(0, 2, 3))
    synth_mean = backgrounds.mean(axis=(0, 2, 3))
    report = {
        "training_patches_screened": int(train.shape[0]),
        "dihedral_templates": int(bank.shape[0]),
        "spatial_bands": BANDS.tolist(),
        "max_spatial_correlation_threshold": args.threshold,
        "candidates_drawn": int(attempts),
        "backgrounds_accepted": int(backgrounds.shape[0]),
        "accepted_nearest_training_correlation": percentile_summary(nearest),
        "distinct_training_pair_correlation": percentile_summary(pair_corr),
        "mean_spectrum_sam_radians": spectral_angle(train_mean, synth_mean),
        "training_value_range": [float(train.min()), float(train.max())],
        "synthetic_value_range": [float(backgrounds.min()), float(backgrounds.max())],
        "copies_at_or_above_0.99": int(np.sum(nearest >= 0.99)),
    }
    print(json.dumps(report, indent=2), flush=True)

    if args.limit > 0:
        print("limit set; evaluation arrays were not written", flush=True)
        return

    n_normal = N_NORMAL
    n_anomaly = N_PER_SCENARIO * len(SCENARIOS)
    if backgrounds.shape[0] != n_normal + n_anomaly:
        raise RuntimeError("Unexpected background count")

    flat = train.transpose(0, 2, 3, 1).reshape(-1, 131).astype(np.float64)
    band_mean = flat.mean(axis=0).astype(np.float32)
    band_std = np.maximum(flat.std(axis=0), 1.0).astype(np.float32)

    anomaly_cubes = []
    anomaly_labels = []
    scenario_rows = []
    cursor = n_normal
    for scenario_index, scenario in enumerate(SCENARIOS):
        part = backgrounds[cursor : cursor + N_PER_SCENARIO]
        cube, labels = inject(part, band_mean, band_std, scenario, args.seed + 100 + scenario_index)
        anomaly_cubes.append(cube)
        anomaly_labels.append(labels)
        scenario_rows.append(
            {
                "scenario": scenario,
                "scenario_id": scenario_index + 1,
                "patches": int(cube.shape[0]),
                "start": int(n_normal + scenario_index * N_PER_SCENARIO),
                "stop": int(n_normal + (scenario_index + 1) * N_PER_SCENARIO),
                "anomaly_pixels_per_patch": int(labels[0].sum()),
                "anomaly_fraction": float(labels.mean()),
                "seed": int(args.seed + 100 + scenario_index),
            }
        )
        cursor += N_PER_SCENARIO

    eval_patches = np.concatenate([backgrounds[:n_normal], *anomaly_cubes], axis=0)
    pixel_labels = np.concatenate(
        [np.zeros((n_normal, 32, 32), dtype=np.uint8), *anomaly_labels],
        axis=0,
    )
    patch_labels = np.zeros(eval_patches.shape[0], dtype=np.uint8)
    patch_labels[n_normal:] = 1
    scenario_ids = np.zeros(eval_patches.shape[0], dtype=np.uint8)
    for row in scenario_rows:
        scenario_ids[row["start"] : row["stop"]] = row["scenario_id"]

    if eval_patches.shape[0] != pixel_labels.shape[0] or patch_labels.sum() != n_anomaly:
        raise RuntimeError("Label alignment failed")
    if np.isnan(eval_patches).any():
        raise RuntimeError("Synthetic cubes contain NaN")

    normalized = mad_normalize_hsi(eval_patches[:8])
    if normalized.shape != (8, 131, 32, 32) or normalized.dtype != np.float32:
        raise RuntimeError(f"mad_normalize_hsi returned {normalized.shape} {normalized.dtype}")
    if float(np.max(np.abs(normalized))) > 1.0 + 1e-5:
        raise RuntimeError("Normalized preview left the [-1, 1] range")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "eval_patches.npy", eval_patches)
    np.save(args.out_dir / "patch_labels.npy", patch_labels)
    np.save(args.out_dir / "scenario_ids.npy", scenario_ids)
    np.save(args.out_dir / "pixel_labels.npy", pixel_labels)
    np.save(args.out_dir / "band_mean.npy", band_mean)
    np.save(args.out_dir / "band_std.npy", band_std)

    manifest = {
        "purpose": "evaluation_only",
        "task": "Differentiate normal and anomalous crop patches with the FedAvg ConvAE3D global model.",
        "format": {
            "eval_patches.npy": "float32 (N, 131, 32, 32), raw stored intensity, same layout as client data",
            "patch_labels.npy": "uint8 (N,), 0=synthetic normal, 1=patch contains a synthetic anomaly",
            "scenario_ids.npy": "uint8 (N,), 0=normal, 1=spectral_spike, 2=spectral_substitution, 3=band_absorption",
            "pixel_labels.npy": "uint8 (N, 32, 32), 1=synthetically anomalous pixel, 0=background",
        },
        "model_input": "mad_normalize_hsi(eval_patches) then unsqueeze a channel axis to (N, 1, 131, 32, 32)",
        "n_patches": int(eval_patches.shape[0]),
        "n_normal": n_normal,
        "n_anomaly": n_anomaly,
        "layout": {
            "normal": [0, n_normal],
            **{row["scenario"]: [row["start"], row["stop"]] for row in scenario_rows},
        },
        "scenarios": scenario_rows,
        "source_clients": inventory,
        "leakage_protocol": (
            "No training patch is copied. Each synthetic patch mixes six training donors in PCA "
            "score space on an 8x8 grid, flips blocks at random, and adds score noise. "
            "Candidates whose maximum spatial correlation with any training patch or its "
            "rotation/mirror exceeds the threshold are discarded. Augmented patches were not "
            "used as donors because they share spatial layout with the training files."
        ),
        "not_for_training": True,
        "seed": args.seed,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out_dir / "leakage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {eval_patches.shape[0]} patches to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
