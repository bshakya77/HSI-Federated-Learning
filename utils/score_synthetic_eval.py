"""Score the synthetic evaluation set with a saved ConvAE3D global model.

The checkpoint may be a state_dict, a dict with a ``state_dict`` entry, or a
list of parameter tensors in ``ConvAE3D.state_dict()`` order. Patches are
median/MAD normalized with the same function the clients use, then scored by
patch reconstruction MSE and spectral angle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.generate_synthetic_eval import mad_normalize_hsi  # noqa: E402


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both classes")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    start = 0
    while start < scores.shape[0]:
        stop = start + 1
        while stop < scores.shape[0] and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average = 0.5 * (start + 1 + stop)
        ranks[order[start:stop]] = average
        start = stop
    sum_pos = float(ranks[labels == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_model(checkpoint: Path):
    import torch

    from shared.model import ConvAE3D

    model = ConvAE3D()
    obj = torch.load(checkpoint, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        model.load_state_dict(obj["state_dict"])
    elif isinstance(obj, dict):
        model.load_state_dict(obj)
    else:
        keys = list(model.state_dict().keys())
        state = {key: torch.as_tensor(value) for key, value in zip(keys, obj)}
        model.load_state_dict(state)
    model.eval()
    return model


def reconstruction_scores(model, patches: np.ndarray, batch_size: int = 4) -> tuple[np.ndarray, np.ndarray]:
    import torch

    from shared.client import per_patch_mse, per_patch_sam

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    normalized = mad_normalize_hsi(patches)
    mse_parts = []
    sam_parts = []
    with torch.no_grad():
        for start in range(0, normalized.shape[0], batch_size):
            batch = torch.from_numpy(normalized[start : start + batch_size]).unsqueeze(1).to(device)
            restored = model(batch)
            mse_parts.append(per_patch_mse(restored, batch).cpu().numpy())
            sam_parts.append(per_patch_sam(restored, batch).cpu().numpy())
    return np.concatenate(mse_parts), np.concatenate(sam_parts)


def class_summary(labels: np.ndarray, scores: np.ndarray) -> dict:
    normal = scores[labels == 0]
    anomaly = scores[labels == 1]
    return {
        "auroc": auroc(labels, scores),
        "normal_median": float(np.median(normal)),
        "anomaly_median": float(np.median(anomaly)),
        "normal_p95": float(np.percentile(normal, 95)),
        "anomaly_p95": float(np.percentile(anomaly, 95)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score synthetic HSI patches with the global autoencoder.")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "synthetic_eval")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patches = np.load(args.eval_dir / "eval_patches.npy")
    labels = np.load(args.eval_dir / "patch_labels.npy")
    scenarios = np.load(args.eval_dir / "scenario_ids.npy")
    model = load_model(args.checkpoint)
    mse, sam = reconstruction_scores(model, patches, batch_size=args.batch_size)
    summary = {
        "mse": class_summary(labels, mse),
        "sam": class_summary(labels, sam),
        "by_scenario": {},
    }
    names = {1: "spectral_spike", 2: "spectral_substitution", 3: "band_absorption"}
    normal = labels == 0
    for scenario_id, name in names.items():
        mask = normal | (scenarios == scenario_id)
        summary["by_scenario"][name] = {
            "mse_auroc": auroc(labels[mask], mse[mask]),
            "sam_auroc": auroc(labels[mask], sam[mask]),
            "patches": int(mask.sum()),
        }
    out_dir = args.out_dir or args.eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "patch_mse.npy", mse.astype(np.float32))
    np.save(out_dir / "patch_sam.npy", sam.astype(np.float32))
    (out_dir / "score_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
