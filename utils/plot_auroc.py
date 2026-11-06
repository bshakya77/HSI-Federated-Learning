"""Draw patch-level ROC curves from reconstruction scores and patch labels.

``score_summary.json`` stores only the area under the curve. The curve itself
is computed from a score vector and ``patch_labels.npy``. A higher score is
treated as more anomalous, which matches patch MSE and patch spectral angle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = (
    ("All anomalous patches", None, "#0072B2"),
    ("Spectral spike", 1, "#D55E00"),
    ("Spectral substitution", 2, "#E69F00"),
    ("Band absorption", 3, "#009E73"),
)


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return FPR, TPR, and trapezoidal AUROC. Higher scores are more anomalous."""
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC requires both normal and anomalous patches")

    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    true_positives = np.cumsum(ranked_labels == 1)
    false_positives = np.cumsum(ranked_labels == 0)
    tie_ends = np.flatnonzero(np.diff(ranked_scores))
    keep = np.r_[tie_ends, ranked_scores.size - 1]
    true_positives = true_positives[keep]
    false_positives = false_positives[keep]
    fpr = np.r_[0.0, false_positives / n_neg]
    tpr = np.r_[0.0, true_positives / n_pos]
    area = float(np.trapz(tpr, fpr))
    return fpr, tpr, area


def subset(labels: np.ndarray, scenarios: np.ndarray, scores: np.ndarray, scenario_id: int | None):
    if scenario_id is None:
        return labels, scores
    keep = (labels == 0) | (scenarios == scenario_id)
    return labels[keep], scores[keep]


def plot_score(
    labels: np.ndarray,
    scenarios: np.ndarray,
    scores: np.ndarray,
    score_name: str,
    out_path: Path,
) -> dict:
    fig, ax = plt.subplots(figsize=(6.4, 5.6), layout="constrained")
    ax.plot([0, 1], [0, 1], color="#888888", lw=1.0, ls="--", label="Chance (AUROC 0.50)")
    recorded = {}
    for name, scenario_id, color in SCENARIOS:
        y, s = subset(labels, scenarios, scores, scenario_id)
        fpr, tpr, area = roc_curve(y, s)
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{name} ({area:.3f})")
        recorded[name] = {
            "auroc": area,
            "normal_patches": int((y == 0).sum()),
            "anomalous_patches": int((y == 1).sum()),
        }
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Patch-level ROC from {score_name}")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.28)
    ax.set_axisbelow(True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return recorded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot patch-level ROC curves from saved reconstruction scores.")
    parser.add_argument("--score-dir", type=Path, default=ROOT / "outputs" / "synthetic_eval_scores")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "synthetic_eval")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.score_dir
    labels = np.load(args.eval_dir / "patch_labels.npy")
    scenarios = np.load(args.eval_dir / "scenario_ids.npy")
    if labels.shape != scenarios.shape:
        raise ValueError(f"Label length {labels.shape} does not match scenarios {scenarios.shape}")

    available = {
        "patch MSE": args.score_dir / "patch_mse.npy",
        "patch spectral angle": args.score_dir / "patch_sam.npy",
    }
    found = {name: path for name, path in available.items() if path.exists()}
    if not found:
        raise FileNotFoundError(
            "No patch_mse.npy or patch_sam.npy in "
            f"{args.score_dir}. Run scripts/score_synthetic_eval.py first. "
            "score_summary.json cannot draw the curve because it stores only the area."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, path in found.items():
        scores = np.load(path)
        if scores.shape != labels.shape:
            raise ValueError(f"{path.name} has shape {scores.shape}, labels have shape {labels.shape}")
        stem = "auroc_patch_mse.png" if "MSE" in name else "auroc_patch_sam.png"
        summary[name] = plot_score(labels, scenarios, scores, name, out_dir / stem)
        print(f"wrote {out_dir / stem}")

    report_path = out_dir / "auroc_curves.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
