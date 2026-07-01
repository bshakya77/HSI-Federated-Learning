import re
from pathlib import Path
import matplotlib.pyplot as plt


LOG_PATH = ".\\logs\\server-sign-flip-logs\\sign-flip-logs\\m5_normal_server_20260428_161300.log"   # <-- change if needed
OUT_DIR = ".\\graphs\\plots_convergence_normal_m5"
DPI = 640
# Wider figure so rounds are spaced out horizontally; markers read clearly on the line.
FIG_SIZE_INCH = (20, 5.5)


def parse_history_loss(text: str):
    """Parse 'History (loss, distributed): round k: v' block."""
    # Capture lines like: "round 1: 0.05837"
    pairs = re.findall(r"round\s+(\d+):\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", text)
    return {int(r): float(v) for r, v in pairs}


def parse_metrics_block(text: str, key: str):
    """
    Parse distributed evaluate metrics lines like:
      'val_mse': [(1, 0.0535), (2, 0.0414), ...]
    """
    # Find "'val_mse': [(1, ...), (2, ...)]" including newlines
    m = re.search(rf"'{re.escape(key)}'\s*:\s*\[(.*?)\]\s*(?:,|\}})", text, flags=re.DOTALL)
    if not m:
        return {}

    block = m.group(1)
    pairs = re.findall(r"\(\s*(\d+)\s*,\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*\)", block)
    return {int(r): float(v) for r, v in pairs}


def plot_series(d: dict, title: str, ylabel: str, out_path: Path, dpi: int = 640):
    rounds = sorted(d.keys())
    values = [d[r] for r in rounds]

    plt.figure(figsize=FIG_SIZE_INCH)
    plt.plot(rounds, values, marker="o", linewidth=2, markersize=5, label=ylabel, zorder=2)
    ax = plt.gca()
    ax.set_xticks(rounds)
    if len(rounds) > 20:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    plt.margins(x=0.02, y=0.04)
    plt.xlabel("Federated Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()

    # annotate each point with its value (stagger offsets so labels do not overlap)
    for i, (r, v) in enumerate(zip(rounds, values)):
        off_y = 8 + 9 * (i % 3)
        off_x = 4 * (i % 2) - 2
        plt.annotate(
            f"{v:.3f}",
            (r, v),
            textcoords="offset points",
            xytext=(off_x, off_y),
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main():
    text = Path(LOG_PATH).read_text(encoding="utf-8", errors="ignore")

    loss_dict = parse_history_loss(text)
    val_mse_dict = parse_metrics_block(text, "val_mse")
    val_sam_dict = parse_metrics_block(text, "val_sam")

    # Print dictionaries
    print("\n=== loss (History (loss, distributed)) ===")
    print(loss_dict)

    print("\n=== val_mse (History (metrics, distributed, evaluate)) ===")
    print(val_mse_dict)

    print("\n=== val_sam (History (metrics, distributed, evaluate)) ===")
    print(val_sam_dict)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot & save (3 separate figures)
    plot_series(loss_dict, "Global Loss Convergence", "loss", out_dir / "global_loss.png", dpi=DPI)
    plot_series(val_mse_dict, "Validation MSE Convergence", "val_mse", out_dir / "val_mse.png", dpi=DPI)
    plot_series(val_sam_dict, "Validation SAM Convergence", "val_sam", out_dir / "val_sam.png", dpi=DPI)

    print(f"\nSaved plots to: {out_dir.resolve()}")
    print(" - global_loss.png")
    print(" - val_mse.png")
    print(" - val_sam.png")


if __name__ == "__main__":
    main()