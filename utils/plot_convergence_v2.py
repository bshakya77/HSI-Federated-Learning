"""
Visualize Loss, val_mse, and val_sam across federated rounds from Flower log.
Creates three separate line plots and saves as PNG with dpi=640.
"""
import re
from pathlib import Path
import matplotlib.pyplot as plt

# --------- CONFIG ---------
LOG_PATH = Path("../logs/v2/log-FedAvg-20260331_145525.txt")  # change if needed
OUT_DIR = Path("../results/plots_convergence_sign_flip_augment_v2_m5")
DPI = 640
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------- PARSER ---------
# Parses Flower log SUMMARY: "History (loss, distributed)" and "History (metrics, distributed, evaluate)"
if not LOG_PATH.exists():
    raise FileNotFoundError(f"Could not find log file at: {LOG_PATH.resolve()}")

text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")
lines = text.splitlines()

# 1. Parse loss from "round N: value" (use search to handle "INFO :" prefix in log lines)
round_loss_pat = re.compile(r"round\s+(\d+):\s*([0-9.eE+-]+)")
rounds, loss_values = [], []
in_loss_section = False
for line in lines:
    if "History (loss, distributed)" in line:
        in_loss_section = True
        continue
    if in_loss_section:
        m = round_loss_pat.search(line)  # search, not match: handles "INFO :" prefix
        if m:
            rounds.append(int(m.group(1)))
            loss_values.append(round(float(m.group(2)), 3))
        elif line.strip() and not round_loss_pat.search(line):
            in_loss_section = False

# 2. Extract val_mse and val_sam from evaluate block (bracket-matching to avoid wrong lists)
evaluate_start = text.find("History (metrics, distributed, evaluate):")
evaluate_block = text[evaluate_start:] if evaluate_start >= 0 else text

pair_pat = re.compile(r"\(\s*(\d+)\s*,\s*([0-9.eE+-]+)\s*\)")


def _extract_list_by_brackets(block_text, key_name):
    """Extract list content for 'key_name': [...] using bracket matching."""
    start = block_text.find(f"'{key_name}': [")
    if start < 0:
        return []
    list_start = block_text.find("[", start) + 1
    depth, i = 1, list_start
    while i < len(block_text) and depth > 0:
        c = block_text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return block_text[list_start:i]
        i += 1
    return ""


def extract_metric_pairs(block_text, metric_name):
    """Extract (round, value) pairs from 'metric_name': [(1, v), (2, v), ...]"""
    list_content = _extract_list_by_brackets(block_text, metric_name)
    pairs = pair_pat.findall(list_content)
    return [(int(r), round(float(v), 3)) for r, v in pairs]


val_mse_pairs = extract_metric_pairs(evaluate_block, "val_mse")
val_sam_pairs = extract_metric_pairs(evaluate_block, "val_sam")

if rounds and val_mse_pairs and val_sam_pairs:
    round_to_val = dict(val_mse_pairs)
    val_mse = [round_to_val.get(r) for r in rounds]
    round_to_val = dict(val_sam_pairs)
    val_sam = [round_to_val.get(r) for r in rounds]
else:
    val_mse = [v for _, v in sorted(val_mse_pairs)] if val_mse_pairs else []
    val_sam = [v for _, v in sorted(val_sam_pairs)] if val_sam_pairs else []
    if not rounds and val_mse_pairs:
        rounds = [r for r, _ in sorted(val_mse_pairs)]


def _filter_xy(x, y):
    xx, yy = [], []
    for a, b in zip(x, y):
        if b is None:
            continue
        xx.append(a)
        yy.append(b)
    return xx, yy


def _format_value(v):
    """Format value for annotation: use scientific notation for very small values."""
    if abs(v) < 1e-3 or abs(v) >= 1e4:
        return f"{v:.3e}"
    return f"{v:.3f}"


def plot_and_save(x, y, title, ylabel, filename):
    x, y = _filter_xy(x, y)
    if not x:
        raise ValueError(f"No data found for: {ylabel}. Check log file format.")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, y, marker="o", linewidth=2, markersize=6, color="C0")
    # Add value labels at each point with alternating vertical offset to avoid overlap
    y_range = max(y) - min(y) if len(y) > 1 else 1.0
    base_offset = y_range * 0.04  # 4% of data range as base padding
    for idx, (xi, yi) in enumerate(zip(x, y)):
        # Alternate offset: even indices above, odd indices further above
        offset = base_offset if idx % 2 == 0 else base_offset * 1.2
        ax.annotate(
            _format_value(yi),
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 8 + (idx % 2) * 14),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
        )
    ax.set_xlabel("Federated Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)  # Plot each round explicitly (1, 2, 3, ... 20), not 2.5 intervals
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend([ylabel])
    plt.tight_layout()
    out_path = OUT_DIR / filename
    plt.savefig(out_path, dpi=DPI)
    plt.show()
    print(f"Saved: {out_path.resolve()}")


# Create three separate line plots (loss, val_mse, val_sam)
plot_and_save(rounds, loss_values, "Loss across Federated Rounds", "loss", "loss.png")
plot_and_save(rounds, val_mse, "Validation MSE across Federated Rounds", "val_mse", "val_mse.png")
plot_and_save(rounds, val_sam, "Validation SAM across Federated Rounds", "val_sam", "val_sam.png")