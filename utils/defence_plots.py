import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Paths
SERVER_LOG = "/home/ren-admin/cyber-ag/Defence/HSI-FedAvg-Sign-Flip-server-latest-fixed_clipping/logs/fedavg_attack_fixedclip3/server_20260429_182537.log"
CLIENT_METRICS_LOG = "/home/ren-admin/cyber-ag/Defence/HSI-FedAvg-Sign-Flip-server-latest-fixed_clipping/logs/fedavg_attack_fixedclip3/client_metrics_20260429_182537.log"

OUT_DIR = Path("/home/ren-admin/cyber-ag/Defence/HSI-FedAvg-Sign-Flip-server-latest-fixed_clipping/logs/fedavg_attack_fixedclip3/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Parse distributed loss from server log
def parse_server_loss(server_log_path: Path) -> pd.DataFrame:
    pattern = re.compile(r"round\s+(\d+):\s+([0-9.]+)")

    rows = []
    with open(server_log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                rows.append({
                    "round": int(match.group(1)),
                    "distributed_loss": float(match.group(2)),
                })

    return pd.DataFrame(rows).sort_values("round")


# Parse defense summary lines
def parse_defense_summary(client_metrics_path: Path) -> pd.DataFrame:
    pattern = re.compile(
        r"\[ROUND\s+(\d+)\]\s+defense_summary\s+"
        r"aggregation=(\w+)\s+"
        r"clipping=(\w+)\s+"
        r"clip_norm=([0-9.]+)\s+"
        r"pre_clip_mean=([0-9.]+)\s+"
        r"pre_clip_max=([0-9.]+)\s+"
        r"post_clip_mean=([0-9.]+)\s+"
        r"post_clip_max=([0-9.]+)\s+"
        r"honest_vs_mal_cos=([-0-9.]+)\s+"
        r"global_shift_norm=([0-9.]+)\s+"
        r"cos_to_honest=([-0-9.]+)\s+"
        r"cos_to_mal=([-0-9.]+)"
    )

    rows = []
    with open(client_metrics_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                rows.append({
                    "round": int(match.group(1)),
                    "aggregation": match.group(2),
                    "clipping": match.group(3),
                    "clip_norm": float(match.group(4)),
                    "pre_clip_mean": float(match.group(5)),
                    "pre_clip_max": float(match.group(6)),
                    "post_clip_mean": float(match.group(7)),
                    "post_clip_max": float(match.group(8)),
                    "honest_vs_mal_cos": float(match.group(9)),
                    "global_shift_norm": float(match.group(10)),
                    "cos_to_honest": float(match.group(11)),
                    "cos_to_mal": float(match.group(12)),
                })

    return pd.DataFrame(rows).sort_values("round")


# Parse per-client training metrics
def parse_client_metrics(client_metrics_path: Path) -> pd.DataFrame:
    pattern = re.compile(
        r"\[ROUND\s+(\d+)\]\s+client\s+"
        r"proxy_cid=(\S+)\s+"
        r"metric_cid=(-?\d+)\s+"
        r"malicious=(True|False)\s+"
        r"alpha=([0-9.]+)\s+"
        r"num_examples=(\d+)\s+"
        r"pre_update_norm=([0-9.]+)\s+"
        r"post_update_norm=([0-9.]+)\s+"
        r"server_pre_clip_norm=([0-9.]+)\s+"
        r"server_post_clip_norm=([0-9.]+)\s+"
        r"update_cosine=([-0-9.]+)\s+"
        r"train_loss=([0-9.]+)\s+"
        r"train_mse=([0-9.]+)\s+"
        r"train_sam=([0-9.]+)"
    )

    rows = []
    with open(client_metrics_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                rows.append({
                    "round": int(match.group(1)),
                    "proxy_cid": match.group(2),
                    "metric_cid": int(match.group(3)),
                    "malicious": match.group(4) == "True",
                    "alpha": float(match.group(5)),
                    "num_examples": int(match.group(6)),
                    "pre_update_norm": float(match.group(7)),
                    "post_update_norm": float(match.group(8)),
                    "server_pre_clip_norm": float(match.group(9)),
                    "server_post_clip_norm": float(match.group(10)),
                    "update_cosine": float(match.group(11)),
                    "train_loss": float(match.group(12)),
                    "train_mse": float(match.group(13)),
                    "train_sam": float(match.group(14)),
                })

    return pd.DataFrame(rows).sort_values(["round", "metric_cid"])


# Plot helper
def save_line_plot(df, x, y, title, ylabel, filename):
    plt.figure(figsize=(9, 5))
    plt.plot(df[x], df[y], marker="o")
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300)
    plt.close()


# Run parsing
loss_df = parse_server_loss(SERVER_LOG)
defense_df = parse_defense_summary(CLIENT_METRICS_LOG)
client_df = parse_client_metrics(CLIENT_METRICS_LOG)

print("Server loss rows:", len(loss_df))
print("Defense summary rows:", len(defense_df))
print("Client metric rows:", len(client_df))

loss_df.to_csv(OUT_DIR / "distributed_loss.csv", index=False)
defense_df.to_csv(OUT_DIR / "defense_summary.csv", index=False)
client_df.to_csv(OUT_DIR / "client_metrics.csv", index=False)


# Aggregated client metrics per round
round_client_avg = (
    client_df
    .groupby("round")
    .agg(
        avg_train_loss=("train_loss", "mean"),
        avg_train_mse=("train_mse", "mean"),
        avg_train_sam=("train_sam", "mean"),
        avg_server_pre_clip_norm=("server_pre_clip_norm", "mean"),
        avg_server_post_clip_norm=("server_post_clip_norm", "mean"),
    )
    .reset_index()
)

round_client_avg.to_csv(OUT_DIR / "round_client_averages.csv", index=False)


# 7. Generate convergence plots
if not loss_df.empty:
    save_line_plot(
        loss_df,
        x="round",
        y="distributed_loss",
        title="Distributed Loss Convergence",
        ylabel="Distributed Loss",
        filename="distributed_loss_convergence.png",
    )

if not round_client_avg.empty:
    save_line_plot(
        round_client_avg,
        x="round",
        y="avg_train_loss",
        title="Average Client Training Loss",
        ylabel="Average Train Loss",
        filename="avg_client_train_loss.png",
    )

    save_line_plot(
        round_client_avg,
        x="round",
        y="avg_train_mse",
        title="Average Client Training MSE",
        ylabel="Average Train MSE",
        filename="avg_client_train_mse.png",
    )

    save_line_plot(
        round_client_avg,
        x="round",
        y="avg_train_sam",
        title="Average Client Training SAM",
        ylabel="Average Train SAM",
        filename="avg_client_train_sam.png",
    )

if not defense_df.empty:
    plt.figure(figsize=(9, 5))
    plt.plot(defense_df["round"], defense_df["pre_clip_mean"], marker="o", label="Pre-clip mean")
    plt.plot(defense_df["round"], defense_df["post_clip_mean"], marker="o", label="Post-clip mean")
    plt.xlabel("Round")
    plt.ylabel("Update Norm")
    plt.title("Server-Side Clipping Effect: Mean Update Norm")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pre_post_clip_mean_norm.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(defense_df["round"], defense_df["pre_clip_max"], marker="o", label="Pre-clip max")
    plt.plot(defense_df["round"], defense_df["post_clip_max"], marker="o", label="Post-clip max")
    plt.xlabel("Round")
    plt.ylabel("Update Norm")
    plt.title("Server-Side Clipping Effect: Max Update Norm")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pre_post_clip_max_norm.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(defense_df["round"], defense_df["cos_to_honest"], marker="o", label="Cosine to honest mean")
    plt.plot(defense_df["round"], defense_df["cos_to_mal"], marker="o", label="Cosine to malicious mean")
    plt.xlabel("Round")
    plt.ylabel("Cosine Similarity")
    plt.title("Global Update Alignment: Honest vs Malicious")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "honest_vs_malicious_alignment.png", dpi=300)
    plt.close()

    save_line_plot(
        defense_df,
        x="round",
        y="global_shift_norm",
        title="Global Update Shift Norm",
        ylabel="Global Shift Norm",
        filename="global_shift_norm.png",
    )


# Per-client loss curves
if not client_df.empty:
    plt.figure(figsize=(10, 6))

    for cid, group in client_df.groupby("metric_cid"):
        label = f"Client {cid}"
        if group["malicious"].iloc[0]:
            label += " malicious"
        plt.plot(group["round"], group["train_loss"], marker="o", label=label)

    plt.xlabel("Round")
    plt.ylabel("Train Loss")
    plt.title("Per-Client Training Loss Across Rounds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_client_train_loss.png", dpi=300)
    plt.close()


print(f"Saved CSVs and plots to: {OUT_DIR.resolve()}")