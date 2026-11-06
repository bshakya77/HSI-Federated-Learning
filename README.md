# Federated Robustness Experiments: Gaussian Noise Attack and Defenses

This README records the hyperparameter sets used for the federated learning experiments.

The experiments compare a clean FedAvg baseline, a Gaussian-noise attack against FedAvg, and several defense settings using server-side adaptive clipping, Robust Federated Aggregation (RFA), and a DP-style noise layer.

---

## 1. Common Training Configuration

These settings are shared across all six experiments unless explicitly overridden in the experiment-specific table.

| Parameter | Value |
|---|---:|
| `NUM_CLIENTS` | `7` |
| `NUM_ROUNDS` | `40` |
| `LR` | `1e-4` |
| `LOCAL_EPOCHS` | `5` |
| `CLIPNORM` | `0.0` |
| `LAMBDA_SAM` | `0.05` |
| `SEED` | `42` |
| `BATCH_SIZE` | `4` |
| `VAL_RATIO` | `0.15` |
| `ANOMALY_PCT` | `95.0` |

Notes:

- `CLIPNORM=0.0` disables local client-side gradient clipping.
- Server-side clipping is controlled separately using `SERVER_CLIPPING_METHOD` and `SERVER_ADAPTIVE_CLIP_MULTIPLIER`.
- The reconstruction objective uses the composite loss with `LAMBDA_SAM=0.05`.

---

## 2. Client/Data Manifest

All experiments use the same deterministic client-to-data mapping.

| Client | `CLIENT_ID` | Data file |
|---:|---:|---|
| client-0 | `0` | `/data/AEZ_6.npy` |
| client-1 | `1` | `/data/AEZ_8.npy` |
| client-2 | `2` | `/data/AEZ_9.npy` |
| client-3 | `3` | `/data/AEZ_10.npy` |
| client-4 | `4` | `/data/Split_1_ImageOnly.npy` |
| client-5 | `5` | `/data/Split_2_ImageOnly.npy` |
| client-6 | `6` | `/data/Split_3_ImageOnly.npy` |

Each client uses:

```yaml
BATCH_SIZE: "4"
VAL_RATIO: "0.15"
SEED: "42"
```

---

## 3. Attack Configuration

The Gaussian-noise attack is a model-replacement style attack where malicious clients send:

```text
sent_parameters = current_global_parameters + GaussianNoise(mean, std)
```

The common attack settings for attacked runs are:

| Parameter | Value |
|---|---:|
| `ATTACK_MODE` | `gaussian_noise` |
| `ATTACK_SEED` | `42` |
| `MALICIOUS_CIDS` | `0,2,5,6` |
| `GAUSSIAN_NOISE_MEAN` | `0.0` |
| `GAUSSIAN_NOISE_STD` | `0.2` |
| `SIGN_FLIP_ALPHA` | `1.5` |

Notes:

- `SIGN_FLIP_ALPHA=1.5` is present in the experiment but is not used when `ATTACK_MODE=gaussian_noise`.
- For the clean FedAvg baseline, set `ATTACK_MODE=none` and `MALICIOUS_CIDS=""`.

---

## 4. Experiment Matrix

### Experiment 1: FedAvg baseline without attack

| Parameter | Value |
|---|---:|
| Experiment name | `fedavg_baseline_no_attack` |
| `ATTACK_MODE` | `none` |
| `MALICIOUS_CIDS` | `""` |
| `AGGREGATION_METHOD` | `fedavg` |
| `SERVER_CLIPPING_METHOD` | `none` |
| `SERVER_ADAPTIVE_CLIP_MULTIPLIER` | `1.0` or unused |
| `DP_MODE` | `none` |
| `DP_NOISE_STD` | `0.001` configured but unused |
| `DP_NOISE_MULTIPLIER` | `0.001` configured but unused |
| `DP_SEED` | `42` |
| DP epsilon | Not applicable |
| DP noise scale used | `0.0` |
| Suggested `SCORE_DIR` | `/app/outputs/fedavg_baseline_no_attack` |
| Suggested `LOG_DIR` | `/app/logs/fedavg_baseline_no_attack` |

---

### Experiment 2: FedAvg with Gaussian noise attack

| Parameter | Value |
|---|---:|
| Experiment name | `fedavg_w_gaussian_noise` |
| `ATTACK_MODE` | `gaussian_noise` |
| `MALICIOUS_CIDS` | `0,2,5,6` |
| `GAUSSIAN_NOISE_MEAN` | `0.0` |
| `GAUSSIAN_NOISE_STD` | `0.2` |
| `AGGREGATION_METHOD` | `fedavg` |
| `SERVER_CLIPPING_METHOD` | `none` |
| `SERVER_ADAPTIVE_CLIP_MULTIPLIER` | `1.0` or unused |
| `DP_MODE` | `none` |
| `DP_NOISE_STD` | `0.001` configured but unused |
| `DP_NOISE_MULTIPLIER` | `0.001` configured but unused |
| `DP_SEED` | `42` |
| DP epsilon | Not applicable |
| DP noise scale used | `0.0` |
| Suggested `SCORE_DIR` | `/app/outputs/fedavg_w_gaussian_noise` |
| Suggested `LOG_DIR` | `/app/logs/fedavg_w_gaussian_noise` |

---

### Experiment 3: FedAvg + adaptive clipping, 1.0x multiplier, Gaussian noise attack

| Parameter | Value |
|---|---:|
| Experiment name | `fedavg_w_adaptive_1.0_gaussian_noise` |
| `ATTACK_MODE` | `gaussian_noise` |
| `MALICIOUS_CIDS` | `0,2,5,6` |
| `GAUSSIAN_NOISE_MEAN` | `0.0` |
| `GAUSSIAN_NOISE_STD` | `0.2` |
| `AGGREGATION_METHOD` | `fedavg` |
| `SERVER_CLIPPING_METHOD` | `adaptive` |
| `SERVER_ADAPTIVE_CLIP_MULTIPLIER` | `1.0` |
| `SERVER_FIXED_CLIP_NORM` | `2.0` configured but unused for adaptive clipping |
| `DP_MODE` | `none` |
| `DP_NOISE_STD` | `0.001` configured but unused |
| `DP_NOISE_MULTIPLIER` | `0.001` configured but unused |
| `DP_SEED` | `42` |
| DP epsilon | Not applicable |
| DP noise scale used | `0.0` |
| Suggested `SCORE_DIR` | `/app/outputs/fedavg_w_adaptive_1.0` |
| Suggested `LOG_DIR` | `/app/logs/fedavg_w_adaptive_1.0` |

Adaptive clipping rule:

```text
clip_norm_used = median(client_update_norms) * SERVER_ADAPTIVE_CLIP_MULTIPLIER
clip_norm_used = median(client_update_norms) * 1.0
```

---

### Experiment 4: RFA with Gaussian noise attack

| Parameter | Value |
|---|---:|
| Experiment name | `rfa_w_gaussian_noise` |
| `ATTACK_MODE` | `gaussian_noise` |
| `MALICIOUS_CIDS` | `0,2,5,6` |
| `GAUSSIAN_NOISE_MEAN` | `0.0` |
| `GAUSSIAN_NOISE_STD` | `0.2` |
| `AGGREGATION_METHOD` | `rfa` |
| `RFA_MAX_ITER` | `100` |
| `RFA_TOL` | `1e-6` |
| `SERVER_CLIPPING_METHOD` | `none` |
| `SERVER_ADAPTIVE_CLIP_MULTIPLIER` | `1.0` or unused |
| `DP_MODE` | `none` |
| `DP_NOISE_STD` | `0.001` configured but unused |
| `DP_NOISE_MULTIPLIER` | `0.001` configured but unused |
| `DP_SEED` | `42` |
| DP epsilon | Not applicable |
| DP noise scale used | `0.0` |
| Suggested `SCORE_DIR` | `/app/outputs/rfa_w_gaussian_noise` |
| Suggested `LOG_DIR` | `/app/logs/rfa_w_gaussian_noise` |

---

### Experiment 5: RFA + adaptive clipping, 1.0x multiplier, Gaussian noise attack

This matches the uploaded YAML configuration except for output/log naming if needed.

| Parameter | Value |
|---|---:|
| Experiment name | `rfa_w_adaptive_1.0_gaussian_noise` |
| `ATTACK_MODE` | `gaussian_noise` |
| `MALICIOUS_CIDS` | `0,2,5,6` |
| `GAUSSIAN_NOISE_MEAN` | `0.0` |
| `GAUSSIAN_NOISE_STD` | `0.2` |
| `AGGREGATION_METHOD` | `rfa` |
| `RFA_MAX_ITER` | `100` |
| `RFA_TOL` | `1e-6` |
| `SERVER_CLIPPING_METHOD` | `adaptive` |
| `SERVER_ADAPTIVE_CLIP_MULTIPLIER` | `1.0` |
| `SERVER_FIXED_CLIP_NORM` | `2.0` configured but unused for adaptive clipping |
| `DP_MODE` | `none` |
| `DP_NOISE_STD` | `0.001` configured but unused |
| `DP_NOISE_MULTIPLIER` | `0.001` configured but unused |
| `DP_SEED` | `42` |
| DP epsilon | Not applicable |
| DP noise scale used | `0.0` |
| `SCORE_DIR` | `/app/outputs/rfa_w_adaptive_1.0` |
| `LOG_DIR` | `/app/logs/rfa_w_adaptive_1.0` |

Adaptive clipping rule:

```text
clip_norm_used = median(client_update_norms) * 1.0
```

---

### Experiment 6: RFA + DP layer + adaptive clipping, 1.0x multiplier, Gaussian noise attack

| Parameter | Value |
|---|---:|
| Experiment name | `rfa_w_dp_adaptive_1.0_gaussian_noise` |
| `ATTACK_MODE` | `gaussian_noise` |
| `MALICIOUS_CIDS` | `0,2,5,6` |
| `GAUSSIAN_NOISE_MEAN` | `0.0` |
| `GAUSSIAN_NOISE_STD` | `0.2` |
| `AGGREGATION_METHOD` | `rfa` |
| `RFA_MAX_ITER` | `100` |
| `RFA_TOL` | `1e-6` |
| `SERVER_CLIPPING_METHOD` | `adaptive` |
| `SERVER_ADAPTIVE_CLIP_MULTIPLIER` | `1.0` |
| `SERVER_FIXED_CLIP_NORM` | `2.0` configured but unused for adaptive clipping |
| `DP_MODE` | `clipped` |
| `DP_NOISE_STD` | `0.001` configured but unused when `DP_MODE=clipped` |
| `DP_NOISE_MULTIPLIER` | `0.001` |
| `DP_SEED` | `42` |
| DP epsilon | Not computed from the YAML; requires a privacy accountant and a chosen `delta` |
| DP noise scale used | `DP_NOISE_MULTIPLIER * clip_norm_used / NUM_CLIENTS` |
| DP noise scale formula | `0.001 * clip_norm_used / 7` |
| Suggested `SCORE_DIR` | `/app/outputs/rfa_w_dp_adaptive_1.0` |
| Suggested `LOG_DIR` | `/app/logs/rfa_w_dp_adaptive_1.0` |

For this DP-style clipped update setting, the effective Gaussian noise standard deviation is not a fixed scalar in the setting. It is computed per round as:

```text
clip_norm_used = median(client_update_norms) * 1.0
noise_std_used = DP_NOISE_MULTIPLIER * clip_norm_used / NUM_CLIENTS
noise_std_used = 0.001 * clip_norm_used / 7
```

The experiment provides the noise multiplier and seed, but not a formal privacy budget. Therefore, the exact privacy epsilon cannot be claimed from the experiment alone. To report epsilon, we need to run a privacy accountant with the chosen delta, number of rounds, sampling assumptions, clipping norm sequence, and noise multiplier.

---

## 5. DP Hyperparameter Clarification

The experiment contains two DP-related noise parameters:

| Parameter | Value | Used when |
|---|---:|---|
| `DP_NOISE_STD` | `0.001` | Only when `DP_MODE=standalone` |
| `DP_NOISE_MULTIPLIER` | `0.001` | Only when `DP_MODE=clipped` |

For the listed DP experiment, use:

```yaml
DP_MODE: "clipped"
DP_NOISE_MULTIPLIER: "0.001"
DP_SEED: "42"
```

`DP_NOISE_STD=0.001` cannot be described as the active DP noise scale for the clipped DP run. In clipped mode, the active noise scale is computed from the adaptive clipping norm:

```text
noise_std_used = 0.001 * clip_norm_used / 7
```

---

## 6. Determinism and Manifest Confirmation

All runs are configured as deterministic/reproducible experiment manifests using fixed seeds and fixed client-data mappings.

Deterministic controls:

| Component | Deterministic setting |
|---|---|
| Global training seed | `SEED=42` |
| Attack seed | `ATTACK_SEED=42` |
| DP seed | `DP_SEED=42` |
| Client seeds | Each client uses `SEED=42` |
| Client mapping | Fixed `CLIENT_ID -> DATA_FILE` mapping |
| Number of clients | Fixed at `7` |
| Number of rounds | Fixed at `40` |
| Local epochs | Fixed at `5` |
| Batch size | Fixed at `4` |
| Validation split | Fixed at `0.15` |
| RFA convergence | `RFA_MAX_ITER=100`, `RFA_TOL=1e-6` |
| Output/log isolation | One `SCORE_DIR` and `LOG_DIR` per experiment |

Expected manifest artifacts per run:

```text
logs/<experiment_name>/server_<timestamp>.log
logs/<experiment_name>/client_metrics_<timestamp>.log
outputs/<experiment_name>/
checkpoints/<timestamp>/global_model_final.pt
```

A run should be considered fully reproducible only if the following are preserved with the results:

1. The exact Docker Compose/YAML file used for the run.
2. The exact client-data mapping shown in this README.
3. The generated server log and client metrics log.
4. The final checkpoint.
5. The generated client score outputs.

---

## 7. Minimal YAML Changes Per Experiment

Only the following fields need to change between runs.

| Experiment | `ATTACK_MODE` | `MALICIOUS_CIDS` | `AGGREGATION_METHOD` | `SERVER_CLIPPING_METHOD` | `SERVER_ADAPTIVE_CLIP_MULTIPLIER` | `DP_MODE` | `SCORE_DIR` / `LOG_DIR` suffix |
|---|---|---|---|---|---:|---|---|
| FedAvg baseline | `none` | `""` | `fedavg` | `none` | unused | `none` | `fedavg_baseline_no_attack` |
| FedAvg + Gaussian | `gaussian_noise` | `0,2,5,6` | `fedavg` | `none` | unused | `none` | `fedavg_w_gaussian_noise` |
| FedAvg + adaptive 1.0 + Gaussian | `gaussian_noise` | `0,2,5,6` | `fedavg` | `adaptive` | `1.0` | `none` | `fedavg_w_adaptive_1.0` |
| RFA + Gaussian | `gaussian_noise` | `0,2,5,6` | `rfa` | `none` | unused | `none` | `rfa_w_gaussian_noise` |
| RFA + adaptive 1.0 + Gaussian | `gaussian_noise` | `0,2,5,6` | `rfa` | `adaptive` | `1.0` | `none` | `rfa_w_adaptive_1.0` |
| RFA + DP + adaptive 1.0 + Gaussian | `gaussian_noise` | `0,2,5,6` | `rfa` | `adaptive` | `1.0` | `clipped` | `rfa_w_dp_adaptive_1.0` |

---

## 8. Run Command

From the project root:

```bash
docker compose up --build
```

For repeated experiments, update only the experiment-specific environment variables and output/log directory suffixes before each run.

To avoid mixing artifacts between runs, use a unique `SCORE_DIR` and `LOG_DIR` for every experiment.
