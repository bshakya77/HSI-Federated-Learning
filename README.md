# Server-Client Workflow (Gaussian Noise Attack)

This document explains the end-to-end code flow and file structure in `HSI-FedAvg-Gaussian-Noise` after replacing client-side Sign-Flip with Gaussian Noise attack.

## 1) Project Structure and Responsibilities

- `server/app.py`
  - Reads runtime environment variables.
  - Initializes server logging.
  - Builds Flower strategy via `make_strategy(...)`.
  - Starts Flower server and saves final checkpoint.

- `shared/server.py`
  - Defines strategy `FedAvgWithClientMetrics` (extends Flower `FedAvg`).
  - Builds initial global model parameters from `ConvAE3D`.
  - Sends fit/evaluate config to clients.
  - Logs per-client and global update diagnostics.

- `client/train.py`
  - Initializes client logging.
  - Loads local data and performs train/val split.
  - Creates `HSIClient` and starts Flower client process.

- `shared/client.py`
  - Defines `HSIClient` train/evaluate logic.
  - Converts model params between PyTorch and NumPy.
  - Applies malicious behavior for selected clients:
    - Gaussian noise update generation on malicious clients.

- `shared/model.py`
  - Defines `ConvAE3D` and `composite_loss`.

- `shared/data.py`
  - Data normalization and train/val split helpers.

- `docker-compose.yml`
  - Orchestrates one server and multiple clients.
  - Provides environment variables for model/attack configuration.

## 2) Startup Flow

1. `server/app.py` reads env vars:
   - Training: `NUM_CLIENTS`, `NUM_ROUNDS`, `LR`, `LOCAL_EPOCHS`, `CLIPNORM`, `LAMBDA_SAM`
   - Attack: `MALICIOUS_CIDS`, `GAUSSIAN_NOISE_MEAN`, `GAUSSIAN_NOISE_STD`
   - Eval/output: `ANOMALY_PCT`, `SCORE_DIR`, `CHECKPOINT_DIR`, `SERVER_PORT`
2. Server creates strategy using `shared/server.py::make_strategy(...)`.
3. Strategy initializes server-side global model (`ConvAE3D`) and passes initial parameters to Flower.
4. Clients start from `client/train.py`, load local data, and connect to server.

## 3) Round-Level Fit Flow (Training + Attack Injection)

Per communication round:

1. Strategy sends global parameters + fit config to clients:
   - `lr`, `local_epochs`, `clipnorm`, `lambda_sam`
   - `malicious_cids`, `gaussian_noise_mean`, `gaussian_noise_std`
2. In `HSIClient.fit(...)`:
   - Client loads global parameters.
   - Performs local model training for configured epochs.
   - Computes local `updated_params`.
3. Malicious behavior in `shared/client.py`:
   - If `cid in malicious_cids`, client does not send honest local update.
   - It samples Gaussian noise per parameter tensor:
     - `noise ~ N(gaussian_noise_mean, gaussian_noise_std)`
   - Sent parameter becomes:
     - `sent_param = global_param + noise`
4. Honest client behavior:
   - Sends `updated_params` directly.
5. Client returns:
   - Parameters to aggregate
   - Number of local examples
   - Train and diagnostics metrics (loss, norms, cosine, malicious flag)

## 4) Server Aggregation and Diagnostics

In `FedAvgWithClientMetrics.aggregate_fit(...)`:

1. Before aggregation:
   - Computes each client delta (client params - previous global params).
   - Logs per-client diagnostics to client metrics log.
2. Aggregates updates with standard Flower FedAvg:
   - `super().aggregate_fit(...)`
3. After aggregation:
   - Stores `_latest_aggregated_parameters`.
   - Logs global shift norm and direction diagnostics.

## 5) Round-Level Evaluate Flow

1. Server sends evaluate config:
   - `lambda_sam`, `anomaly_pct`, `save_scores`, `score_dir`, `server_round`
2. Each client runs `evaluate(...)` on local validation split:
   - Computes `val_loss`, `val_mse`, `val_sam`
   - Computes patch-level percentile/anomaly metrics
   - Optionally saves patch score arrays
3. Flower aggregates evaluation metrics across clients.

## 6) Parameter/Message Format

- Local model format: PyTorch `state_dict`.
- Transport-ready local format: list of NumPy arrays.
- Flower transport: `Parameters` message over gRPC.

Conversions:

- Client:
  - `get_parameters(model)`: state_dict -> NumPy arrays
  - `set_parameters(model, params)`: NumPy arrays -> state_dict
- Server:
  - `ndarrays_to_parameters(...)`
  - `parameters_to_ndarrays(...)`

## 7) Logging and Checkpoint Flow

- Server logs:
  - Main run log (`server_<timestamp>.log`)
  - Client metrics log (`client_metrics_<timestamp>.log`)
- Client logs:
  - One log file per client run (`client_<id>_<timestamp>.log`)
- Final checkpoint:
  - Saved once after all rounds complete in `server/app.py`.
  - Path: `CHECKPOINT_DIR/<run_timestamp>/global_model_final.pt`
  - Contains run timestamp, number of rounds, and final model `state_dict`.

## 8) Notes for Correct Gaussian Attack Usage

- Attack is controlled centrally by server fit config and applied only in `shared/client.py`.
- `MALICIOUS_CIDS` selects which clients are malicious.
- `GAUSSIAN_NOISE_MEAN` and `GAUSSIAN_NOISE_STD` control noise distribution.
- FedAvg aggregation and checkpoint logic remain unchanged.
