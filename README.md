# Server-Client Workflow

This document explains how server and clients communicate model parameters, configs, and metrics in `HSI-FedAvg-Sign-Flip-server-latest`.

## 1) Startup and Server Initialization

1. The server process starts from `server/app.py`.
2. Logging paths are prepared in `_setup_run_logging()`:
   - Main server log: `server_<timestamp>.log`
   - Client-metrics log: `client_metrics_<timestamp>.log`
3. Runtime settings are loaded from environment variables:
   - `NUM_CLIENTS`, `NUM_ROUNDS`, `LR`, `LOCAL_EPOCHS`, `CLIPNORM`
   - `LAMBDA_SAM`, `SEED`, `MALICIOUS_CIDS`, `SIGN_FLIP_ALPHA`
   - `ANOMALY_PCT`, `SCORE_DIR`, `SERVER_PORT`
4. `make_strategy(...)` is called to build a Flower strategy.

## 2) How Initial Global Parameters Are Built

Inside `shared/server.py` (`make_strategy`):

1. Server seeds Python/NumPy/PyTorch (`seed_everything(seed)`).
2. Server creates a fresh `ConvAE3D()` model.
3. Model `state_dict()` tensors are converted to NumPy arrays.
4. Arrays are wrapped into Flower `initial_parameters`.
5. Strategy `FedAvgWithClientMetrics(...)` is created with those initial parameters.

Result: the first global model is server-initialized (not copied from any client).

## 3) Round-Level Fit Workflow (Client Training)

For each server round:

1. Flower calls strategy `configure_fit(...)`.
2. Current global parameters are captured on server (`_last_global_ndarrays`) for diagnostics.
3. Strategy attaches fit config via `on_fit_config_fn`:
   - Learning setup (`lr`, `local_epochs`, `clipnorm`, `lambda_sam`)
   - Attack settings (`malicious_cids`, `sign_flip_alpha`)
   - Tracking (`server_round`)
4. Server sends to selected clients:
   - Current global parameters
   - Round fit config

### Client-side `fit(...)` steps

1. Client receives `parameters` and `config`.
2. Client copies incoming global parameters (`global_params`) and loads them into local model (`set_parameters`).
3. Client performs local training for `local_epochs`.
4. Client computes `updated_params` from trained model (`get_parameters`).
5. If client is malicious (`cid` in `malicious_cids`), it applies sign-flip scaling:
   - Honest update: `delta = local - global`
   - Poisoned update: `-alpha * delta`
   - Sent params: `global + poisoned_delta`
6. Client returns to server:
   - `sent_params` (weights to aggregate)
   - `num_examples` (`len(train_dataset)`)
   - Fit metrics (`train_loss`, `train_mse`, `train_sam`, attack/diagnostic fields)

## 4) Server Aggregation and Diagnostics

Inside strategy `aggregate_fit(...)`:

1. Server receives `(client_proxy, fit_res)` for each responding client.
2. If client-metrics logger is active and base parameters exist:
   - Server computes each client delta against previous global model.
   - Logs per-client norms, malicious flag, alpha, cosine, train metrics.
   - Logs summary metrics (pairwise cosine mean, honest vs malicious direction).
3. Server runs standard Flower FedAvg by calling:
   - `super().aggregate_fit(server_round, results, failures)`
4. Aggregated global parameters are stored as `_latest_aggregated_parameters`.
5. Server logs global shift diagnostics (`norm`, cosine vs honest/malicious means).

Important: aggregation is weighted FedAvg from Flower base strategy.

## 5) Round-Level Evaluate Workflow (Client Validation)

After fit aggregation each round:

1. Strategy provides evaluate config via `on_evaluate_config_fn`:
   - `lambda_sam`, `anomaly_pct`, `save_scores`, `score_dir`, `server_round`
2. Server sends current global parameters + evaluate config to clients.
3. Each client `evaluate(...)`:
   - Loads received parameters into local model.
   - Runs validation.
   - Computes loss and metrics (`val_mse`, `val_sam`, percentile/anomaly metrics).
   - Optionally saves patch-level arrays to `score_dir`.
   - Returns `(val_loss, num_examples, metrics)` to server.
4. Server aggregates evaluation outputs through Flower evaluation aggregation.

## 6) Communication Format (Under the Hood)

- Local model format: PyTorch `state_dict`.
- Client/server exchange format in this code: list of NumPy ndarrays.
- Flower transport format: `Parameters` message over gRPC.

Conversions used:

- `get_parameters(model)`: `state_dict` -> ndarrays
- `set_parameters(model, params)`: ndarrays -> `state_dict`
- `ndarrays_to_parameters(...)` / `parameters_to_ndarrays(...)` on server side

## 7) End of Training and Checkpoint Save

After `fl.server.start_server(...)` finishes all rounds in `server/app.py`:

1. Server resolves checkpoint base path from `CHECKPOINT_DIR` (default `/app/checkpoints`).
2. Creates run folder `checkpoints/<run_timestamp>/`.
3. Reads final aggregated parameters from strategy (`_latest_aggregated_parameters`).
4. Loads them into `ConvAE3D`.
5. Saves `global_model_final.pt` containing:
   - `run_ts`
   - `num_rounds`
   - `state_dict`

This checkpoint write happens once per server run, after all rounds complete.

## 8) Key Files and Responsibilities

- `server/app.py`
  - Runtime env/config, server startup, logging setup, final checkpoint save
- `shared/server.py`
  - Strategy construction, initial global params, fit/evaluate config, aggregation diagnostics
- `shared/client.py`
  - Parameter conversion helpers, local train/evaluate loops, malicious sign-flip behavior, client metrics

