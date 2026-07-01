# HSI-FedAvg-Gaussian-Noise: Implementation Notes

This README keeps historical implementation notes and log interpretation context.

- `HSI-FedAvg-Gaussian-Noise/logs/docker-logs/server_20260425_235537.log`

Current attack flow for this project is Gaussian noise (not Sign-Flip). For the up-to-date execution flow, see `workflow-v1.md`.

---

## What the implementation is doing

### 1) Global model initialization (server vs client)

- **The global model is initialized on the server** (not from any client).
- The Flower server log confirms this with: *"Using initial global parameters provided by strategy"*.

Code location:
- `shared/server.py` → `make_strategy(...)`
  - Creates `model = ConvAE3D()`
  - Converts `state_dict()` tensors to numpy arrays
  - Passes them as Flower `initial_parameters`

Implication:
- The initial global model weights are the server-side PyTorch initialization of `ConvAE3D()`.

---

### 2) Seeding / reproducibility

Seeding is implemented on **server and clients**.

#### Server
- Seed is read from env var `SEED` (default `42`) in `server/app.py`.
- `shared/server.py` calls `seed_everything(seed)` **before** creating `ConvAE3D()` so initial global weights become deterministic (best effort).

#### Clients
- Seed is read from env var `SEED` (default `42`) in `client/train.py`.
- Each client uses a **distinct but reproducible** seed: `SEED + CLIENT_ID`.
- Clients seed:
  - Python/NumPy/PyTorch RNG (`seed_everything(SEED + cid)`)
  - DataLoader shuffling by passing a `torch.Generator()` seeded with `SEED + cid`

Notes:
- GPU runs can still show nondeterminism depending on hardware/ops, but the code is doing the standard “best effort” steps.

---

### 3) Data split correctness

Data split is deterministic given a seed.

Code location:
- `shared/data.py` → `split_train_val(...)`
  - Uses `sklearn.model_selection.train_test_split(..., random_state=seed, shuffle=True)`

Client usage:
- `client/train.py` passes `seed=SEED + CLIENT_ID` into the split function.

Implication:
- Under a fixed run seed, each client gets a repeatable train/val split; different clients can have different (but deterministic) splits.

---

### 4) FedAvg aggregation correctness

Aggregation is **plain Flower FedAvg**.

Code location:
- `shared/server.py` defines `FedAvgWithClientMetrics(fl.server.strategy.FedAvg)`
  - `aggregate_fit(...)` logs diagnostics and then calls:
    - `super().aggregate_fit(server_round, results, failures)`

Implication:
- Server-side aggregation is standard weighted FedAvg (by `num_examples`).
- No custom robust aggregation is applied in `aggregate_fit`.

---

### 5) Is clipping implemented?

There is **no clipping of updates on the server before aggregation**.

There is **optional client-side gradient clipping** during local training:
- `shared/client.py`:
  - `torch.nn.utils.clip_grad_norm_(..., max_norm=clipnorm)` if `clipnorm > 0`

Implication:
- If `CLIPNORM=0.0`, gradient clipping is disabled.
- Even if enabled, this is **gradient clipping** during local training, not “update clipping before FedAvg on the server”.

---

## Log analysis: `server_20260425_235537.log`

### A) Global init confirmation

The log contains:
- `[INIT] Using initial global parameters provided by strategy`

This matches the server-initialized global model described above.

---

### B) Nature of honest vs malicious clients (sign-flip attack)

The client implementation performs **sign-flip with scaling** for malicious clients:
- Honest client sends `updated_params`
- Malicious client sends `global + (-alpha * delta)` where `delta = local - global`

Expected patterns:
- Malicious: `upd_cos ≈ -1` (sent update opposite direction to honest update)
- Malicious: `post_norm ≈ alpha * pre_norm` (when alpha > 1)
- Honest: `upd_cos ≈ +1` and `pre_norm == post_norm`

Observed in the log (examples in Round 1):
- Malicious clients show:
  - `malicious=True`, `alpha=1.500`, `upd_cos=-1.000000`
  - `post_norm` is ~1.5× `pre_norm`
- Honest clients show:
  - `malicious=False`, `alpha=0.000`, `upd_cos=1.000000`
  - `pre_norm == post_norm`

Conclusion:
- The log matches the intended sign-flip attack behavior.

---

### C) Do the global updates track malicious direction?

The server logs `global_shift` direction cosines:
- `cos_to_mal` is consistently **positive and large**
- `cos_to_honest` becomes **negative** in later rounds

Interpretation:
- The global update is being pulled toward the malicious average update direction,
  which is a typical outcome when enough malicious clients participate with sign-flip scaling.

---

### D) Do evaluation metrics degrade under attack?

The summary history shows steadily increasing values:
- **Distributed loss** increases each round (≈ 0.95 → ≈ 10.14 by round 10)
- **`val_mse`** increases (≈ 0.86 → ≈ 10.01 by round 10)
- **`val_sam`** increases (≈ 1.82 → ≈ 2.59 by round 10)

Interpretation:
- This is consistent with a poisoning attack that destabilizes learning and harms reconstruction performance.

---

### E) How many malicious clients?

The aggregated fit metric `is_malicious` is ~0.53, which suggests **~4 out of 7** clients are malicious (exact average can shift due to weighting by `num_examples`).

---

## Summary

- **Global init**: server-provided initial parameters (correct).
- **Seeding**: implemented server + clients; clients use `SEED + CLIENT_ID`.
- **Data split**: deterministic with `train_test_split(..., random_state=seed)`.
- **FedAvg**: plain Flower FedAvg (no robust aggregation).
- **Clipping**: only optional client gradient clipping; no server-side update clipping.
- **Log patterns**: match sign-flip attack; malicious updates are direction-inverted and scaled; global updates align more with malicious direction; evaluation metrics degrade over rounds.

