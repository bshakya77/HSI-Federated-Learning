# Metrics Analysis (Legacy Sign-Flip Reference)

## Scope
This file documents legacy Sign-Flip metric interpretation and is kept for historical comparison.
The active attack implementation in this project is Gaussian noise.

This note reviews whether the following metrics are correctly implemented and correctly interpreted:
- global loss (distributed)
- val MSE
- val SAM
- pre/post norm update
- global shift
- cosine similarity between honest and malicious clients

## Verdict Summary

- **global loss (distributed): Correct**
- **val MSE: Correct**
- **val SAM: Correct**
- **pre/post norm update: Correct (diagnostic), interpretation-limited**
- **global shift: Correct**
- **cosine similarity between honest and malicious client: Mathematically correct, but interpretation can be misleading depending on expectation**

## Detailed Findings

### 1) Global loss (distributed)
- Flower logs this as `History (loss, distributed)`.
- It is built from each client's `evaluate()` returned loss and aggregated across clients.
- This is a valid federated/distributed validation loss trajectory (not training loss).

### 2) val MSE
- Client computes `val_mse` during evaluation from reconstruction error.
- Server aggregates with weighted averaging by `num_examples`.
- Implementation is consistent and correct for federated metric aggregation.

### 3) val SAM
- Client computes `val_sam` during evaluation.
- Server aggregates with the same weighted averaging function.
- Implementation is consistent and correct.

### 4) Pre/Post update norm
- `pre_update_norm`: norm of honest local update (`updated_params - global_params`).
- `post_update_norm`: norm of actually sent update (after attack transform if malicious).
- Correctly implemented for attack diagnostics:
  - honest client: pre ~= post
  - sign-flip malicious client: post reflects scaled/flipped update
- Caveat: these do not directly quantify server-side influence.

### 5) Global shift
- Computed as `new_global - old_global` after aggregation.
- Logged norm is the actual magnitude of the server model movement per round.
- This is correctly implemented.

### 6) Cosine between honest and malicious clients
- Implemented as cosine between:
  - mean honest update vector, and
  - mean malicious update vector
- This is not pairwise client-to-client cosine; it is group-mean directional alignment.
- So it is mathematically correct for that definition, but can be misleading if interpreted as:
  - typical honest-vs-malicious client pair similarity, or
  - contribution/influence share.

## Important Clarification on `update_cosine`
- `update_cosine` is computed between honest local update and sent update.
- With current sign-flip logic:
  - honest -> cosine ~= +1
  - malicious sign-flip (`-alpha * delta`, alpha>0) -> cosine ~= -1
- Therefore, near-constant values are expected by construction; this is a design property, not a bug.
- It behaves as a binary "direction flipped or not" check, not a nuanced quality metric.

## Implementation Issue vs FedAvg Weakness

- **Constant `update_cosine`**: metric-definition/implementation behavior (expected from current formula).
- **Vulnerability to sign-flip attacks**: weakness of vanilla FedAvg robustness (strategy-level limitation).

## Recommended Interpretation
Use current metrics as:
- `val_mse`, `val_sam`, global loss: performance tracking
- pre/post norms: attack transform diagnostics
- global shift + cosines: directional diagnostics

Do not use current cosine metrics alone to claim exact contribution shares of honest vs malicious groups.


Metric	          Honest client	    Gaussian malicious	    Sign-Flip malicious (old)
pre_update_norm reflects training same (computed pre-attack)  same
post_update_norm  ≈ pre_norm    scales with std, unrelated to training  ≈ alpha × pre_norm
update_cosine   ≈ +1.0          near 0, random        ≈ -1.0
honest_vs_mal_cos —             near 0                near -1.0