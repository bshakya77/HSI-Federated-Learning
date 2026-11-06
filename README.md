# Poisoning Attack Simulation in Federated Hyperspectral Crop Anomlay Detection

- This repository contains the client-server simulation of `Sign-Flip Attack` and `Gaussian Attack` for hyperspectral crop anomaly detection in the federated environment.
- It leverages `FedAVG` strategy as the baseline Federated setting. It compares the reconstruction performance of `Robust Federated Aggregation (RFA)` with different defensive mechanism (Clipping, Adaptive Clipping, Clipping + DP Noise) against FedAvg.
- **Evaluation Metrics:** `Global Loss`, `Validation-MSE`, `Validation-SAM`, `Global Shift-Norm`
- **AUROC Analysis:** Compares the crop-anomaly detection using `MSE-based AUROC` and `SAM-based AUROC`.
- Code Repository is maintained in three separate branches for each different strategies: `fedavg-sign-flip`, `fedavg-gaussian-noise`, `RFA`.
