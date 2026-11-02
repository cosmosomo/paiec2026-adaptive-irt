# Order-Free Subgroup-Kernel Empirical Bayes (`adaptive_irt_v3`) — PAIEC NeurIPS 2026

Public open-source release accompanying our Competition Track submission to **The Predictive AI Evaluation Competition at NeurIPS 2026 (`NeurIPS.cc/2026/Workshop/PAIEC`, Codabench `17828`)**.

## Repository Structure

- `adaptive_irt_v3/`: Primary competition submission (`model.py`, `labeling.py`, `corpus_priors.json`, packaged as `adaptive_irt_v3.zip`, `3,445,756` bytes).
  - Pure Python 3.11 standard library at runtime (zero third-party runtime imports).
  - Strictly permutation-invariant (`max|delta p| = 0.0` across arbitrary `labeled` shuffles) with content-tuple cache invalidation on in-place mutation.
  - Combines **Module M1** (online zero-leakage subgroup/protocol and canonical model-stem peer similarity kernel), **Module M2** (low-budget item-shift variance damping + asymptotic peer weight relaxation), **Module M3** (sample-size-shrunk benchmark priors, zero-reset isolated `LOSO` anchoring, deconfounded peer intercepts, and 5-fold `HistGBDT` metadata priors), and **Module M4** (zero-deepcopy Fisher-information representative core-set acquisition hook in `labeling.py`).
- `adaptive_irt_v1/`: Pure zero-prior cold-start reference estimator (`model.py`, `479` lines, `9,018` bytes).
- `check_submission_zip.py`: Official PAIEC submission archive validator.
- `LICENSE`: MIT License.

## Verification

```bash
python check_submission_zip.py adaptive_irt_v3.zip
```
