# MediTrack: Chronological Train / Validation / Test Splitting Strategy

## 1. Why Random Splitting Fails on Longitudinal Inventory Data
In multi-facility healthcare logistics, observations across consecutive months exhibit strong temporal autocorrelation. A random train/test split would permit training on month $t+1$ and testing on month $t$, creating severe data leakage (future-looking leakage) and producing artificially inflated evaluation metrics that collapse upon real-world deployment.

## 2. Chronological Partitioning Scheme
MediTrack partitions the S2 dataset by calendar observation dates:
- **Total Dataset Timeline**: 50 monthly reporting periods (`2019-10-01` to `2023-11-01`).
- **Training Set (Earliest ~80% of dates)**:
  - Dates: `2019-10-01` to `2022-11-01` (38 months, 298,903 instances).
- **Test Set (Latest ~20% of dates)**:
  - Dates: `2022-12-01` to `2023-10-01` (11 labeled future months, 120,432 instances).
  - Note: Month `2023-11-01` has no subsequent month $T+1$ and is reserved for live forward inference.

## 3. Zero Temporal Overlap Guarantee
$$\\max(T_{\\text{train}}) = \\text{2022-11-01} < \\min(T_{\\text{test}}) = \\text{2022-12-01}$$
All training timestamps strictly precede all test timestamps. The model is evaluated on unseen future healthcare quarters.
