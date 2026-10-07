# MediTrack Data Leakage Prevention Strategy

## 1. Why Random Splitting Is Strictly Inappropriate for Inventory Time-Series
In standard i.i.d. tabular machine learning problems, random train/test splitting is standard practice. However, for longitudinal inventory time-series, random splitting is a catastrophic methodological error:
1. Temporal Information Leakage: Randomly assigning records across the timeline allows the model to train on observations from date T2 and be tested on date T1 where T1 < T2. The model can inadvertently memorize future trend levels or replenishment cycles.
2. Autocorrelation Exploitation: Daily inventory and consumption exhibit strong autocorrelation. Random splitting allows training on day t-1 and day t+1 to predict day t, which inflates test metrics artificially and causes severe real-world deployment failure.

## 2. Chronological Train / Test Split Methodology
MediTrack enforces a strict chronological cutoff:
- Training Set: Earliest 80% of calendar dates (January 2014 to July 2018).
- Test Set: Latest 20% of calendar dates (August 2018 to October 2019).
- Zero Temporal Overlap: All training timestamps strictly precede all test timestamps: max(T_train) < min(T_test).

## 3. Strict Pre-Cutoff Feature Engineering
- All rolling averages (7-day, 14-day, 30-day) and lagging statistics are computed using backward-looking windows ([T-w, T]) only.
- Forward-looking rolling windows ([T, T+w]) are strictly forbidden in feature extraction.

## 4. Verification & Automated Leakage Checks
Automated unit tests in tests/test_feature_engineering.py verify that:
1. Feature values computed at date T remain identical regardless of whether future rows (t > T) are present or masked out.
2. No future target column is ever included in the feature input matrix X.
3. The train and test splits have zero overlapping calendar dates.
