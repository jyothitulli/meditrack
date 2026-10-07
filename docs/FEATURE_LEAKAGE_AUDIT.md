# MediTrack: Feature Leakage Audit & Timing Verification

## 1. Feature Timing Philosophy
Every feature in MediTrack must be strictly computable at prediction cutoff time $T$ (the end of the current monthly reporting cycle). Information regarding the subsequent month ($T+1$) is strictly quarantined for the ground truth target label $Y$.

## 2. Feature Mapping, Source Columns, and Leakage Risk Audit

| Feature Name | Source Columns | Historical Window | Leakage Risk | Mitigation & Timing Guarantee |
|---|---|---|:---:|---|
| `current_stock` | `closeBalance` | Month $T$ | **None** | Physical inventory reported at end of month $T$. |
| `open_balance` | `openBalance` | Month $T$ | **None** | Physical inventory reported at beginning of month $T$. |
| `current_consumption` | `consumption` | Month $T$ | **None** | Total quantity dispensed during month $T$. |
| `current_received` | `received` | Month $T$ | **None** | Total quantity replenished during month $T$. |
| `current_stockout` | `stockout` | Month $T$ | **None** | Empirical stockout status during month $T$. |
| `consumption_roll3_mean` | `consumption` | Months $[T-2, T]$ | **None** | 3-month rolling backward mean; requires $\ge 1$ historical record. |
| `consumption_roll6_mean` | `consumption` | Months $[T-5, T]$ | **None** | 6-month rolling backward mean; strictly backward looking. |
| `consumption_roll3_std` | `consumption` | Months $[T-2, T]$ | **None** | 3-month rolling standard deviation capturing recent demand volatility. |
| `stock_change_1m` | `closeBalance` | Months $T, T-1$ | **None** | Net monthly inventory delta: $\text{closeBalance}_T - \text{closeBalance}_{T-1}$. |
| `stock_to_consumption_ratio` | `closeBalance`, `consumption` | Month $T$ | **None** | Operational inventory coverage ratio: $\text{closeBalance}_T / (\text{consumption}_T + 1e-5)$. |
| `estimated_days_of_stock` | `closeBalance`, `consumption_roll3_mean` | Months $[T-2, T]$ | **None** | Deterministic operational arithmetic metric: $(\text{closeBalance}_T / (\text{consumption_roll3_mean} / 30 + 1e-5))$. |
| `historical_stockout_rate` | `stockout` | All historical $t \le T$ | **None** | Expanding window stockout frequency: $\text{mean}(\text{stockout}_{1:T})$. |
| `facility_type_encoded` | `facility_type` | Static / Metadata | **None** | Categorical facility classification (CHC, CHP, MCHP, Clinic). |
| `district_encoded` | `district` | Static / Metadata | **None** | Geographic district administrative identifier. |

## 3. Strict Verification
Automated unit tests in `tests/test_feature_engineering.py` enforce that:
1. Feature values computed at month $T$ remain mathematically identical if all future rows ($t > T$) are dropped.
2. The target column `is_stockout_next_month` is strictly masked from the feature input matrix $X$.
