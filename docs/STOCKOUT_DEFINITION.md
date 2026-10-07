# MediTrack: S2 Stockout Variable Definition & Semantics

## 1. Source Definition & Encoding
In the S2 DHIS2 dataset from Sierra Leone:
- **Field Name**: `stockout`
- **Data Type**: `int64` (Binary: `0` or `1`)
- **Reporting Cadence**: Monthly facility logistics report
- **Semantics**:
  - `stockout = 1`: The health facility experienced a stock-out of the specific medicine/product at any point during that calendar month (zero stock balance or unable to fulfill patient prescription).
  - `stockout = 0`: The health facility maintained adequate physical stock throughout the calendar month.

## 2. Target Formulation for Predictive Early Warning
At prediction cutoff time $T$ (the end of month $T$):
- **Predictive Task**: Predict whether health facility $i$ will experience a stock-out of essential medicine $j$ during the **next monthly reporting period ($T+1$)**.
- **Target Variable**: `is_stockout_next_month`
$$\\text{is\\_stockout\\_next\\_month}_{i, j, T} = \\text{stockout}_{i, j, T+1}$$

## 3. Strict Anti-Leakage Boundary
- **Allowed in Feature Vector ($t \le T$)**:
  - Closing balance of current month $T$ (`closeBalance_T`)
  - Consumption during current month $T$ (`consumption_T`)
  - Replenishment received in month $T$ (`received_T`)
  - Stockout status in current month $T$ (`stockout_T`) and past months ($T-1, T-2, \\dots$)
  - Rolling consumption statistics (3-month, 6-month averages and variances)
  - Estimated days of stock remaining at end of month $T$
- **Forbidden in Features ($t > T$)**:
  - `consumption_{T+1}`, `received_{T+1}`, `closeBalance_{T+1}`, `stockout_{T+1}`
  - Any future-derived statistics.
