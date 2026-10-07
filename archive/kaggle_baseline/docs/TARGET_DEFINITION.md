# MediTrack Target Definition

## 1. Exact Target Definition
The prediction target for MediTrack's first 30% milestone is a binary classification indicator:

$$\text{is\_stockout\_next\_14d} = \begin{cases} 1 & \text{if } \min_{k \in [1, 14]} (\text{Stock}_{T+k}) \le 0 \\ 0 & \text{otherwise} \end{cases}$$

Where:
- T is the Prediction Date (the current decision-making cutoff timestamp).
- T+k represents subsequent future days in the forward evaluation window (k in [1, 14]).
- Stock_{T+k} is the closing physical inventory stock of the medicine at future day T+k.

## 2. Prediction Horizon: 14 Days
### Why 14 Days Was Selected:
1. Clinical & Procurement Realism: Hospital pharmaceutical replenishment lead times typically range between 7 to 14 days for standard purchase order placement, vendor fulfillment, batch inspection, and stock intake. A 14-day early warning provides sufficient operational buffer for hospital inventory managers to intervene.
2. Dataset Compatibility: With 2,106 daily observations (~5.8 years), a 14-day forward window maintains a healthy balance between total sample size and future event occurrences.
3. Actionability: A 1-day horizon is too reactive; a 90-day horizon suffers from compounding uncertainty in volatile seasonal healthcare demand. 14 days represents an optimal tactical operational window.

## 3. Class Definitions
- Positive Class (Y = 1): The medicine experiences a stock-out event (inventory <= 0) at any point between day T+1 and day T+14.
- Negative Class (Y = 0): The medicine maintains adequate positive inventory balance throughout the entire 14-day window.

## 4. Label Generation Logic & Temporal Partitioning
For each medicine m and observation date T:
1. Check if at least 14 days of subsequent observations exist. If T > T_max - 14, this record cannot be labeled and is excluded from training/testing sets.
2. Compute the minimum closing inventory over the forward window:
   min_future_stock = min(Stock_{m, T+1}, Stock_{m, T+2}, ..., Stock_{m, T+14})
3. Assign label:
   Target = 1 if min_future_stock <= 0 else 0

## 5. Strict Separation of Target Information vs Input Features
- Information available at prediction date T (Allowed in features):
  - Current closing stock at T
  - Historical consumption up to and including T
  - Historical replenishment receipts received on or before T
  - Rolling consumption statistics (7d, 14d, 30d) calculated strictly on [T-30, T]
- Future information (Used ONLY to compute the ground truth label):
  - Stock levels on days T+1, ..., T+14
  - Consumption on days T+1, ..., T+14
  - Replenishment received on days T+1, ..., T+14
  - NEVER exposed to feature extraction routines.
