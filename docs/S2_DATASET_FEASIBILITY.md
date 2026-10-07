# MediTrack: S2 Dataset Feasibility Assessment Report

## 1. Feasibility Verdict & Executive Summary

> **Does S2 genuinely support the MediTrack stock-out-risk prediction problem better than the current Kaggle dataset?**
> 
> **YES ? OVERWHELMINGLY.**
> The S2 dataset from Data Dryad (DOI: [10.5061/dryad.h9w0vt4tw](https://doi.org/10.5061/dryad.h9w0vt4tw), associated with the 2026 *Nature* study on essential medicine allocation in Sierra Leone) is an empirical, multi-facility public health logistics dataset containing 457,225 records across 1,091 health facilities and 36 essential medicines over 50 monthly reporting cycles (2019?2023). Unlike the single-store Kaggle sales dataset that required simulated inventory balance equations, S2 provides genuine ground-truth stock-out events, actual physical inventory balances, received quantities, and dispensing volumes reported directly into DHIS2.

---

## 2. Requirement-by-Requirement Feasibility Matrix

| MediTrack Requirement | Required? | S2 Actual Field | Available? | Usable? | Evidence / Notes |
|---|:---:|---|:---:|:---:|---|
| **Medicine / Product Identity** | **Yes** | `productID`, `name1` | **YES** | **YES** | 36 distinct essential medicines & supplies (Oxytocin, Salbutamol, Paracetamol, Ampicillin, Ceftriaxone, ORS, Zinc, etc.). |
| **Facility Identity** | **Preferred** | `hf_pk`, `facility_type`, `district`, `lat`, `long` | **YES** | **YES** | 1,091 public health facilities (CHC, CHP, MCHP, Clinics) across 16 administrative districts. |
| **Temporal Information** | **Yes** | `date`, `quarter` | **YES** | **YES** | 50 continuous monthly observation cycles from `2019-10-01` to `2023-11-01` (16 quarters). |
| **Consumption / Demand** | **Yes** | `consumption`, `normAvg`, `normStd` | **YES** | **YES** | Monthly quantity dispensed per facility-medicine pair; Mean: 161.03 units. |
| **Inventory Balance** | **Strongly Preferred** | `openBalance`, `closeBalance` | **YES** | **YES** | Real physical opening and closing inventory counts reported by health facility staff. |
| **Replenishment / Received Qty** | **Strongly Preferred** | `received` | **YES** | **YES** | Real batch replenishment intake received during each monthly reporting period. |
| **Actual Stock-Out Information** | **Critical** | `stockout` | **YES** | **YES** | Binary empirical stockout indicator reported in DHIS2; 61,956 positive stockout records (13.55%). |
| **Historical Observations** | **Yes** | 50 monthly periods per facility-product pair | **YES** | **YES** | Longitudinal panel structure supporting lagged feature engineering and rolling trends. |
| **Sufficient Stock-Out Events** | **Critical** | 61,956 stock-out instances | **YES** | **YES** | Sufficient statistical power for supervised machine learning without synthetic oversampling. |
| **Sufficient Observations per Pair** | **Critical** | 27,092 unique facility-medicine time series | **YES** | **YES** | Median series length of 14 to 49 continuous monthly reports. |
| **Leakage-Free Prediction** | **Critical** | Temporal shift ($t+1$) formulation | **YES** | **YES** | Predict whether facility $i$ experiences a stockout of medicine $j$ in month $T+1$ using strictly historical features $t \le T$. |
| **Chronological Evaluation** | **Critical** | Partition by calendar dates | **YES** | **YES** | Earliest 38 months (2019-10 to 2022-11, 80%) for Training; latest 12 months (2022-12 to 2023-11, 20%) for Testing. |

---

## 3. Feasibility Conclusion & Next Step
The dataset satisfies 100% of the MediTrack operational requirements. Proceed with careful migration of the ML pipeline while preserving reusable evaluation and explainability components.
