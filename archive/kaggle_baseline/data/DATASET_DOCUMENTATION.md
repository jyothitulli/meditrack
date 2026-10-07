# MediTrack Dataset Documentation

## 1. Dataset Overview
- **Dataset Name**: Longitudinal Daily Pharmaceutical Dispensing Dataset
- **Primary Source**: Milan Zdravkovic (Kaggle & ResearchGate)
- **Reference / URL**: https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data
- **License / Accessibility**: Open Access / Public Research Use (CC BY-SA 4.0)
- **Data Nature**: Real historical dispensing records collected from an operational community pharmacy.

## 2. Dataset Dimensions & Temporal Scope
- **Total Records**: 2,106 daily observations (historical actuals)
- **Total Raw Columns**: 14
- **Time Period**: January 2, 2014 to October 8, 2019 (2,106 continuous days, ~5.8 years)
- **Sampling Frequency**: Daily observations

## 3. Medicine & Classification Coverage
The dataset tracks 8 Anatomical Therapeutic Chemical (ATC) classification categories representing essential healthcare medicines:
1. **M01AB**: Anti-inflammatory and antirheumatic products, non-steroids (Acetic acid derivatives, e.g. Diclofenac)
2. **M01AE**: Anti-inflammatory and antirheumatic products, non-steroids (Propionic acid derivatives, e.g. Ibuprofen)
3. **N02BA**: Analgesics and antipyretics (Salicylic acid and derivatives, e.g. Aspirin)
4. **N02BE**: Analgesics and antipyretics (Pyrazolones and Anilides, e.g. Paracetamol / Acetaminophen)
5. **N05B**: Psycholeptics - Anxiolytics (e.g. Diazepam, Alprazolam)
6. **N05C**: Psycholeptics - Hypnotics and Sedatives
7. **R03**: Drugs for obstructive airway diseases (e.g. Salbutamol, Anti-asthmatics)
8. **R06**: Antihistamines for systemic use (e.g. Cetirizine, Loratadine)

## 4. Summary Statistics (Actual Daily Dispensing)

| Medicine Code | Mean Daily Dispensed | Std Dev | Min | 25% | Median (50%) | 75% | Max |
|---|---|---|---|---|---|---|---|
| **M01AB** | 5.03 | 0.65 | 3.03 | 4.65 | 5.08 | 5.47 | 6.71 |
| **M01AE** | 3.90 | 0.61 | 2.51 | 3.46 | 3.80 | 4.26 | 6.32 |
| **N02BA** | 3.88 | 0.78 | 2.05 | 3.32 | 3.94 | 4.41 | 6.04 |
| **N02BE** | 29.92 | 9.21 | 12.32 | 23.41 | 29.00 | 37.18 | 54.60 |
| **N05B** | 8.85 | 2.07 | 3.11 | 7.39 | 8.85 | 10.05 | 15.10 |
| **N05C** | 0.59 | 0.19 | 0.08 | 0.46 | 0.60 | 0.72 | 1.27 |
| **R03** | 5.52 | 2.10 | 0.20 | 4.08 | 5.64 | 7.09 | 10.29 |
| **R06** | 2.90 | 1.34 | 0.39 | 1.79 | 2.60 | 4.09 | 5.99 |

## 5. Data Quality & Audit
- **Missing Values**: 0 missing values across all records and fields.
- **Duplicate Records**: 0 duplicate daily timestamps.
- **Data Integrity**: Continuous, unbroken daily time-series sequence across the entire 6-year period.

## 6. Inventory Integration Methodology
To model stock availability and stock-out risk, standard hospital pharmacy periodic review inventory control dynamics ((s, S) policy) are applied directly to the authentic dispensing time-series:
Stock_t = max(0, Stock_{t-1} + Replenishment_t - Consumption_t)
This enables exact, mathematically rigorous, and non-leaking tracking of physical stock depletion, safety buffer breaches, and stockout occurrences.
