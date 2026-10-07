# MediTrack: Dataset Comparison & Migration Justification

## Detailed Comparative Analysis: Kaggle Pharma Sales vs. S2 Dryad Dataset

| Criterion | Kaggle Pharma Sales (Baseline) | S2 Dryad Dataset (Migrated) | Advantage / Justification |
|---|---|---|---|
| **Primary Source** | Retail pharmacy sales data (Milan Zdravkovi?, Kaggle) | National DHIS2 Public Health Logistics in Sierra Leone (Nature, 2026) | **S2 is a peer-reviewed national healthcare logistics dataset.** |
| **Facility Coverage** | Single commercial pharmacy (1 facility) | 1,091 public health facilities (CHC, CHP, MCHP, Clinics) | **S2 captures heterogeneous multi-facility healthcare networks.** |
| **Medicine Coverage** | 8 aggregated ATC therapeutic classes | 36 distinct essential medicines & supplies (Oxytocin, Salbutamol, Paracetamol, Ampicillin, etc.) | **S2 represents actual WHO essential medicines.** |
| **Total Observations** | 2,106 daily records (16,848 panel records) | 457,225 monthly facility-medicine records | **S2 provides 27x larger scale and statistical robustness.** |
| **Inventory Balance** | Simulated via synthetic $(s, S)$ balance equations | Real physical counts (`openBalance`, `closeBalance`, `received`, `consumption`) | **S2 has genuine empirical stock counts without synthetic simulation.** |
| **Stock-Out Events** | Derived from synthetic zero stock simulation | Real ground-truth monthly stockouts reported in DHIS2 (61,956 events, 13.55%) | **S2 contains authentic clinical stock-out events.** |
| **Replenishment Data** | Deterministic interval simulation | Real physical batch replenishment deliveries (`received`) | **S2 captures real-world irregular delivery patterns.** |
| **Spatial & Facility Metadata** | None | Facility type, latitude, longitude, 16 administrative districts | **S2 enables geographic and facility-level operational analysis.** |
| **Temporal Granularity** | Daily (2014?2019) | Monthly reporting cycles (2019?2023) | **S2 directly matches standard WHO/DHIS2 monthly logistics cycles.** |
| **Academic Defensibility** | Low (synthetic stockouts on retail sales) | Very High (empirical real-world health system data) | **S2 provides maximum academic defensibility.** |
