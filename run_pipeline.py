"""
MediTrack Master Pipeline Orchestrator (S2 Dataset)
Executes the complete end-to-end ML pipeline from data ingestion to explainability.
"""
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.preprocess import run_preprocessing
from src.data.feature_engineering import build_feature_dataset
from src.ml.train import run_training
from src.ml.calibrate import run_probability_calibration
from src.ml.benchmark import run_tournament
from src.ml.explain import run_explainability
from src.ml.predict import get_latest_snapshot_predictions


def main():
    print("=" * 80)
    print("MEDITRACK - 50% MILESTONE COMPLETE PIPELINE ORCHESTRATOR")
    print("Executing End-to-End Ingestion, Benchmarking, Calibration & Prescriptive Engine")
    print("=" * 80)
    
    start_time = time.time()
    
    # Step 1: Preprocessing & Cleaning
    print("\n[STEP 1/7] Ingesting & Validating S2 DHIS2 Longitudinal Dataset...")
    processed_df = run_preprocessing()
    
    # Step 2: Feature Engineering v2 (Seasonality, Volatility, Acceleration)
    print("\n[STEP 2/7] Engineering Temporal Features & Forward Next-Month Targets...")
    features_df = build_feature_dataset()
    
    # Step 3: Chronological Split, Training & Evaluation (Baseline RF)
    print("\n[STEP 3/7] Chronological Splitting, Model Training & Holdout Evaluation...")
    model, metrics = run_training()
    
    # Step 4: Probability Calibration (Platt Scaling & Isotonic Regression)
    print("\n[STEP 4/7] Probability Calibration & Reliability Analysis (Platt / Isotonic)...")
    cal_results = run_probability_calibration(features_df=features_df)
    
    # Step 5: Multi-Model Tournament & Walk-Forward Temporal CV
    print("\n[STEP 5/7] Executing Multi-Model Tournament & Walk-Forward Temporal CV...")
    benchmark_results = run_tournament()
    
    # Step 6: SHAP Explainability Generation
    print("\n[STEP 6/7] Computing Global & Local SHAP Decision Attributions...")
    run_explainability()
    
    # Step 7: Live Snapshot Predictions & Prescriptive Replenishment
    print("\n[STEP 7/7] Generating Prescriptive Surveillance Inferences across 1,091 Facilities...")
    latest_preds = get_latest_snapshot_predictions()
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"50% MILESTONE PIPELINE EXECUTION COMPLETED in {elapsed:.2f} seconds!")
    print("=" * 80)
    print(f"\nACTIVE HEALTHCARE SURVEILLANCE OVERVIEW:")
    print(f"  * Total Active Facility-Medicine Pairs: {len(latest_preds):,}")
    print(f"  * High-Risk Imminent Stockout Alerts:   {(latest_preds['risk_level'] == 'HIGH').sum():,}")
    print(f"  * Medium-Risk Buffer Warnings:          {(latest_preds['risk_level'] == 'MEDIUM').sum():,}")
    print(f"  * Low-Risk Adequate Stock:              {(latest_preds['risk_level'] == 'LOW').sum():,}")
    
    print("\nMODEL TOURNAMENT TEST SET SUMMARY (92,406 Holdout Instances, 2023):")
    for m_name, m_stats in benchmark_results["tournament_holdout_test"].items():
        print(f"  * {m_name:<26}: Recall={m_stats['recall_stockout_class']*100:.2f}% | ROC-AUC={m_stats['roc_auc']:.4f} | PR-AUC={m_stats['pr_auc']:.4f} | Brier={m_stats['brier_score']:.4f}")
        
    print(f"\nPROBABILITY CALIBRATION PERFORMANCE:")
    print(f"  * Optimal Calibrator: {cal_results['best_method']}")
    print(f"  * Uncalibrated Brier Score: {cal_results['uncalibrated']['brier_score']:.4f} -> Calibrated: {cal_results['best_brier']:.4f}")
    print(f"  * Uncalibrated ECE:         {cal_results['uncalibrated']['ece']:.4f} -> Calibrated: {cal_results['best_ece']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
