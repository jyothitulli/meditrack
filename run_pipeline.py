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
from src.ml.explain import run_explainability
from src.ml.predict import get_latest_snapshot_predictions


def main():
    print("=" * 80)
    print("MEDITRACK - EXPLAINABLE MEDICINE STOCK-OUT RISK EARLY WARNING SYSTEM")
    print("Executing End-to-End Pipeline on S2 National DHIS2 Dataset (1,091 Facilities)")
    print("=" * 80)
    
    start_time = time.time()
    
    # Step 1: Preprocessing & Cleaning
    print("\n[STEP 1/5] Ingesting & Validating S2 DHIS2 Longitudinal Dataset...")
    processed_df = run_preprocessing()
    
    # Step 2: Feature Engineering & Non-Leaking Target Creation
    print("\n[STEP 2/5] Engineering Temporal Features & Forward Next-Month Targets...")
    features_df = build_feature_dataset()
    
    # Step 3: Chronological Split, Training & Evaluation
    print("\n[STEP 3/5] Chronological Splitting, Model Training & Holdout Evaluation...")
    model, metrics = run_training()
    
    # Step 4: SHAP Explainability Generation
    print("\n[STEP 4/5] Computing Global & Local SHAP Decision Attributions...")
    run_explainability()
    
    # Step 5: Live Snapshot Predictions & Surveillance Summary
    print("\n[STEP 5/5] Generating Active Surveillance Inferences across 1,091 Facilities...")
    latest_preds = get_latest_snapshot_predictions()
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"PIPELINE EXECUTION COMPLETED SUCCESSFULLY in {elapsed:.2f} seconds!")
    print("=" * 80)
    print(f"\nACTIVE HEALTHCARE SURVEILLANCE OVERVIEW:")
    print(f"  * Total Active Facility-Medicine Pairs: {len(latest_preds):,}")
    print(f"  * High-Risk Imminent Stockout Alerts:   {(latest_preds['risk_level'] == 'HIGH').sum():,}")
    print(f"  * Medium-Risk Buffer Warnings:          {(latest_preds['risk_level'] == 'MEDIUM').sum():,}")
    print(f"  * Low-Risk Adequate Stock:              {(latest_preds['risk_level'] == 'LOW').sum():,}")
    
    print("\nCHRONOLOGICAL TEST SET PERFORMANCE METRICS (92,406 Instances, 2023):")
    print(f"  * Recall (Stockout Class):   {metrics['recall_stockout_class']:.4f} ({metrics['recall_stockout_class']*100:.2f}%)")
    print(f"  * Precision:                 {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  * F1-Score:                  {metrics['f1_score']:.4f}")
    print(f"  * ROC-AUC:                   {metrics['roc_auc']:.4f}")
    print(f"  * PR-AUC:                    {metrics['pr_auc']:.4f}")
    print(f"  * Accuracy:                  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
