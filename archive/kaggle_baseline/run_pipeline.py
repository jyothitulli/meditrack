"""
MediTrack Master Pipeline Orchestrator
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
from src.ml.predict import get_latest_medicine_predictions


def main():
    print("=" * 80)
    print("MEDITRACK - EXPLAINABLE MEDICINE STOCK-OUT RISK EARLY WARNING SYSTEM")
    print("Executing End-to-End Pipeline (First 30% Milestone Prototype)")
    print("=" * 80)
    
    start_time = time.time()
    
    # Step 1: Preprocessing & Inventory Ledger
    print("\n[STEP 1/5] Ingesting & Preprocessing Raw Data...")
    processed_df = run_preprocessing()
    
    # Step 2: Feature Engineering & Non-Leaking Target Creation
    print("\n[STEP 2/5] Engineering Temporal Features & Forward Target...")
    features_df = build_feature_dataset()
    
    # Step 3: Chronological Split, Training & Evaluation
    print("\n[STEP 3/5] Chronological Splitting, Model Training & Metric Evaluation...")
    model, metrics = run_training()
    
    # Step 4: SHAP Explainability Generation
    print("\n[STEP 4/5] Computing Global & Local SHAP Explainability...")
    run_explainability()
    
    # Step 5: Live Snapshot Predictions & Risk Prioritization
    print("\n[STEP 5/5] Generating Latest Monitored Medicine Predictions & Risk Categories...")
    latest_preds = get_latest_medicine_predictions()
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"PIPELINE EXECUTION COMPLETED SUCCESSFULLY in {elapsed:.2f} seconds!")
    print("=" * 80)
    print("\nLATEST MEDICINE INVENTORY RISK SNAPSHOT:")
    display_cols = [
        "medicine_id", "medicine_name", "date", "current_stock",
        "consumption_7d_mean", "estimated_days_of_stock",
        "stockout_probability", "risk_level"
    ]
    print(latest_preds[display_cols].to_string(index=False))
    
    print("\nSUMMARY METRICS ON CHRONOLOGICAL TEST SET:")
    print(f"  * Accuracy:                  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  * Recall (Stockout Class):   {metrics['recall_stockout_class']:.4f} ({metrics['recall_stockout_class']*100:.2f}%)")
    print(f"  * Precision:                 {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  * F1-Score:                  {metrics['f1_score']:.4f}")
    print(f"  * ROC-AUC:                   {metrics['roc_auc']:.4f}")
    print(f"  * PR-AUC:                    {metrics['pr_auc']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
