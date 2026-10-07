"""
MediTrack Prediction & Inference Module (S2 Dataset)
Generates stockout probabilities, prototype risk categorizations, and facility-level alerts.
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    MODEL_SAVE_PATH,
    METADATA_SAVE_PATH,
    FEATURE_COLUMNS,
    get_risk_level
)


class MediTrackPredictor:
    """
    Inference engine for MediTrack early-warning stock-out risk predictions on S2 data.
    """
    def __init__(self, model_path: Path = MODEL_SAVE_PATH, metadata_path: Path = METADATA_SAVE_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at: {model_path}. Run training first.")
            
        self.model = joblib.load(model_path)
        self.features = FEATURE_COLUMNS
        
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions and risk levels for an input DataFrame containing S2 engineered features.
        """
        missing_feats = [col for col in self.features if col not in df.columns]
        if missing_feats:
            raise ValueError(f"Input DataFrame missing required feature columns: {missing_feats}")
            
        X = df[self.features]
        probs = self.model.predict_proba(X)[:, 1]
        preds = self.model.predict(X)
        
        out_df = df.copy()
        out_df["stockout_probability"] = np.round(probs, 4)
        out_df["predicted_class"] = preds.astype(int)
        out_df["risk_level"] = [get_risk_level(p) for p in probs]
        return out_df

    def predict_single_instance(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates prediction for a single facility-medicine feature vector.
        """
        row_df = pd.DataFrame([feature_dict])
        missing_feats = [col for col in self.features if col not in row_df.columns]
        if missing_feats:
            raise ValueError(f"Feature vector missing required columns: {missing_feats}")
            
        X = row_df[self.features]
        prob = float(self.model.predict_proba(X)[0, 1])
        pred_class = int(self.model.predict(X)[0])
        risk_lvl = get_risk_level(prob)
        
        return {
            "hf_pk": feature_dict.get("hf_pk", "UNKNOWN"),
            "productID": feature_dict.get("productID", "UNKNOWN"),
            "medicine_name": feature_dict.get("name1", "Essential Medicine"),
            "district": feature_dict.get("district", "National"),
            "facility_type": feature_dict.get("facility_type", "Health Facility"),
            "current_stock": float(feature_dict.get("current_stock", 0.0)),
            "current_consumption": float(feature_dict.get("current_consumption", 0.0)),
            "estimated_days_of_stock": float(feature_dict.get("estimated_days_of_stock", 0.0)),
            "historical_stockout_rate": float(feature_dict.get("historical_stockout_rate", 0.0)),
            "stockout_probability": round(prob, 4),
            "predicted_stockout": pred_class,
            "risk_level": risk_lvl
        }


def get_latest_snapshot_predictions(features_csv: Optional[Path] = None) -> pd.DataFrame:
    """
    Extracts the latest snapshot month for all facility-product pairs and generates live predictions.
    """
    from src.utils.config import FEATURES_DATA_PATH
    path = features_csv or FEATURES_DATA_PATH
    
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    
    # Latest month per facility-product pair
    latest_rows = df.sort_values("date").groupby(["hf_pk", "productID"]).last().reset_index()
    
    predictor = MediTrackPredictor()
    predictions_df = predictor.predict_dataframe(latest_rows)
    return predictions_df


if __name__ == "__main__":
    print("[PREDICT] Generating latest snapshot predictions across facilities...")
    preds = get_latest_snapshot_predictions()
    print(f"Total active surveillance pairs: {len(preds):,}")
    print("\nRisk Level Breakdown:")
    print(preds["risk_level"].value_counts())
    display_cols = [
        "hf_pk", "district", "facility_type", "name1",
        "current_stock", "estimated_days_of_stock",
        "stockout_probability", "risk_level"
    ]
    print("\nSample Predictions:")
    print(preds[display_cols].head(10).to_string(index=False))
