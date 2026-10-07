"""
MediTrack Model Prediction & Inference Module
Generates stock-out probabilities, prototype risk categorizations, and structured alerts.
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    MODEL_SAVE_PATH,
    METADATA_SAVE_PATH,
    FEATURE_COLUMNS,
    MEDICINE_METADATA,
    get_risk_level
)


class MediTrackPredictor:
    """
    Inference engine for MediTrack early-warning stock-out risk predictions.
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
        Generates predictions and risk levels for an input DataFrame containing engineered features.
        
        Args:
            df: DataFrame containing at least FEATURE_COLUMNS and identifier columns.
            
        Returns:
            DataFrame with prediction columns attached.
        """
        missing_feats = [col for col in self.features if col not in df.columns]
        if missing_feats:
            raise ValueError(f"Input DataFrame is missing required feature columns: {missing_feats}")
            
        X = df[self.features]
        probs = self.model.predict_proba(X)[:, 1]
        preds = self.model.predict(X)
        
        out_df = df.copy()
        out_df["stockout_probability"] = np.round(probs, 4)
        out_df["predicted_class"] = preds.astype(int)
        out_df["risk_level"] = [get_risk_level(p) for p in probs]
        
        # Attach medicine human-readable names if medicine_id exists
        if "medicine_id" in out_df.columns:
            out_df["medicine_name"] = out_df["medicine_id"].map(
                lambda mid: MEDICINE_METADATA.get(mid, {}).get("name", mid)
            )
            out_df["medicine_category"] = out_df["medicine_id"].map(
                lambda mid: MEDICINE_METADATA.get(mid, {}).get("category", "General")
            )
            
        return out_df

    def predict_single_medicine(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates prediction for a single medicine feature vector.
        
        Args:
            feature_dict: Dictionary mapping feature names to numerical values.
            
        Returns:
            Dictionary containing prediction probability, risk level, and operational metrics.
        """
        row_df = pd.DataFrame([feature_dict])
        missing_feats = [col for col in self.features if col not in row_df.columns]
        if missing_feats:
            raise ValueError(f"Feature vector is missing required columns: {missing_feats}")
            
        X = row_df[self.features]
        prob = float(self.model.predict_proba(X)[0, 1])
        pred_class = int(self.model.predict(X)[0])
        risk_lvl = get_risk_level(prob)
        
        med_id = feature_dict.get("medicine_id", "UNKNOWN")
        med_info = MEDICINE_METADATA.get(med_id, {})
        
        return {
            "medicine_id": med_id,
            "medicine_name": med_info.get("name", med_id),
            "medicine_category": med_info.get("category", "Healthcare"),
            "current_stock": float(feature_dict.get("current_stock", 0.0)),
            "recent_consumption_7d": float(feature_dict.get("consumption_7d_mean", 0.0)),
            "estimated_days_of_stock": float(feature_dict.get("estimated_days_of_stock", 0.0)),
            "days_since_last_replenishment": float(feature_dict.get("days_since_last_replenishment", 0.0)),
            "stockout_probability": round(prob, 4),
            "predicted_stockout": pred_class,
            "risk_level": risk_lvl
        }


def get_latest_medicine_predictions(features_csv: Path = None) -> pd.DataFrame:
    """
    Extracts the latest snapshot date for all medicines and generates live predictions.
    """
    from src.utils.config import FEATURES_DATA_PATH
    path = features_csv or FEATURES_DATA_PATH
    
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    
    # Get the latest available observation date per medicine
    latest_rows = df.sort_values("date").groupby("medicine_id").last().reset_index()
    
    predictor = MediTrackPredictor()
    predictions_df = predictor.predict_dataframe(latest_rows)
    return predictions_df


if __name__ == "__main__":
    print("[PREDICT] Generating latest snapshot predictions for all monitored medicines...")
    preds = get_latest_medicine_predictions()
    display_cols = [
        "medicine_id", "medicine_name", "date", "current_stock",
        "consumption_7d_mean", "estimated_days_of_stock",
        "stockout_probability", "risk_level"
    ]
    print(preds[display_cols].to_string(index=False))
