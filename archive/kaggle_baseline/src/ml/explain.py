"""
MediTrack Explainability Module
Provides global feature importance and individual medicine risk factor attribution using SHAP.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    MODEL_SAVE_PATH,
    EXPLAINABILITY_DIR,
    FEATURE_COLUMNS,
    MEDICINE_METADATA,
    RANDOM_SEED
)


class MediTrackExplainer:
    """
    SHAP-based explainability engine for the MediTrack Random Forest model.
    """
    def __init__(self, model_path: Path = MODEL_SAVE_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        self.model = joblib.load(model_path)
        self.features = FEATURE_COLUMNS
        
        # Initialize SHAP TreeExplainer
        print("[EXPLAIN] Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)

    def generate_global_explanations(self, X_sample: pd.DataFrame, max_display: int = 10):
        """
        Computes SHAP values over a sample and generates global feature importance plots.
        """
        EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)
        X = X_sample[self.features]
        
        print(f"[EXPLAIN] Computing SHAP values across {len(X)} background samples...")
        shap_values = self.explainer.shap_values(X)
        
        # Handle SHAP output format across scikit-learn/SHAP versions
        # For binary classification: list of 2 arrays or 3D array (samples, features, classes)
        if isinstance(shap_values, list):
            sv_positive = shap_values[1]  # Class 1 (Stockout)
        elif len(shap_values.shape) == 3:
            sv_positive = shap_values[:, :, 1]
        else:
            sv_positive = shap_values
            
        # 1. Save SHAP Summary Beeswarm Plot
        plt.figure(figsize=(9, 6))
        shap.summary_plot(
            sv_positive,
            X,
            feature_names=self.features,
            max_display=max_display,
            show=False
        )
        plt.title("MediTrack - Global SHAP Feature Impact on Stock-Out Risk", fontsize=12)
        plt.tight_layout()
        summary_plot_path = EXPLAINABILITY_DIR / "shap_summary.png"
        plt.savefig(summary_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[EXPLAIN] Saved SHAP summary plot to: {summary_plot_path}")
        
        # 2. Save SHAP Global Bar Plot (Mean Absolute SHAP)
        plt.figure(figsize=(9, 6))
        shap.summary_plot(
            sv_positive,
            X,
            plot_type="bar",
            feature_names=self.features,
            max_display=max_display,
            show=False
        )
        plt.title("MediTrack - Global Mean |SHAP| Feature Importance", fontsize=12)
        plt.tight_layout()
        bar_plot_path = EXPLAINABILITY_DIR / "shap_bar.png"
        plt.savefig(bar_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[EXPLAIN] Saved SHAP bar plot to: {bar_plot_path}")
        
        return summary_plot_path, bar_plot_path

    def explain_instance(self, feature_row: pd.Series) -> Dict[str, Any]:
        """
        Explains a single medicine prediction instance using SHAP attributions.
        
        Args:
            feature_row: Series containing FEATURE_COLUMNS.
            
        Returns:
            Dictionary containing feature values, SHAP contributions, and natural language narrative.
        """
        row_df = pd.DataFrame([feature_row[self.features]])
        shap_values = self.explainer.shap_values(row_df)
        
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]
            
        pred_prob = float(self.model.predict_proba(row_df)[0, 1])
        
        # Compile feature contributions
        contributions = []
        for feat_name, feat_val, s_val in zip(self.features, row_df.iloc[0], sv):
            contributions.append({
                "feature": feat_name,
                "value": float(feat_val),
                "shap_value": float(s_val),
                "abs_shap": abs(float(s_val)),
                "impact": "Increases Risk" if s_val > 0 else "Decreases Risk"
            })
            
        # Sort by magnitude of contribution
        contributions = sorted(contributions, key=lambda x: x["abs_shap"], reverse=True)
        
        # Generate academic narrative explanations
        narratives = []
        for c in contributions[:4]:
            feat = c["feature"]
            val = c["value"]
            s = c["shap_value"]
            impact_str = "increased predicted stock-out risk" if s > 0 else "reduced predicted stock-out risk"
            
            if feat == "estimated_days_of_stock":
                narratives.append(f"Estimated days of stock remaining ({val:.1f} days) {impact_str} (SHAP: {s:+.3f}).")
            elif feat == "current_stock":
                narratives.append(f"Current stock level ({val:.1f} units) {impact_str} (SHAP: {s:+.3f}).")
            elif feat == "consumption_to_stock_ratio":
                narratives.append(f"Recent demand-to-stock ratio ({val:.2f}) {impact_str} (SHAP: {s:+.3f}).")
            elif feat == "days_since_last_replenishment":
                narratives.append(f"Elapsed time since last replenishment ({val:.0f} days) {impact_str} (SHAP: {s:+.3f}).")
            elif feat == "consumption_7d_mean":
                narratives.append(f"Recent 7-day average consumption ({val:.2f} units/day) {impact_str} (SHAP: {s:+.3f}).")
            elif feat == "stock_change_7d":
                narratives.append(f"7-day net stock trajectory ({val:+.1f} units) {impact_str} (SHAP: {s:+.3f}).")
            else:
                narratives.append(f"Feature '{feat}' with value {val:.2f} {impact_str} (SHAP: {s:+.3f}).")
                
        return {
            "predicted_probability": round(pred_prob, 4),
            "top_contributions": contributions,
            "narratives": narratives
        }


def run_explainability():
    """
    Executes global explainability generation on test data sample.
    """
    from src.utils.config import FEATURES_DATA_PATH
    print(f"[EXPLAIN] Loading test samples from {FEATURES_DATA_PATH}...")
    df = pd.read_csv(FEATURES_DATA_PATH)
    
    # Take a representative test sample for SHAP background
    sample_df = df.sample(n=min(500, len(df)), random_state=RANDOM_SEED)
    
    explainer = MediTrackExplainer()
    explainer.generate_global_explanations(sample_df)
    
    # Test single instance explanation on the latest record
    latest_rec = df.iloc[-1]
    single_exp = explainer.explain_instance(latest_rec)
    print("\n[EXPLAIN] Example Local Instance Explanation:")
    print(f"          Predicted Stock-out Probability: {single_exp['predicted_probability']}")
    for n in single_exp["narratives"]:
        print(f"          - {n}")


if __name__ == "__main__":
    run_explainability()
