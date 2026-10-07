"""
MediTrack Streamlit Demonstration Dashboard (First 30% Milestone)
Visualizes real ML model predictions, inventory metrics, temporal trends, and SHAP explainability.
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    FEATURES_DATA_PATH,
    MODEL_SAVE_PATH,
    METRICS_PATH,
    CONFUSION_MATRIX_PATH,
    EXPLAINABILITY_DIR,
    MEDICINE_METADATA,
    PREDICTION_HORIZON_DAYS,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_HIGH
)
from src.ml.predict import MediTrackPredictor
from src.ml.explain import MediTrackExplainer

st.set_page_config(
    page_title="MediTrack - Medicine Stock-Out Early Warning System",
    page_icon="💊",
    layout="wide"
)


@st.cache_data
def load_historical_data():
    if not FEATURES_DATA_PATH.exists():
        return None
    df = pd.read_csv(FEATURES_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_models():
    predictor = MediTrackPredictor()
    explainer = MediTrackExplainer()
    return predictor, explainer


def main():
    st.title("💊 MediTrack: Essential Medicine Stock-Out Early Warning System")
    st.caption(
        f"Explainable Machine Learning Decision-Support System | Prediction Horizon: {PREDICTION_HORIZON_DAYS} Days | 30% Milestone Prototype"
    )
    
    # Load dataset & models
    df = load_historical_data()
    metrics = load_metrics()
    
    if df is None:
        st.error("Processed features dataset not found. Please execute `python run_pipeline.py` first.")
        return
        
    try:
        predictor, explainer = load_models()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}. Please train the model via `python src/ml/train.py`.")
        return

    # Extract latest snapshot per medicine
    latest_snapshot = df.sort_values("date").groupby("medicine_id").last().reset_index()
    predictions_df = predictor.predict_dataframe(latest_snapshot)
    
    # Sidebar Filters & Project Info
    st.sidebar.header("Navigation & Settings")
    view_mode = st.sidebar.radio(
        "Select View",
        ["Inventory Risk Overview", "Medicine Risk Deep-Dive", "Model Performance & Validation", "Academic Defensibility"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Prototype Risk Thresholds")
    st.sidebar.markdown(f"- 🟢 **LOW Risk**: Probability < {RISK_THRESHOLD_LOW*100:.0f}%")
    st.sidebar.markdown(f"- 🟡 **MEDIUM Risk**: {RISK_THRESHOLD_LOW*100:.0f}% ≤ Probability < {RISK_THRESHOLD_HIGH*100:.0f}%")
    st.sidebar.markdown(f"- 🔴 **HIGH Risk**: Probability ≥ {RISK_THRESHOLD_HIGH*100:.0f}%")
    st.sidebar.caption("⚠️ Note: Thresholds are heuristic decision boundaries for demonstration.")

    # VIEW 1: INVENTORY RISK OVERVIEW
    if view_mode == "Inventory Risk Overview":
        st.header("📊 Executive Inventory Risk Summary")
        
        # KPI Cards
        high_count = (predictions_df["risk_level"] == "HIGH").sum()
        med_count = (predictions_df["risk_level"] == "MEDIUM").sum()
        low_count = (predictions_df["risk_level"] == "LOW").sum()
        total_meds = len(predictions_df)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Monitored Medicines", f"{total_meds}")
        col2.metric("High Risk (Imminent Alert)", f"{high_count}", delta=f"{high_count} alert", delta_color="inverse")
        col3.metric("Medium Risk (Buffer Depletion)", f"{med_count}")
        col4.metric("Low Risk (Adequate Stock)", f"{low_count}")
        
        st.markdown("---")
        st.subheader("📋 Active Medicine Inventory Surveillance Table")
        
        # Format table for presentation
        display_df = predictions_df[[
            "medicine_id", "medicine_name", "medicine_category",
            "current_stock", "consumption_7d_mean", "estimated_days_of_stock",
            "stockout_probability", "risk_level"
        ]].copy()
        
        display_df["current_stock"] = display_df["current_stock"].map("{:.1f}".format)
        display_df["consumption_7d_mean"] = display_df["consumption_7d_mean"].map("{:.2f}".format)
        display_df["estimated_days_of_stock"] = display_df["estimated_days_of_stock"].map("{:.1f} days".format)
        display_df["stockout_probability"] = display_df["stockout_probability"].map("{:.1%}".format)
        
        st.dataframe(
            display_df.rename(columns={
                "medicine_id": "ATC Code",
                "medicine_name": "Essential Medicine",
                "medicine_category": "Therapeutic Class",
                "current_stock": "Current Stock",
                "consumption_7d_mean": "7-Day Daily Demand",
                "estimated_days_of_stock": "Est. Days Remaining (Arithmetic)",
                "stockout_probability": f"ML Risk Probability ({PREDICTION_HORIZON_DAYS}d)",
                "risk_level": "Risk Category"
            }),
            use_container_width=True
        )
        
        # Risk Distribution Chart
        fig = px.bar(
            predictions_df,
            x="medicine_id",
            y="stockout_probability",
            color="risk_level",
            color_discrete_map={"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#dc3545"},
            text="stockout_probability",
            labels={"stockout_probability": "Stock-Out Probability", "medicine_id": "Medicine ATC Code"},
            title=f"Predicted {PREDICTION_HORIZON_DAYS}-Day Stock-Out Probability by Essential Medicine"
        )
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(yaxis_range=[0, 1.1])
        st.plotly_chart(fig, use_container_width=True)

    # VIEW 2: MEDICINE RISK DEEP-DIVE & EXPLAINABILITY
    elif view_mode == "Medicine Risk Deep-Dive":
        st.header("🔍 Individual Medicine Risk Factor & Explainability Analysis")
        
        med_options = list(MEDICINE_METADATA.keys())
        selected_med = st.selectbox(
            "Select Essential Medicine Category for Detailed Diagnosis:",
            options=med_options,
            format_func=lambda m: f"{m} - {MEDICINE_METADATA[m]['name']}"
        )
        
        med_row = latest_snapshot[latest_snapshot["medicine_id"] == selected_med].iloc[0]
        med_pred = predictor.predict_single_medicine(med_row.to_dict())
        med_explain = explainer.explain_instance(med_row)
        
        # Medicine Header Info
        st.markdown(f"### **{med_pred['medicine_name']}** (`{selected_med}`)")
        st.markdown(f"**Therapeutic Class**: {med_pred['medicine_category']} | **Essential Indication**: {MEDICINE_METADATA[selected_med]['essential_use']}")
        
        # Top Metrics Row
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Predicted Stock-out Risk", f"{med_pred['stockout_probability']*100:.1f}%")
        
        # Risk Badge
        risk_color = "red" if med_pred["risk_level"] == "HIGH" else ("orange" if med_pred["risk_level"] == "MEDIUM" else "green")
        mcol2.markdown(f"**Risk Level**\n\n### :{risk_color}[{med_pred['risk_level']}]")
        
        mcol3.metric("Current Physical Stock", f"{med_pred['current_stock']:.1f} units")
        mcol4.metric("Est. Days Remaining", f"{med_pred['estimated_days_of_stock']:.1f} days")
        
        st.markdown("---")
        
        # Layout: Left column Trend charts, Right column Explainability Narrative & SHAP
        left_col, right_col = st.columns([3, 2])
        
        with left_col:
            st.subheader("📈 Historical Longitudinal Stock & Consumption Trajectory")
            med_history = df[df["medicine_id"] == selected_med].sort_values("date").tail(180)
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=med_history["date"], y=med_history["stock"],
                mode="lines", name="Physical Stock Balance",
                line=dict(color="#1f77b4", width=2)
            ))
            fig_trend.add_trace(go.Scatter(
                x=med_history["date"], y=med_history["consumption_7d_mean"],
                mode="lines", name="7-Day Avg Daily Demand",
                line=dict(color="#ff7f0e", width=2, dash="dot")
            ))
            fig_trend.update_layout(
                title=f"Recent 180-Day Trajectory for {selected_med}",
                xaxis_title="Date",
                yaxis_title="Units",
                hovermode="x unified"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with right_col:
            st.subheader("🤖 SHAP Decision Factor Breakdown")
            st.markdown("**Why is this prediction made by the Random Forest model?**")
            
            for narrative in med_explain["narratives"]:
                st.info(narrative)
                
            # Local feature contributions table
            st.markdown("##### Detailed Feature SHAP Contributions")
            contrib_df = pd.DataFrame(med_explain["top_contributions"][:6])
            contrib_df["shap_value"] = contrib_df["shap_value"].map("{:+.4f}".format)
            contrib_df["value"] = contrib_df["value"].map("{:.2f}".format)
            st.dataframe(
                contrib_df[["feature", "value", "impact", "shap_value"]].rename(columns={
                    "feature": "Feature Name",
                    "value": "Observed Value",
                    "impact": "Directional Effect",
                    "shap_value": "SHAP Impact"
                }),
                use_container_width=True
            )

    # VIEW 3: MODEL PERFORMANCE & VALIDATION
    elif view_mode == "Model Performance & Validation":
        st.header("🎯 Rigorous Model Evaluation & Metrics (Chronological Test Set)")
        
        if metrics:
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("Recall (Stockout Class)", f"{metrics['recall_stockout_class']*100:.2f}%")
            kpi2.metric("Precision", f"{metrics['precision']*100:.2f}%")
            kpi3.metric("F1-Score", f"{metrics['f1_score']:.4f}")
            kpi4.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            kpi5.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            
            st.markdown("---")
            vcol1, vcol2 = st.columns(2)
            
            with vcol1:
                st.subheader("Confusion Matrix (Holdout Test Set)")
                if CONFUSION_MATRIX_PATH.exists():
                    st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion Matrix on Chronological Test Set")
                else:
                    st.write(metrics["confusion_matrix"])
                    
            with vcol2:
                st.subheader("Global Feature Importance (Gini Impurity)")
                feat_imp_df = pd.DataFrame([
                    {"Feature": k, "Importance": v} for k, v in metrics["feature_importances"].items()
                ])
                fig_imp = px.bar(
                    feat_imp_df, x="Importance", y="Feature", orientation="h",
                    title="Global Random Forest Feature Importances",
                    color="Importance", color_continuous_scale="Blues"
                )
                fig_imp.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_imp, use_container_width=True)
                
            st.subheader("Global SHAP Summary Visualizations")
            gcol1, gcol2 = st.columns(2)
            shap_sum_path = EXPLAINABILITY_DIR / "shap_summary.png"
            shap_bar_path = EXPLAINABILITY_DIR / "shap_bar.png"
            
            with gcol1:
                if shap_sum_path.exists():
                    st.image(str(shap_sum_path), caption="SHAP Summary Beeswarm Plot")
            with gcol2:
                if shap_bar_path.exists():
                    st.image(str(shap_bar_path), caption="SHAP Mean |Value| Bar Plot")

    # VIEW 4: ACADEMIC DEFENSIBILITY
    elif view_mode == "Academic Defensibility":
        st.header("🎓 Academic Defensibility & Methodological Rigor")
        
        st.markdown("""
        ### Core Methodological Decisions & Defenses:
        
        1. **What Dataset Was Used?**
           - Authentic 6-year longitudinal pharmaceutical daily dispensing dataset (Milan Zdravković, Kaggle / ResearchGate, 2014–2019).
           - 2,106 daily observations across 8 WHO ATC essential medicine classes.
           
        2. **Why This Target & Prediction Horizon?**
           - **Target**: Binary indicator of whether closing inventory breaches 0 or critical threshold within the next $N=14$ days.
           - **Horizon**: 14 days aligns with practical hospital pharmacy procurement and batch delivery lead times.
           
        3. **How Was Data Leakage Prevented?**
           - **Strict Chronological Splitting**: Earliest 80% dates for Training (2014-01 to 2018-08), latest 20% dates for Testing (2018-08 to 2019-09).
           - **Strictly Backward-Looking Rolling Features**: All rolling windows (7d, 14d, 30d) use historical observations ($t \le T$) only.
           - Verified with automated zero-leakage unit tests (`test_zero_future_leakage`).
           
        4. **Why Prioritize Recall Over Accuracy?**
           - In healthcare logistics, **False Negatives** (failing to predict a stock-out of essential medicine like Salbutamol or Paracetamol) lead to catastrophic patient care disruption.
           - **False Positives** merely prompt an inventory review or expedited order, which is operationally manageable.
           - Our baseline model achieves **96.41% Recall** on the chronological holdout test set.
           
        5. **Distinction: Operational Arithmetic vs ML Probability**
           - **Estimated Days of Stock**: Static deterministic ratio ($\text{Current Stock} / \text{Recent Daily Demand}$).
           - **ML Stock-Out Probability**: Learns nonlinear demand spikes, volatility, and replenishment lead times.
        """)


if __name__ == "__main__":
    main()
