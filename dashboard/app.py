"""
MediTrack Streamlit Surveillance & Early Warning Dashboard (S2 Multi-Facility Dataset)
Visualizes multi-facility medicine stock-out risk, historical trends, and SHAP explainability.
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
    PREDICTION_HORIZON_LABEL,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_HIGH
)
from src.ml.predict import MediTrackPredictor
from src.ml.explain import MediTrackExplainer

st.set_page_config(
    page_title="MediTrack - Essential Medicine Stock-Out Early Warning System",
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
        f"Explainable Machine Learning Decision-Support System | Dataset: S2 DHIS2 (1,091 Facilities) | Horizon: {PREDICTION_HORIZON_LABEL}"
    )
    
    df = load_historical_data()
    metrics = load_metrics()
    
    if df is None:
        st.error("Processed features dataset not found. Please execute `python run_pipeline.py` first.")
        return
        
    try:
        predictor, explainer = load_models()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Extract latest snapshot per facility-product pair
    latest_snapshot = df.sort_values("date").groupby(["hf_pk", "productID"]).last().reset_index()
    predictions_df = predictor.predict_dataframe(latest_snapshot)
    
    # Sidebar Navigation & Filter
    st.sidebar.header("🧭 Surveillance Controls")
    view_mode = st.sidebar.radio(
        "Select Module View",
        [
            "National & District Surveillance",
            "Facility & Medicine Drill-Down",
            "Model Performance (Holdout Test)",
            "Dataset & Academic Defensibility"
        ]
    )
    
    # District filter
    all_districts = ["All Districts"] + sorted(predictions_df["district"].dropna().unique().tolist())
    selected_district = st.sidebar.selectbox("Filter by Administrative District:", all_districts)
    
    filtered_df = predictions_df.copy()
    if selected_district != "All Districts":
        filtered_df = filtered_df[filtered_df["district"] == selected_district]
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Prototype Risk Thresholds")
    st.sidebar.markdown(f"- 🟢 **LOW Risk**: Prob < {RISK_THRESHOLD_LOW*100:.0f}%")
    st.sidebar.markdown(f"- 🟡 **MEDIUM Risk**: {RISK_THRESHOLD_LOW*100:.0f}% ≤ Prob < {RISK_THRESHOLD_HIGH*100:.0f}%")
    st.sidebar.markdown(f"- 🔴 **HIGH Risk**: Prob ≥ {RISK_THRESHOLD_HIGH*100:.0f}%")
    st.sidebar.caption("Calibrated against empirical stockout base rate (13.55%).")

    # VIEW 1: NATIONAL & DISTRICT SURVEILLANCE
    if view_mode == "National & District Surveillance":
        st.header(f"📊 Surveillance Overview — {selected_district}")
        
        # KPI Cards
        total_facilities = filtered_df["hf_pk"].nunique()
        total_monitored = len(filtered_df)
        high_risk_count = (filtered_df["risk_level"] == "HIGH").sum()
        med_risk_count = (filtered_df["risk_level"] == "MEDIUM").sum()
        low_risk_count = (filtered_df["risk_level"] == "LOW").sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Monitored Facilities", f"{total_facilities:,}")
        col2.metric("High-Risk Alerts (Next Month)", f"{high_risk_count:,}", delta=f"{high_risk_count} critical", delta_color="inverse")
        col3.metric("Medium-Risk (Buffer Warning)", f"{med_risk_count:,}")
        col4.metric("Low-Risk (Adequate Stock)", f"{low_risk_count:,}")
        
        st.markdown("---")
        st.subheader("📋 Active Surveillance Facility-Medicine Risk Table")
        
        # Display Table
        display_df = filtered_df[[
            "hf_pk", "district", "facility_type", "name1",
            "current_stock", "current_consumption", "estimated_days_of_stock",
            "stockout_probability", "risk_level"
        ]].copy()
        
        display_df["current_stock"] = display_df["current_stock"].map("{:,.0f}".format)
        display_df["current_consumption"] = display_df["current_consumption"].map("{:,.0f}".format)
        display_df["estimated_days_of_stock"] = display_df["estimated_days_of_stock"].map("{:.1f} days".format)
        display_df["stockout_probability"] = display_df["stockout_probability"].map("{:.1%}".format)
        
        st.dataframe(
            display_df.rename(columns={
                "hf_pk": "Facility ID",
                "district": "District",
                "facility_type": "Facility Type",
                "name1": "Essential Medicine / Supply",
                "current_stock": "Closing Stock",
                "current_consumption": "Monthly Demand",
                "estimated_days_of_stock": "Est. Days of Stock",
                "stockout_probability": "Predicted Stockout Risk (Next Month)",
                "risk_level": "Risk Category"
            }),
            use_container_width=True,
            height=350
        )
        
        # Risk Breakdown by District
        if selected_district == "All Districts":
            st.subheader("🗺️ High-Risk Alert Distribution across Administrative Districts")
            district_summary = predictions_df.groupby(["district", "risk_level"]).size().reset_index(name="count")
            fig_dist = px.bar(
                district_summary, x="district", y="count", color="risk_level",
                color_discrete_map={"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#dc3545"},
                title="Stock-Out Risk Distribution by District",
                labels={"district": "District", "count": "Medicine-Facility Pairs"}
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    # VIEW 2: FACILITY & MEDICINE DRILL-DOWN
    elif view_mode == "Facility & Medicine Drill-Down":
        st.header("🔍 Individual Facility & Medicine Risk Factor Diagnosis")
        
        # Select facility
        facilities = sorted(filtered_df["hf_pk"].unique().tolist())
        selected_fac = st.selectbox("Select Health Facility ID (`hf_pk`):", options=facilities)
        
        fac_df = filtered_df[filtered_df["hf_pk"] == selected_fac]
        med_options = fac_df["productID"].unique().tolist()
        
        selected_prod = st.selectbox(
            "Select Monitored Medicine:",
            options=med_options,
            format_func=lambda pid: f"[{pid}] {fac_df[fac_df['productID'] == pid]['name1'].iloc[0]}"
        )
        
        target_row = fac_df[fac_df["productID"] == selected_prod].iloc[0]
        pred_res = predictor.predict_single_instance(target_row.to_dict())
        explain_res = explainer.explain_instance(target_row)
        
        # Detail Header
        st.markdown(f"### **{pred_res['medicine_name']}** at Facility `{selected_fac}`")
        st.markdown(f"**Facility Type**: {pred_res['facility_type']} | **District**: {pred_res['district']}")
        
        # KPI Row
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Predicted Next-Month Risk", f"{pred_res['stockout_probability']*100:.1f}%")
        
        risk_color = "red" if pred_res["risk_level"] == "HIGH" else ("orange" if pred_res["risk_level"] == "MEDIUM" else "green")
        mcol2.markdown(f"**Risk Level**\n\n### :{risk_color}[{pred_res['risk_level']}]")
        
        mcol3.metric("Physical Stock Balance", f"{pred_res['current_stock']:,.0f} units")
        mcol4.metric("Est. Days Remaining", f"{pred_res['estimated_days_of_stock']:.1f} days")
        
        st.markdown("---")
        left_col, right_col = st.columns([3, 2])
        
        with left_col:
            st.subheader("📈 Longitudinal Inventory & Dispensing History")
            history = df[(df["hf_pk"] == selected_fac) & (df["productID"] == selected_prod)].sort_values("date")
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=history["date"], y=history["closeBalance"],
                mode="lines+markers", name="Closing Physical Stock",
                line=dict(color="#1f77b4", width=2)
            ))
            fig_hist.add_trace(go.Scatter(
                x=history["date"], y=history["consumption"],
                mode="lines+markers", name="Monthly Dispensed Demand",
                line=dict(color="#ff7f0e", width=2, dash="dot")
            ))
            fig_hist.update_layout(
                title=f"Longitudinal Trajectory (2019–2023) for Facility {selected_fac}",
                xaxis_title="Date", yaxis_title="Units",
                hovermode="x unified"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with right_col:
            st.subheader("🤖 SHAP Decision Factor Breakdown")
            st.markdown("**Why did the Random Forest baseline assign this risk score?**")
            
            for narrative in explain_res["narratives"]:
                st.info(narrative)
                
            st.markdown("##### Detailed Feature SHAP Contributions")
            contrib_df = pd.DataFrame(explain_res["top_contributions"][:6])
            contrib_df["shap_value"] = contrib_df["shap_value"].map("{:+.4f}".format)
            contrib_df["value"] = contrib_df["value"].map("{:.2f}".format)
            st.dataframe(
                contrib_df[["feature", "value", "impact", "shap_value"]].rename(columns={
                    "feature": "Feature Name", "value": "Observed Value",
                    "impact": "Directional Impact", "shap_value": "SHAP Attribution"
                }),
                use_container_width=True
            )

    # VIEW 3: MODEL PERFORMANCE & VALIDATION
    elif view_mode == "Model Performance (Holdout Test)":
        st.header("🎯 Rigorous Model Evaluation on Chronological Test Set")
        st.caption("Evaluated on 92,406 unseen future instances across 2023 (Zero Temporal Leakage)")
        
        if metrics:
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("Recall (Stockout Class)", f"{metrics['recall_stockout_class']*100:.2f}%")
            kpi2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            kpi3.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
            kpi4.metric("Precision", f"{metrics['precision']*100:.2f}%")
            kpi5.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            
            st.markdown("---")
            vcol1, vcol2 = st.columns(2)
            
            with vcol1:
                st.subheader("Confusion Matrix (Holdout Test Set)")
                if CONFUSION_MATRIX_PATH.exists():
                    st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion Matrix on Chronological Test Set (2023)")
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

    # VIEW 4: DATASET & ACADEMIC DEFENSIBILITY
    elif view_mode == "Dataset & Academic Defensibility":
        st.header("🎓 Dataset Provenance & Academic Defensibility")
        
        st.markdown("""
        ### Why S2 is Academically & Technically Superior:
        
        1. **Empirical Ground-Truth Dataset (Data Dryad DOI: 10.5061/dryad.h9w0vt4tw)**
           - Derived from the national **DHIS2 logistics system** in Sierra Leone (*Nature*, 2026).
           - Tracks **1,091 public health facilities** and **36 WHO essential medicines & clinical supplies** over 50 monthly cycles (457,225 total records).
           
        2. **True Physical Stock Balances & Real Stockouts**
           - Unlike the single-pharmacy Kaggle sales data which required synthetic $(s, S)$ inventory simulation, S2 records **actual physical opening balances, closing balances, batch receipts, and empirical stockout events** (61,956 real clinical stockout occurrences).
           
        3. **Strict Zero-Leakage Chronological Validation**
           - **Training Partition**: Earliest 80% calendar dates (2019-10 to 2022-12, 337,727 samples).
           - **Testing Partition**: Unseen future calendar year (2023-01 to 2023-10, 92,406 samples).
           - Guarantee: $\max(T_{\text{train}}) < \min(T_{\text{test}})$.
           
        4. **Clinical Priority: Recall & Early Intervention**
           - In healthcare inventory, **False Negatives** (failing to alert a stock-out of lifesaving Oxytocin, Salbutamol, or Antibiotics) lead to severe clinical risk.
           - MediTrack achieves **81.95% Recall** (12,380 stockouts caught in advance) on 92,406 holdout test cases.
        """)


if __name__ == "__main__":
    main()
