"""
MediTrack: Hospital-Level Medicine Inventory & Stock-Out Early Warning Dashboard
Facility-centric decision support interface for hospital and pharmacy inventory managers.
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

# Page configuration
st.set_page_config(
    page_title="MediTrack - Hospital Medicine Stock-Out Early Warning",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for healthcare dashboard styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .high-risk-badge {
        color: #ef4444;
        font-weight: 700;
    }
    .med-risk-badge {
        color: #f59e0b;
        font-weight: 700;
    }
    .low-risk-badge {
        color: #3b82f6;
        font-weight: 700;
    }
    .facility-subheading {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


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


@st.cache_data
def compute_all_latest_predictions(_predictor, df):
    latest_snapshot = df.sort_values("date").groupby(["hf_pk", "productID"]).last().reset_index()
    predictions_df = _predictor.predict_dataframe(latest_snapshot)
    return predictions_df


def main():
    df = load_historical_data()
    metrics = load_metrics()
    
    if df is None:
        st.error("Processed features dataset not found. Please execute `python run_pipeline.py` first.")
        return
        
    try:
        predictor, explainer = load_models()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}. Run `python src/ml/train.py` first.")
        return

    # Compute latest predictions
    all_preds_df = compute_all_latest_predictions(predictor, df)
    
    # -------------------------------------------------------------
    # SIDEBAR NAVIGATION & FACILITY CONTROLS
    # -------------------------------------------------------------
    st.sidebar.markdown("## 🏥 MediTrack Navigation")
    nav_selection = st.sidebar.radio(
        "Select View",
        ["🏥 Hospital Inventory", "📊 Model Evaluation", "📚 Methodology & About"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    if nav_selection == "🏥 Hospital Inventory":
        st.sidebar.markdown("### 🏥 Select Facility")
        
        # Build facility dropdown options from actual dataset
        unique_facilities = all_preds_df[["hf_pk", "facility_type", "district"]].drop_duplicates().sort_values("hf_pk")
        facility_map = {
            row["hf_pk"]: f"Facility ID: {row['hf_pk']} ({row['facility_type']} · {row['district']})"
            for _, row in unique_facilities.iterrows()
        }
        
        selected_hf_pk = st.sidebar.selectbox(
            "Select Health Facility:",
            options=list(facility_map.keys()),
            format_func=lambda x: facility_map[x],
            index=0
        )
        
        # Filter all data strictly to the selected facility
        fac_df = all_preds_df[all_preds_df["hf_pk"] == selected_hf_pk].copy()
        fac_info = unique_facilities[unique_facilities["hf_pk"] == selected_hf_pk].iloc[0]
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔍 Inventory Filters")
        
        # Medicine filter
        all_med_names = ["All Medicines"] + sorted(fac_df["name1"].unique().tolist())
        selected_med_filter = st.sidebar.selectbox("💊 Medicine:", all_med_names)
        
        # Risk filter
        risk_filter = st.sidebar.selectbox("⚠️ Risk Level:", ["All Risks", "🔴 High Risk", "🟡 Medium Risk", "🔵 Low Risk"])
        
        # Search box
        search_query = st.sidebar.text_input("🔍 Search Medicine Name:", "")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📌 Risk Legend")
        st.sidebar.markdown("- 🔴 **HIGH (≥50%)** — Immediate review required")
        st.sidebar.markdown("- 🟡 **MEDIUM (20–50%)** — Monitor & plan order")
        st.sidebar.markdown("- 🔵 **LOW (<20%)** — Stock balance adequate")
        st.sidebar.caption("Thresholds calibrated to 13.55% empirical stockout baseline.")
        
        # Apply filters to facility medicines
        filtered_fac_df = fac_df.copy()
        
        if selected_med_filter != "All Medicines":
            filtered_fac_df = filtered_fac_df[filtered_fac_df["name1"] == selected_med_filter]
            
        if risk_filter == "🔴 High Risk":
            filtered_fac_df = filtered_fac_df[filtered_fac_df["risk_level"] == "HIGH"]
        elif risk_filter == "🟡 Medium Risk":
            filtered_fac_df = filtered_fac_df[filtered_fac_df["risk_level"] == "MEDIUM"]
        elif risk_filter == "🔵 Low Risk":
            filtered_fac_df = filtered_fac_df[filtered_fac_df["risk_level"] == "LOW"]
            
        if search_query.strip():
            filtered_fac_df = filtered_fac_df[
                filtered_fac_df["name1"].str.contains(search_query.strip(), case=False, na=False)
            ]
            
        # -------------------------------------------------------------
        # PRIMARY VIEW: HOSPITAL MEDICINE STOCK OVERVIEW
        # -------------------------------------------------------------
        st.markdown("# 🏥 Hospital Medicine Stock Overview")
        st.markdown(
            f"<div class='facility-subheading'>Selected Facility: <b>Facility ID: {fac_info['hf_pk']}</b> &nbsp;|&nbsp; "
            f"Type: <b>{fac_info['facility_type']}</b> &nbsp;|&nbsp; "
            f"District: <b>{fac_info['district']}</b> &nbsp;|&nbsp; "
            f"Horizon: <b>{PREDICTION_HORIZON_LABEL}</b></div>",
            unsafe_allow_html=True
        )
        
        # Top Facility Summary Metric Cards
        total_meds = len(fac_df)
        high_risk_count = (fac_df["risk_level"] == "HIGH").sum()
        med_risk_count = (fac_df["risk_level"] == "MEDIUM").sum()
        low_risk_count = (fac_df["risk_level"] == "LOW").sum()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💊 Medicines Monitored", f"{total_meds}")
        c2.metric("🔴 High Risk", f"{high_risk_count}", delta=f"{high_risk_count} critical" if high_risk_count > 0 else "0 critical", delta_color="inverse")
        c3.metric("🟡 Medium Risk", f"{med_risk_count}")
        c4.metric("🔵 Low Risk", f"{low_risk_count}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # -------------------------------------------------------------
        # MAIN COMPONENT: MEDICINE STOCK-OUT RISK MONITOR
        # -------------------------------------------------------------
        st.subheader("💊 Medicine Stock-Out Risk Monitor")
        
        # Sort highest-risk medicines first: HIGH -> MEDIUM -> LOW, then probability descending
        risk_rank_map = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        filtered_fac_df["risk_rank"] = filtered_fac_df["risk_level"].map(risk_rank_map)
        filtered_fac_df = filtered_fac_df.sort_values(
            ["risk_rank", "stockout_probability"],
            ascending=[True, False]
        ).reset_index(drop=True)
        
        # Build Table View
        risk_dot_map = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}
        
        table_df = pd.DataFrame()
        table_df["Risk"] = filtered_fac_df["risk_level"].map(risk_dot_map)
        table_df["Medicine"] = filtered_fac_df["name1"]
        table_df["Stock Available"] = filtered_fac_df["current_stock"].map("{:,.0f} units".format)
        table_df["Avg. Monthly Consumption"] = filtered_fac_df["current_consumption"].map("{:,.0f} units/mo".format)
        table_df["Est. Days of Stock"] = filtered_fac_df["estimated_days_of_stock"].map("{:.1f} days".format)
        
        # VISUAL REQUIREMENT: Colored dot + Risk Text + Probability (e.g. 🔴 HIGH · 87%)
        table_df["Stock-Out Risk"] = filtered_fac_df.apply(
            lambda r: f"{risk_dot_map[r['risk_level']]} {r['risk_level']} · {r['stockout_probability']*100:.1f}%",
            axis=1
        )
        
        if len(table_df) > 0:
            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True,
                height=min(400, 40 + len(table_df) * 36)
            )
        else:
            st.info("No medicines match the selected filter criteria for this facility.")
            
        st.markdown("---")
        
        # -------------------------------------------------------------
        # MEDICINE DETAIL VIEW (DIAGNOSIS & EXPLAINABILITY)
        # -------------------------------------------------------------
        st.subheader("🔍 Medicine Risk Diagnosis & Longitudinal Trajectory")
        
        # Select medicine to inspect from available facility medicines
        available_med_list = filtered_fac_df["productID"].tolist() if len(filtered_fac_df) > 0 else fac_df["productID"].tolist()
        
        selected_prod_id = st.selectbox(
            "Select a medicine to inspect risk diagnosis & historical trend:",
            options=available_med_list,
            format_func=lambda pid: f"[{pid}] {fac_df[fac_df['productID'] == pid]['name1'].iloc[0]} ({risk_dot_map[fac_df[fac_df['productID'] == pid]['risk_level'].iloc[0]]} {fac_df[fac_df['productID'] == pid]['risk_level'].iloc[0]})"
        )
        
        med_row = fac_df[fac_df["productID"] == selected_prod_id].iloc[0]
        pred_res = predictor.predict_single_instance(med_row.to_dict())
        explain_res = explainer.explain_instance(med_row)
        
        # Medicine Header Banner
        risk_tag_color = "red" if pred_res["risk_level"] == "HIGH" else ("orange" if pred_res["risk_level"] == "MEDIUM" else "blue")
        st.markdown(f"### {risk_dot_map[pred_res['risk_level']]} **{pred_res['medicine_name']}**")
        
        # Detail KPI Row
        dk1, dk2, dk3, dk4 = st.columns(4)
        dk1.metric("Stock Available", f"{pred_res['current_stock']:,.0f} units")
        dk2.metric("Avg. Monthly Demand", f"{pred_res['current_consumption']:,.0f} units/mo")
        dk3.metric("Est. Days of Stock", f"{pred_res['estimated_days_of_stock']:.1f} days")
        dk4.markdown(f"**Stock-Out Risk ({PREDICTION_HORIZON_LABEL})**\n\n### :{risk_tag_color}[{risk_dot_map[pred_res['risk_level']]} {pred_res['risk_level']} · {pred_res['stockout_probability']*100:.1f}%]")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        left_col, right_col = st.columns([3, 2])
        
        with left_col:
            st.markdown("##### 📈 Historical Longitudinal Trajectory (2019–2023)")
            history = df[(df["hf_pk"] == selected_hf_pk) & (df["productID"] == selected_prod_id)].sort_values("date")
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=history["date"], y=history["closeBalance"],
                mode="lines+markers", name="Closing Physical Stock",
                line=dict(color="#3b82f6", width=2.5),
                marker=dict(size=6)
            ))
            fig_hist.add_trace(go.Scatter(
                x=history["date"], y=history["consumption"],
                mode="lines+markers", name="Monthly Dispensed Demand",
                line=dict(color="#f97316", width=2, dash="dot"),
                marker=dict(size=5)
            ))
            fig_hist.update_layout(
                title=f"Stock Balance & Consumption History for Facility {selected_hf_pk}",
                xaxis_title="Reporting Date",
                yaxis_title="Units",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with right_col:
            st.markdown("##### 🤖 Why is this medicine at risk?")
            st.caption("Model-supported SHAP decision factor breakdown:")
            
            for narrative in explain_res["narratives"]:
                if "increased" in narrative:
                    st.warning(f"⚠️ {narrative}")
                else:
                    st.info(f"✅ {narrative}")
                    
            st.markdown("###### Top Contributing Feature Attributions")
            contrib_df = pd.DataFrame(explain_res["top_contributions"][:5])
            contrib_df["shap_value"] = contrib_df["shap_value"].map("{:+.4f}".format)
            contrib_df["value"] = contrib_df["value"].map("{:.1f}".format)
            st.dataframe(
                contrib_df[["feature", "value", "impact", "shap_value"]].rename(columns={
                    "feature": "Feature", "value": "Value",
                    "impact": "Direction", "shap_value": "SHAP Impact"
                }),
                use_container_width=True,
                hide_index=True
            )

    # -------------------------------------------------------------
    # SECONDARY VIEW: MODEL EVALUATION (FOR REVIEWERS & RESEARCHERS)
    # -------------------------------------------------------------
    elif nav_selection == "📊 Model Evaluation":
        st.markdown("# 📊 Model Performance & Validation")
        st.caption("Evaluated on 92,406 unseen chronological test instances in 2023 (Zero Temporal Leakage)")
        
        if metrics:
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Recall (Stockout Class)", f"{metrics['recall_stockout_class']*100:.2f}%")
            k2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            k3.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
            k4.metric("Precision", f"{metrics['precision']*100:.2f}%")
            k5.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            
            st.markdown("---")
            vcol1, vcol2 = st.columns(2)
            
            with vcol1:
                st.subheader("Confusion Matrix (Chronological Test Set)")
                if CONFUSION_MATRIX_PATH.exists():
                    st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion Matrix on Unseen 2023 Test Stream")
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

    # -------------------------------------------------------------
    # SECONDARY VIEW: METHODOLOGY & ABOUT
    # -------------------------------------------------------------
    elif nav_selection == "📚 Methodology & About":
        st.markdown("# 📚 About MediTrack & Methodology")
        
        st.markdown(r"""
        ### System Overview
        **MediTrack** is an explainable machine learning decision-support early warning system designed for hospital and healthcare inventory managers to anticipate essential medicine stock-outs before physical supplies are depleted.

        ---

        ### Dataset Provenance
        - **Source**: S2 Longitudinal Medicine Consumption and Inventory Dataset from Data Dryad ([DOI: 10.5061/dryad.h9w0vt4tw](https://doi.org/10.5061/dryad.h9w0vt4tw)).
        - **Study Reference**: Chung et al., *"Improving access to essential medicines via decision-aware machine learning"*, Nature, 2026.
        - **Scale**: 457,225 records across 1,091 public health facilities and 36 essential medicines over 50 monthly reporting cycles (2019–2023).

        ---

        ### Anti-Leakage & Temporal Validation Guarantee
        - **Chronological Cutoff**:
          - Training Partition: Earliest 80% dates (2019-10 to 2022-12, 337,727 samples).
          - Holdout Test Partition: Latest 20% dates (2023-01 to 2023-10, 92,406 samples).
          - Zero temporal overlap: $\max(T_{\text{train}}) < \min(T_{\text{test}})$.
        - **Feature Timing**: All rolling demand averages, stock velocity, and historical stockout rates are computed strictly using observations on or before cutoff month $T$.

        ---

        ### Clinical & Decision-Support Disclaimer
        MediTrack provides probabilistic risk early warnings to assist authorized inventory personnel in prioritizing replenishment reviews. It does not provide clinical diagnoses or guarantee automated stockout prevention.
        """)


if __name__ == "__main__":
    main()
