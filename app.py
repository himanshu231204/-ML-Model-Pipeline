import streamlit as st
import pandas as pd
import os
from automl_pipeline import run_end_to_end, CONFIG

st.set_page_config(page_title="AutoML Pipeline", layout="wide")

st.title("🤖 AutoML Pipeline Demo")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Data Preview:", df.head())

    target = st.selectbox("🎯 Select Target Column", df.columns)

    if st.button("🚀 Run Pipeline"):
        with st.spinner("Training in progress... ⏳"):
            model = run_end_to_end(uploaded_file, target)
        st.success("✅ Pipeline completed!")

        # ==== Show EDA Report Download ====
        eda_path = os.path.join(CONFIG["save_dir"], CONFIG["html_report_name"])
        if os.path.exists(eda_path):
            with open(eda_path, "rb") as f:
                st.download_button(
                    label="📥 Download EDA Report",
                    data=f,
                    file_name="EDA_Report.html",
                    mime="text/html"
                )

        # ==== Show Feature Importance Plot ====
        fi_path = os.path.join(CONFIG["save_dir"], "feature_importance_topK.png")
        if os.path.exists(fi_path):
            st.subheader("🔑 Feature Importance")
            st.image(fi_path, use_column_width=True)

        # ==== Show SHAP Plots ====
        shap_bar = os.path.join(CONFIG["save_dir"], "shap_summary_bar.png")
        shap_bee = os.path.join(CONFIG["save_dir"], "shap_beeswarm.png")
        if os.path.exists(shap_bar):
            st.subheader("📊 SHAP Summary (Bar)")
            st.image(shap_bar, use_column_width=True)
        if os.path.exists(shap_bee):
            st.subheader("🐝 SHAP Beeswarm")
            st.image(shap_bee, use_column_width=True)

        # ==== Show Metrics from manifest.json ====
        mf_path = os.path.join(CONFIG["save_dir"], "manifest.json")
        if os.path.exists(mf_path):
            import json
            with open(mf_path, "r") as f:
                manifest = json.load(f)
            st.subheader("📈 Metrics")
            st.json(manifest)
