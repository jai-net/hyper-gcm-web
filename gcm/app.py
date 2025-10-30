# app.py
import streamlit as st
from pipeline import run_pipeline


st.set_page_config(page_title="Hyperspectral Classifier (GCN + SVM)", layout="wide")
st.title("🌈 Hyperspectral Image Classification (GCN + SVM)")

st.write("Upload your hyperspectral `.mat` data file and corresponding ground truth file.")

data_file = st.file_uploader("Upload hyperspectral data (.mat)", type=["mat"])
gt_file = st.file_uploader("Upload ground truth labels (.mat)", type=["mat"])

if data_file and gt_file:
    if st.button("🚀 Run Classification"):
        with st.spinner("Processing... please wait, this may take 1–2 minutes."):
            try:
                acc, fig_buf = run_pipeline(data_file.read(), gt_file.read())
                st.success(f"✅ Test Accuracy: {acc:.4f}")
                st.image(fig_buf, caption="Classification Result", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
else:
    st.info("Please upload both `.mat` files to start.")
