import streamlit as st
from pipeline import run_pipeline, predict_with_saved_model

st.set_page_config(page_title="Hyperspectral GCN+SVM Classifier", layout="wide")
st.title("🌈 Hyperspectral Image Classification using GCN + SVM")

st.write("""
Upload your hyperspectral dataset (`.mat` file).  
You can **train a new model** or **predict** using the saved trained model.
""")

mode = st.radio("Select Mode:", ["Train New Model", "Predict with Saved Model"])

if mode == "Train New Model":
    st.subheader("📘 Train Mode")
    data_file = st.file_uploader("Upload Hyperspectral Data (.mat)", type=["mat"])
    gt_file = st.file_uploader("Upload Ground Truth (.mat)", type=["mat"])

    if st.button("🚀 Train Model"):
        if data_file and gt_file:
            with st.spinner("Training model... please wait (1–2 min)..."):
                acc, fig_buf = run_pipeline(data_file.getvalue(), gt_file.getvalue())
                st.success(f"✅ Model trained successfully! Accuracy: {acc:.4f}")
                st.image(fig_buf, caption="Ground Truth vs Prediction", use_container_width=True)
        else:
            st.warning("Please upload both data and ground truth files.")
else:
    st.subheader("🔍 Prediction Mode")
    data_file = st.file_uploader("Upload New Hyperspectral Scene (.mat)", type=["mat"])

    if st.button("📊 Predict Using Saved Model"):
        if data_file:
            with st.spinner("Predicting scene..."):
                try:
                    fig_buf = predict_with_saved_model(data_file.getvalue())
                    st.image(fig_buf, caption="Predicted Scene", use_container_width=True)
                    st.success("✅ Prediction completed!")
                except FileNotFoundError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please upload a scene file first.")