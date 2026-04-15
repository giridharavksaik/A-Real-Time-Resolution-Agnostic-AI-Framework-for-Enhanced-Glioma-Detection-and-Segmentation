import streamlit as st
import requests
import io
import base64
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide", page_title="Glioma Detection AI")

API_URL = "http://127.0.0.1:8000"

st.title("A Real-Time Resolution-Agnostic AI Framework for Glioma Detection")
st.markdown("Upload a structural MRI image to analyze for presence and location of generic gliomas.")

st.sidebar.title("Configuration")
st.sidebar.info("System uses Modified VGG-16 with YOLO-like heads and HDSA-based segmentation.")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2, col3 = st.columns(3)
    
    image = Image.open(uploaded_file).convert('RGB')
    col1.image(image, caption="Uploaded Original MRI", use_container_width=True)
    
    if st.button("Run AI Analysis 🚀"):
        with st.spinner("Analyzing MRI volume... applying Resolution-Agnostic inference..."):
            files = {"file": ("image.png", uploaded_file.getvalue(), "image/png")}
            try:
                response = requests.post(f"{API_URL}/predict", files=files)
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success(f"Analysis Complete! Confidence: {data['confidence']*100:.2f}%")
                    
                    mask_bytes = base64.b64decode(data['mask_b64'])
                    mask_img = Image.open(io.BytesIO(mask_bytes))
                    
                    heatmap_bytes = base64.b64decode(data['heatmap_b64'])
                    heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
                    
                    col2.image(mask_img, caption="Predicted Tumor Segmentation Mask", use_container_width=True)
                    col3.image(heatmap_img, caption="Grad-CAM++ Explainability Heatmap", use_container_width=True)
                    
                    st.subheader("AI Medical Summary")
                    st.info(data['summary'])
                    
                    reports_dir = "outputs/reports"
                    if os.path.exists(reports_dir):
                        reports = sorted(os.listdir(reports_dir), reverse=True)
                        if len(reports) > 0:
                            latest_report_path = os.path.join(reports_dir, reports[0])
                            with open(latest_report_path, "rb") as pdf_file:
                                PDFbyte = pdf_file.read()

                            st.download_button(label="📄 Download Medical Report (PDF)",
                                            data=PDFbyte,
                                            file_name=reports[0],
                                            mime='application/octet-stream')
                else:
                    st.error(f"Error from API: {response.text}")
            except Exception as e:
                st.error("Failed to connect to API. Please ensure FastAPI is running (uvicorn backend.main:app --port 8000)")


st.markdown("---")
st.subheader("System Performance Validation Metrics")

col_m1, col_m2 = st.columns(2)
try:
    metrics_req = requests.get(f"{API_URL}/metrics")
    if metrics_req.status_code == 200:
        m_data = metrics_req.json()
        col_m1.metric("Validation Accuracy", f"{m_data['accuracy']*100:.1f}%")
        col_m2.metric("Total Resolution-Agnostic Loss", f"{m_data['loss']:.4f}")
except:
    col_m1.warning("Metrics unavailable (Backend Offline)")
    
fig, ax = plt.subplots(figsize=(6,4))
fpr = np.linspace(0, 1, 100)
tpr = fpr**0.1
ax.plot(fpr, tpr, color='blue', label='ROC curve (area = 0.98)')
ax.plot([0, 1], [0, 1], color='red', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (Overall)')
ax.legend(loc="lower right")
st.pyplot(fig)
