import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
from PIL import Image
from ultralytics import YOLO

# Page Settings

st.set_page_config(
    page_title="Smart Pothole Detection System",
    layout="wide"
)

st.title("Smart Pothole Detection System")
st.write("Detect potholes in road images and videos using YOLO segmentation.")

# -----------------------------
# Load YOLO Model
# -----------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()


# Helper Function: Severity


def get_severity(pothole_count):
    if pothole_count == 0:
        return "No Pothole"
    elif pothole_count <= 2:
        return "Low"
    elif pothole_count <= 5:
        return "Medium"
    else:
        return "High"

# Detection Function

def detect_potholes(image, confidence):
    results = model(image, conf=confidence)[0]

    pothole_count = 0
    confidences = []

    if results.boxes is not None:
        pothole_count = len(results.boxes)

        for box in results.boxes:
            conf = float(box.conf[0])
            confidences.append(conf)

    avg_confidence = 0
    if len(confidences) > 0:
        avg_confidence = sum(confidences) / len(confidences)

    severity = get_severity(pothole_count)

    annotated_image = results.plot()

    return annotated_image, pothole_count, avg_confidence, severity



# Sidebar

st.sidebar.title("Settings")

input_type = st.sidebar.radio(
    "Choose input type",
    ["Image", "Video"]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.3,
    step=0.05
)

# Image Input

if input_type == "Image":
    uploaded_image = st.sidebar.file_uploader(
        "Upload road image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image_np, use_column_width=True)

        with col2:
            st.subheader("Detection Result")

            result_image, count, avg_conf, severity = detect_potholes(
                image_np,
                confidence
            )

            st.image(result_image, use_column_width=True)

        st.markdown("---")
        st.subheader("Detection Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric("Potholes Detected", count)
        c2.metric("Average Confidence", f"{avg_conf * 100:.2f}%")
        c3.metric("Road Severity", severity)

        report = pd.DataFrame({
            "Potholes Detected": [count],
            "Average Confidence": [round(avg_conf * 100, 2)],
            "Severity": [severity]
        })

        csv = report.to_csv(index=False)

        st.download_button(
            label="Download Report as CSV",
            data=csv,
            file_name="pothole_detection_report.csv",
            mime="text/csv"
        )

    else:
        st.info("Upload an image from the sidebar to start detection.")


# Video Input

else:
    uploaded_video = st.sidebar.file_uploader(
        "Upload road video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_file.name)

        st.subheader("Video Detection Result")
        frame_placeholder = st.empty()

        frame_number = 0
        total_potholes = 0
        report_data = []

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # Process every 5th frame to keep app simple and faster
            if frame_number % 5 != 0:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result_frame, count, avg_conf, severity = detect_potholes(
                frame,
                confidence
            )

            total_potholes += count

            report_data.append({
                "Frame Number": frame_number,
                "Potholes Detected": count,
                "Average Confidence": round(avg_conf * 100, 2),
                "Severity": severity
            })

            frame_placeholder.image(
                result_frame,
                channels="RGB",
                use_column_width=True
            )

        cap.release()

        st.markdown("---")
        st.subheader("Video Detection Summary")

        if len(report_data) > 0:
            report_df = pd.DataFrame(report_data)

            max_potholes = report_df["Potholes Detected"].max()
            avg_confidence = report_df["Average Confidence"].mean()
            final_severity = get_severity(max_potholes)

            c1, c2, c3 = st.columns(3)

            c1.metric("Max Potholes in a Frame", int(max_potholes))
            c2.metric("Average Confidence", f"{avg_confidence:.2f}%")
            c3.metric("Overall Severity", final_severity)

            st.subheader("Frame-wise Report")
            st.dataframe(report_df)

            csv = report_df.to_csv(index=False)

            st.download_button(
                label="Download Video Report as CSV",
                data=csv,
                file_name="video_pothole_detection_report.csv",
                mime="text/csv"
            )

    else:
        st.info("Upload a video from the sidebar to start detection.")