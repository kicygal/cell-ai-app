import streamlit as st
from PIL import Image
import numpy as np

from utils.cell_cycle_classifier import predict_cell_stage
from utils.hemocytometer_analysis import analyze_hemocytometer

st.set_page_config(page_title="AI Cell Analysis App", layout="wide")

st.title("🧬 AI Cell Analysis App")
st.write("Analyze cell cycle stages and hemocytometer counting accuracy.")

tab1, tab2 = st.tabs(["Cell Cycle Classifier", "Hemocytometer Error Analysis"])

with tab1:
    st.header("Cell Cycle Stage Classifier")

    input_choice = st.radio(
        "Choose cell image source:",
        ["Upload cell image", "Take a picture"],
        key="cell_choice"
    )

    image = None

    if input_choice == "Upload cell image":
        uploaded_file = st.file_uploader(
            "Upload a cell image",
            type=["jpg", "jpeg", "png"],
            key="cell_cycle_upload"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

    else:
        camera_file = st.camera_input("Take a cell image", key="cell_camera")
        if camera_file is not None:
            image = Image.open(camera_file).convert("RGB")

    if image is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Selected Cell Image", use_container_width=True)

        with col2:
            if st.button("Classify Cell Cycle Stage", key="classify_button"):
                stage, confidence = predict_cell_stage(image)

                st.subheader("Prediction")
                st.write(f"**Stage:** {stage}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.progress(confidence / 100)

                if confidence < 70:
                    st.warning("Low confidence. Retake or recheck the image.")
                else:
                    st.success("Prediction completed.")


with tab2:
    st.header("Hemocytometer Error Analysis")

    hemo_choice = st.radio(
        "Choose hemocytometer image source:",
        ["Upload hemocytometer image", "Take a picture"],
        key="hemo_choice"
    )

    hemo_image = None

    if hemo_choice == "Upload hemocytometer image":
        hemo_file = st.file_uploader(
            "Upload a hemocytometer image",
            type=["jpg", "jpeg", "png"],
            key="hemo_upload"
        )
        if hemo_file is not None:
            hemo_image = Image.open(hemo_file).convert("RGB")

    else:
        hemo_camera = st.camera_input("Take a hemocytometer image", key="hemo_camera")
        if hemo_camera is not None:
            hemo_image = Image.open(hemo_camera).convert("RGB")

    manual_live = st.number_input("Manual live cell count", min_value=0, step=1)
    manual_dead = st.number_input("Manual dead cell count", min_value=0, step=1)

    if hemo_image is not None:
        hemo_array = np.array(hemo_image)

        st.image(hemo_image, caption="Selected Hemocytometer Image", use_container_width=True)

        if st.button("Analyze Hemocytometer Image", key="hemo_analyze_button"):
            results_df, summary, comparison, consistency, decision, region_visuals = analyze_hemocytometer(
                hemo_array,
                manual_live,
                manual_dead
            )

            st.subheader("Final Decision")

            if decision["final_flag"] == "green":
                st.success(decision["final_message"])
            elif decision["final_flag"] == "yellow":
                st.warning(decision["final_message"])
            else:
                st.error(decision["final_message"])

            st.subheader("AI Summary")
            st.json(summary)

            st.subheader("Manual vs AI Comparison")
            st.json(comparison)

            st.subheader("Region Consistency")
            st.json(consistency)

            st.subheader("Region Results")
            st.dataframe(results_df)

            st.subheader("Detected Regions")
            cols = st.columns(5)

            for col, (region_name, region_img) in zip(cols, region_visuals.items()):
                with col:
                    st.image(region_img, caption=region_name, use_container_width=True)