import streamlit as st
import tempfile
import os
import cv2
import shutil
import csv
import time
import numpy as np
from organizer_backend import load_models, analyze_image

# --------------------------------------
# Cache models (so they load only once)
# --------------------------------------
@st.cache_resource
def load_all_models():
    return load_models()


# --------------------------------------
# Streamlit Page Setup
# --------------------------------------
st.set_page_config(
    page_title="CortexVault - By Team Zenith",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------
# Title + Description
# --------------------------------------
st.markdown("""
## 🤖 CortexVault - By Team Zenith
Upload images and let **YOLOv8** detect objects while **EfficientNetV2-M** classifies them.
The app intelligently groups your images into hierarchical folders.
""")

st.divider()

# --------------------------------------
# Sidebar Controls
# --------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    image_display_width = st.slider("🖼️ Image width", 250, 800, 400)
    save_root = st.text_input("💾 Save folder", value="organized_storage/frontend_classified")
    os.makedirs(save_root, exist_ok=True)

    log_csv = st.checkbox("🧾 Enable CSV logging", value=True)
    csv_path = os.path.join(save_root, "results_log.csv")

    if log_csv and not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "filename", "detected_objects", "mapped_categories"])

# --------------------------------------
# Load Models
# --------------------------------------
yolo_model, clf_model, preprocess, labels = load_all_models()

# --------------------------------------
# File Upload Section
# --------------------------------------
st.markdown("### 📤 Upload Images")
uploaded_files = st.file_uploader(
    "Select one or more image files",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

processed_data = []

# --------------------------------------
# Process Each Uploaded Image
# --------------------------------------
if uploaded_files:
    st.info(f"📸 {len(uploaded_files)} image(s) uploaded.")

    # Two-column layout for better presentation
    columns = st.columns(2)

    for i, uploaded_file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        detected_labels, mapped_categories, annotated_img = analyze_image(
            tmp_path, yolo_model, clf_model, preprocess, labels
        )

        # Prepare image for display
        if annotated_img is not None:
            arr = annotated_img
            if arr.ndim == 3 and arr.shape[2] == 3:
                if arr.dtype == np.float32 or arr.max() <= 1.0:
                    arr = (arr * 255).astype("uint8")
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            display_img = arr
        else:
            display_img = np.zeros((200, 200, 3), dtype=np.uint8)

        # Choose which column to display in
        col = columns[i % 2]
        with col.container(border=True):
            st.image(display_img, caption=f"🖼️ {uploaded_file.name}", width=image_display_width)

            st.markdown(f"#### 🧠 Detected Objects")
            if detected_labels:
                st.write(", ".join(detected_labels))
            else:
                st.caption("None detected")

            st.markdown(f"#### 🗂️ Categories")
            if mapped_categories:
                cat_display = [f"`{main}/{sub}`" for main, sub in mapped_categories]
                st.markdown("  •  ".join(cat_display))
            else:
                st.caption("Uncategorized / Unknown")

            save_btn_key = f"save_{i}_{uploaded_file.name}"
            if st.button(f"💾 Save {uploaded_file.name}", key=save_btn_key, use_container_width=True):
                for main, sub in mapped_categories:
                    folder_path = os.path.join(save_root, main, sub)
                    os.makedirs(folder_path, exist_ok=True)
                    shutil.copy(tmp_path, os.path.join(folder_path, uploaded_file.name))
                st.success(f"✅ Saved {uploaded_file.name}")

                if log_csv:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    detected_str = ";".join(detected_labels)
                    mapped_str = ";".join([f"{m}/{s}" for m, s in mapped_categories])
                    with open(csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, uploaded_file.name, detected_str, mapped_str])

        processed_data.append({
            "path": tmp_path,
            "filename": uploaded_file.name,
            "detected_labels": detected_labels,
            "mapped_categories": mapped_categories,
        })

    # --------------------------------------
    # Save All Button
    # --------------------------------------
    st.divider()
    st.markdown("### 💾 Save All Processed Images")

    if processed_data and st.button("Save All to Organized Folders", type="primary", use_container_width=True):
        total_saved, total_folders = 0, 0
        for data in processed_data:
            tmp_path = data["path"]
            fname = data["filename"]
            for main, sub in data["mapped_categories"]:
                folder_path = os.path.join(save_root, main, sub)
                os.makedirs(folder_path, exist_ok=True)
                shutil.copy(tmp_path, os.path.join(folder_path, fname))
                total_folders += 1
            total_saved += 1

            if log_csv:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                detected_str = ";".join(data["detected_labels"])
                mapped_str = ";".join([f"{m}/{s}" for m, s in data["mapped_categories"]])
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, fname, detected_str, mapped_str])

        st.success(f"✅ Saved {total_saved} images into {total_folders} folders total.")
        st.balloons()

# --------------------------------------
# Footer
# --------------------------------------
st.divider()
st.caption("💡 Built with Streamlit, YOLOv8, and EfficientNetV2-M — © 2025 Team Zenith")
