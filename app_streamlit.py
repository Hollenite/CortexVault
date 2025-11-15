import streamlit as st
import tempfile
import os
import shutil
import csv
import time
import numpy as np
import re
from organizer_backend import load_models, analyze_image

# --------------------------------------
# Cache models (so they load only once)
# --------------------------------------
@st.cache_resource
def load_all_models():
    return load_models()

# --------------------------------------
# UI
# --------------------------------------
st.set_page_config(page_title="Auraverse — Smart Organizer", layout="wide", initial_sidebar_state="expanded")
st.title("Auraverse — Intelligent Multi-Modal Storage System")
st.caption("Upload images and let the system classify and organize them into Main / Sub / Specific ...")

# --------------------------------------
# Sidebar options
# --------------------------------------
st.sidebar.header("Settings")
save_root = st.sidebar.text_input("Save root folder", value=os.path.join(os.getcwd(), "organized"))
os.makedirs(save_root, exist_ok=True)

log_csv = st.sidebar.checkbox("Log CSV (save a mapping log)", value=True)
csv_path = os.path.join(save_root, "mapping_log.csv")
if log_csv and not os.path.exists(csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "filename", "detected_objects", "mapped_categories"])

# New option: whether to show manual gender resolution for ambiguous humans
show_gender_resolve = st.sidebar.checkbox("Enable manual gender-resolve for ambiguous humans", value=True)

# --------------------------------------
# Load Models
# --------------------------------------
yolo_model, clf_model, preprocess, labels = load_all_models()

# --------------------------------------
# helper: safe_name (used earlier when needed)
# --------------------------------------
def safe_name(name: str, max_len: int = 128) -> str:
    name = name.strip()
    name = re.sub(r'[\\/:"*?<>|]+', '', name)
    name = re.sub(r'\s+', '_', name)
    return name[:max_len] if len(name) > max_len else name

# --------------------------------------
# File Upload Section
# --------------------------------------
st.markdown("### 📤 Upload Images")
uploaded_files = st.file_uploader(
    "Select one or more image files", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True
)

if uploaded_files:
    cols = st.columns(2)
    columns = cols
    image_display_width = 360

    processed_data = []

    for i, uploaded_file in enumerate(uploaded_files):
        # save uploaded to a temporary path to pass to YOLO and to copy later
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # analyze
        detected_labels, mapped_categories, annotated_img = analyze_image(tmp_path, yolo_model, clf_model, preprocess, labels)

        # show preview
        display_img = annotated_img
        col = columns[i % 2]
        with col.container():
            st.image(display_img, caption=f"🖼️ {uploaded_file.name}", width=image_display_width)

            st.markdown(f"#### 🧠 Detected Objects")
            if detected_labels:
                st.write(", ".join(detected_labels))
            else:
                st.caption("None detected")

            st.markdown(f"#### 🗂️ Categories")
            if mapped_categories:
                cat_display = [f"`{main}/{sub}/{sp}`" for main, sub, sp in mapped_categories]
                st.markdown("  •  ".join(cat_display))
            else:
                st.caption("Uncategorized")

            # Determine if human mapping is ambiguous for this file
            # Ambiguous if any mapping has main == "Human" and specific in ("Person", "Unknown")
            ambiguous_humans = []
            for idx, (m, s, sp) in enumerate(mapped_categories):
                if m == "Human" and (sp.lower() in ("person", "unknown")):
                    ambiguous_humans.append((idx, m, s, sp))

            # If enabled and ambiguous humans present, present a selector to resolve
            user_gender_choice = None
            if show_gender_resolve and ambiguous_humans:
                # show a per-file selector; using keys unique per file
                choice = st.selectbox(
                    label="Ambiguous human detected — resolve gender for saving",
                    options=["Auto", "Male", "Female", "Skip"],
                    index=0,
                    key=f"gender_resolve_{i}"
                )
                user_gender_choice = choice  # "Auto", "Male", "Female", "Skip"
                st.caption("If you choose Male/Female the file will be saved under Human/<Sub>/Male (or Female). 'Skip' will not save to ambiguous human categories.")
            else:
                # no UI; default Auto
                user_gender_choice = "Auto"

            # Save single file (button)
            save_btn_key = f"save_{i}"
            if st.button(f"💾 Save {uploaded_file.name}", key=save_btn_key, use_container_width=True):
                saved_any = False
                saved_paths = set()  # dedupe same destination path

                # Build an adjusted mapping list honoring manual gender choice
                adjusted_mappings = []
                for (m, s, sp) in mapped_categories:
                    # handle ambiguous humans
                    if m == "Human" and (sp.lower() in ("person", "unknown")):
                        if user_gender_choice == "Male":
                            adjusted_mappings.append((m, s, "Male"))
                        elif user_gender_choice == "Female":
                            adjusted_mappings.append((m, s, "Female"))
                        elif user_gender_choice == "Skip":
                            # skip adding this human mapping
                            continue
                        else:  # Auto
                            adjusted_mappings.append((m, s, sp))
                    else:
                        adjusted_mappings.append((m, s, sp))

                # Save to each (distinct) folder path
                for main, sub, sp in adjusted_mappings:
                    specific_safe = sp if sp and sp != "Unknown" else None

                    # Build destination folder (NO per-file subfolder)
                    if specific_safe:
                        folder_path = os.path.join(save_root, main, sub, specific_safe)
                    else:
                        folder_path = os.path.join(save_root, main, sub)

                    os.makedirs(folder_path, exist_ok=True)
                    dst = os.path.join(folder_path, uploaded_file.name)

                    # dedupe same destination
                    if dst not in saved_paths:
                        shutil.copy(tmp_path, dst)
                        saved_paths.add(dst)
                        saved_any = True

                if saved_any:
                    st.success(f"✅ Saved {uploaded_file.name}")
                else:
                    st.warning(f"⚠️ No category saved for {uploaded_file.name} (you may have chosen Skip or nothing was detected).")

                # Log CSV once per file with adjusted mappings (deduped)
                if log_csv:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    detected_str = ";".join(detected_labels) if detected_labels else ""
                    # dedupe textual mapped categories as strings
                    mapped_strs = []
                    for m, s, sp in adjusted_mappings:
                        mapped_strs.append(f"{m}/{s}/{sp}")
                    mapped_strs = list(dict.fromkeys(mapped_strs))  # preserve order + dedupe
                    mapped_str = ";".join(mapped_strs) if mapped_strs else "Uncategorized"
                    with open(csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, uploaded_file.name, detected_str, mapped_str])

        processed_data.append({
            "path": tmp_path,
            "filename": uploaded_file.name,
            "detected_labels": detected_labels,
            "mapped_categories": mapped_categories
        })

    # Bulk save (Save All) area
    st.markdown("---")
    if st.button("💾 Save ALL processed files", use_container_width=True):
        total_saved = 0
        total_files_saved = 0

        for j, data in enumerate(processed_data):
            tmp_path = data["path"]
            fname = data["filename"]
            detected_labels = data["detected_labels"]
            original_mappings = data["mapped_categories"]

            # For bulk save, we need to check if there are ambiguous humans and — if the UI was enabled — pick per-file choice
            # If show_gender_resolve is enabled and ambiguous exists, read the selectbox state (same key used above)
            if show_gender_resolve:
                # If at least one ambiguous human was present earlier, the selectbox exists; else key may not exist
                key = f"gender_resolve_{j}"
                try:
                    user_choice = st.session_state.get(key, "Auto")
                except Exception:
                    user_choice = "Auto"
            else:
                user_choice = "Auto"

            # Build adjusted mappings honoring user_choice for ambiguous humans
            adjusted_mappings = []
            for (m, s, sp) in original_mappings:
                if m == "Human" and (sp.lower() in ("person", "unknown")):
                    if user_choice == "Male":
                        adjusted_mappings.append((m, s, "Male"))
                    elif user_choice == "Female":
                        adjusted_mappings.append((m, s, "Female"))
                    elif user_choice == "Skip":
                        continue
                    else:
                        adjusted_mappings.append((m, s, sp))
                else:
                    adjusted_mappings.append((m, s, sp))

            saved_paths = set()
            saved_this_file = False

            for main, sub, sp in adjusted_mappings:
                specific_safe = sp if sp and sp != "Unknown" else None

                if specific_safe:
                    folder_path = os.path.join(save_root, main, sub, specific_safe)
                else:
                    folder_path = os.path.join(save_root, main, sub)

                os.makedirs(folder_path, exist_ok=True)
                dst = os.path.join(folder_path, fname)

                if dst not in saved_paths:
                    shutil.copy(tmp_path, dst)
                    saved_paths.add(dst)
                    total_saved += 1
                    saved_this_file = True

            if saved_this_file:
                total_files_saved += 1

            # Log CSV once per file (deduped mappings)
            if log_csv:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                detected_str = ";".join(detected_labels) if detected_labels else ""
                mapped_strs = [f"{m}/{s}/{sp}" for m, s, sp in adjusted_mappings]
                mapped_strs = list(dict.fromkeys(mapped_strs))
                mapped_str = ";".join(mapped_strs) if mapped_strs else "Uncategorized"
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, fname, detected_str, mapped_str])

        if total_saved > 0:
            st.success(f"✅ Saved {total_saved} file-copies across categories (affecting {total_files_saved} files).")
            st.balloons()
        else:
            st.warning("⚠️ No files were saved (no categories detected or user skipped ambiguous humans).")

# --------------------------------------
# Footer
# --------------------------------------
st.divider()
st.caption("💡 Built with Streamlit, YOLOv8, and EfficientNetV2-M — © 2025 Team Zenith")
