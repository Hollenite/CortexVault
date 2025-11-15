import streamlit as st
import tempfile
import os
import shutil
import time
import json
import cv2
from pathlib import Path

from organizer_backend import analyze_path, save_item, ORGANIZED_ROOT, ingest_json_batch

# --------------------------------------------------------------
# Streamlit Page Setup
# --------------------------------------------------------------
st.set_page_config(page_title="CortexVault — Clean UI", page_icon="🗂️", layout="wide")

st.markdown("# CortexVault — Clean, Stable Streamlit UI")
st.write("Upload images, videos, documents, or paste ANY JSON (object / array / multi-collection).\n")
st.write("---")

# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    st.code(f"Backend Root: {ORGANIZED_ROOT}")
    show_thumbnails = st.checkbox("Show video thumbnails", True)
    show_doc_snippet = st.checkbox("Show document preview", True)

# --------------------------------------------------------------
# Upload Files
# --------------------------------------------------------------
st.subheader("Upload Files")
uploaded_files = st.file_uploader("Select files", accept_multiple_files=True)

# --------------------------------------------------------------
# JSON Input
# --------------------------------------------------------------
st.subheader("Or Paste Any JSON (Object, Array, Multi-Collection)")
json_text = st.text_area("Paste JSON here", height=200)
json_batch_parsed = None
json_batch_info = None

# Temp folder for uploads
TMP_BASE = Path(tempfile.gettempdir()) / "cortexvault_tmp"
TMP_BASE.mkdir(parents=True, exist_ok=True)

processed = []

# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def save_to_tmp(uploaded_file):
    fname = uploaded_file.name
    uid = str(int(time.time() * 1000))
    d = TMP_BASE / f"u_{uid}"
    d.mkdir(exist_ok=True)
    p = d / fname
    with open(p, "wb") as f:
        f.write(uploaded_file.read())
    return str(p)


def extract_frame(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        out = str(TMP_BASE / f"thumb_{int(time.time()*1000)}.jpg")
        cv2.imwrite(out, frame)
        return out
    except Exception:
        return None

# --------------------------------------------------------------
# Process Uploaded Files
# --------------------------------------------------------------
if uploaded_files:
    cols = st.columns(2)

    for i, up in enumerate(uploaded_files):
        tmp_path = save_to_tmp(up)
        info = analyze_path(tmp_path)
        col = cols[i % 2]

        with col:
            st.markdown(f"### {up.name}")
            kind = info.get("type")
            suggested = info.get("suggested_folder", "Documents/other")

            # Images
            if kind == "image":
                st.image(tmp_path, use_container_width=True)
                override = st.text_input("Folder", suggested, key=f"ov{i}")

            # Videos
            elif kind == "video":
                if show_thumbnails:
                    th = extract_frame(tmp_path)
                    if th:
                        st.image(th, caption="thumbnail", use_container_width=True)
                override = st.text_input("Folder", suggested, key=f"ov{i}")

            # Documents
            elif kind == "document":
                snippet = info.get("snippet", "")
                if show_doc_snippet:
                    st.text_area("Preview", snippet[:600], height=140, key=f"pv{i}")
                override = st.text_input("Folder", suggested, key=f"ov{i}")

            # JSON files
            elif isinstance(kind, str) and kind.startswith("json"):
                st.write("JSON detected → backend ingestion")
                st.json(info)
                override = None

            else:
                override = st.text_input("Folder", suggested, key=f"ov{i}")

            processed.append({"tmp_path": tmp_path, "filename": up.name, "override": override})

# --------------------------------------------------------------
# Process Pasted JSON
# --------------------------------------------------------------
if json_text:
    try:
        parsed = json.loads(json_text)
        json_batch_parsed = parsed

        # Send to backend for ingestion + analysis
        json_batch_info = analyze_path(None, batch_json=parsed)

        st.subheader("JSON Analysis — Simplified View")

        # ---- Decision Summary ----
        st.markdown("### 📌 Decision Summary")
        dec = json_batch_info.get("decision_summary", {}) if isinstance(json_batch_info, dict) else {}
        if dec:
            for col, d in dec.items():
                st.markdown(f"- **{col}** → `{str(d).upper()}`")
        else:
            st.markdown("_No decisions available_")

        # ---- Structure Overview (ASCII Tree) ----
        st.markdown("### 🗂️ Structure Overview")
        diag = json_batch_info.get("diagnostics", {}) if isinstance(json_batch_info, dict) else {}

        for col, d in diag.items():
            st.markdown(f"#### ◼️ {col}")
            n = d.get("n", 0)
            keys = d.get("n_keys", 0)
            nesting = d.get("nesting", False)

            tree = (
                f"{col}/\n"
                f"├── rows: {n}\n"
                f"├── keys: {keys}\n"
                f"└── nested: {nesting}"
            )

            st.code(tree)

        # Raw diagnostics expandable
        st.markdown("### 📘 Raw Diagnostics (Optional)")
        with st.expander("Show full JSON diagnostics"):
            st.json(json_batch_info)

    except Exception as e:
        st.error(f"Invalid JSON: {e}")

# --------------------------------------------------------------
# SAVE ALL BUTTON
# --------------------------------------------------------------
st.write("---")
if (processed or json_batch_parsed) and st.button("Save All"):
    results = []
    progress = st.progress(0)
    stat = st.empty()

    total = len(processed) + (1 if json_batch_parsed else 0)
    idx = 0

    # Save normal files
    for item in processed:
        idx += 1
        stat.write(f"Saving {item['filename']}...")
        override = item.get("override")

        try:
            dest = save_item(item["tmp_path"], custom_folder=override) if override else save_item(item["tmp_path"])
            results.append(dest)
        except Exception as e:
            st.error(f"Save failed: {e}")

        progress.progress(idx / total)

    # Save JSON batch
    if json_batch_parsed:
        idx += 1
        stat.write("Saving JSON batch...")
        res = ingest_json_batch(json_batch_parsed)
        results.append(res)
        progress.progress(idx / total)

        st.subheader("JSON Ingestion Result — Summary")

        artifacts = res.get("artifacts", {}) if isinstance(res, dict) else {}

        if artifacts.get("db_path"):
            st.markdown(f"- **SQLite DB created:** `{artifacts['db_path']}`")
        if artifacts.get("saved_files"):
            st.markdown(f"- **NoSQL files saved:** {len(artifacts['saved_files'])}")
        if artifacts.get("schema_files"):
            st.markdown(f"- **Schema files:** {len(artifacts['schema_files'])}")

    stat.write("Done!")
    st.success("All saved successfully.")
    st.subheader("Saved Outputs")
    st.write(results)

st.write("---")
st.caption("CortexVault — Clean, Error-Free Streamlit UI")