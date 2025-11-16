import streamlit as st
import tempfile
import os
import shutil
import time
import json
import cv2
from pathlib import Path

# Backend (includes ML helpers + hybrid query)
from organizer_backend import (
    analyze_path,
    save_item,
    ORGANIZED_ROOT,
    ingest_json_batch,
    load_models,
    analyze_image,
    hybrid_query,
)

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
# Load ML models (optional) — cached so they load only once
# --------------------------------------------------------------
@st.cache_resource
def _load_models_safe():
    try:
        return load_models()
    except Exception:
        return None, None, None, None

yolo_model, clf_model, preprocess, labels = _load_models_safe()

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
                # show smaller raw thumbnail
                st.image(tmp_path, width=360)

                # Run image analysis (labels only) if models available; degrade gracefully
                detected = []
                try:
                    detected, mapped, annotated = analyze_image(tmp_path, yolo_model, clf_model, preprocess, labels)
                except Exception:
                    detected = []

                if detected:
                    st.markdown("**Detected objects:**")
                    st.write(", ".join(detected))
                else:
                    st.caption("Detected objects: none")

                # NO manual override for images (AI-only saving)
                override = None

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
            # images will be saved by backend into AI folders (no override)
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

# --------------------------------------------------------------
# GLOBAL QUERY: SQL + JSON + IMAGES (Hybrid)
# --------------------------------------------------------------
st.write("---")
st.subheader("🔎 Global Query — SQL + JSON + Images")

if "query_page" not in st.session_state:
    st.session_state["query_page"] = 1

if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""

query_input = st.text_input(
    "Enter SQL (SELECT ...) or keyword / filter (e.g. `age > 25`, `India`, `user = 10`)",
    value=st.session_state["query_text"],
)

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    run_btn = st.button("Run Query")
with c2:
    prev_btn = st.button("Prev Page (SQL)", disabled=st.session_state["query_page"] <= 1)
with c3:
    next_btn = st.button("Next Page (SQL)")

if run_btn:
    st.session_state["query_text"] = query_input
    st.session_state["query_page"] = 1
elif prev_btn:
    st.session_state["query_page"] = max(1, st.session_state["query_page"] - 1)
elif next_btn:
    st.session_state["query_page"] = st.session_state["query_page"] + 1

active_query = st.session_state["query_text"].strip()

if active_query:
    try:
        result = hybrid_query(
            active_query,
            page=st.session_state["query_page"],
            page_size=100,
        )
    except Exception as e:
        st.error(f"Query failed: {e}")
        result = None

    if result:
        if result.get("error"):
            st.error(result["error"])
        else:
            mode = result.get("mode")

            # ---------------- SQL MODE (SELECT ...)
            if mode == "sql":
                st.markdown("### 🟢 SQL Query Results")
                st.caption(
                    f"Page {result.get('page', 1)} • Page size {result.get('page_size', 100)}"
                )

                db_results = result.get("db_results", [])
                if not db_results:
                    st.info("No SQLite databases found in `output/JSON/SQL`.")
                else:
                    for db_entry in db_results:
                        st.markdown(f"#### DB: `{db_entry.get('db_path', '')}`")
                        if db_entry.get("error"):
                            st.error(db_entry["error"])
                            continue

                        total_rows = db_entry.get("total_rows", 0)
                        st.caption(f"Total rows: {total_rows}")

                        rows = db_entry.get("rows", [])
                        if rows:
                            st.dataframe(rows, use_container_width=True)
                        else:
                            st.write("No rows on this page.")

            # ---------------- TEXT / FILTER MODE (JSON + SQL keyword + images)
            else:
                st.markdown("### 🟡 JSON Results")
                json_res = result.get("json_results", {})
                jmatches = json_res.get("matches", []) if isinstance(json_res, dict) else []
                st.caption(
                    f"Matches: {len(jmatches)} • Files scanned: {json_res.get('total_files_scanned', 0)}"
                )

                max_json_show = min(50, len(jmatches))
                for m in jmatches[:max_json_show]:
                    title = f"{m.get('file', '')} [index {m.get('index', 0)}]"
                    with st.expander(title):
                        st.json(m.get("object", {}))

                st.markdown("### 🟣 SQL Keyword Matches (Simple Scan)")
                sql_kw = result.get("sql_keyword_results", {})
                dbs = sql_kw.get("dbs", []) if isinstance(sql_kw, dict) else []
                if not dbs:
                    st.caption("No SQL keyword matches.")
                else:
                    for db_entry in dbs:
                        st.markdown(f"#### DB: `{db_entry.get('db_path', '')}`")
                        for t in db_entry.get("tables", []):
                            st.markdown(f"**Table:** `{t.get('table', '')}`")
                            rows = t.get("rows", [])
                            if rows:
                                st.dataframe(rows, use_container_width=True)
                            else:
                                st.caption("No rows matched in this table.")

                st.markdown("### 🔵 Image Path Matches")
                img_res = result.get("image_results", {})
                imgs = img_res.get("matches", []) if isinstance(img_res, dict) else []
                st.caption(f"Image matches: {len(imgs)}")
                max_imgs_show = min(50, len(imgs))
                for p in imgs[:max_imgs_show]:
                    st.write(p)

st.write("---")
st.caption("CortexVault — Clean, Error-Free Streamlit UI")
