"""
Merged organizer_backend — JSON ingestion + SQL/NoSQL decision + image analysis + AI-only image saving + hybrid query

Behaviors:
- JSON ingestion & SQL/NoSQL decisioning (rich diagnostics) preserved.
- Images are analyzed (YOLO/EfficientNet) and saved into AI category folders under top-level 'output'.
  Example: output/Human/Adults/Male/img.jpg
- If ML libs unavailable or model loading fails, images fallback to output/Images/.
- Videos/documents saved as before (organized under folders like Videos, Documents/<ext>).
- NEW: Hybrid query engine for everything in output:
    - SQL: run SELECT queries across all .sqlite DBs (paginated)
    - JSON: filter or keyword search across all JSON files
    - Images: name/category substring search
"""

import os
import json
import sqlite3
import shutil
import re
from glob import glob
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

# ---------------- CONFIG ----------------
ORGANIZED_ROOT = "output"   # <-- top-level folder as requested
JSON_BASE = os.path.join(ORGANIZED_ROOT, "JSON")
SQL_DIR = os.path.join(JSON_BASE, "SQL")
NOSQL_DIR = os.path.join(JSON_BASE, "NoSQL")
SCHEMA_DIR = os.path.join(SQL_DIR, "schemas")

# thresholds for JSON heuristics
STABLE_PRESENCE_THRESHOLD = 0.7
MAX_KEYS_FOR_SQL_FLAT = 200
HETEROGENEITY_KEY_COUNT = 400

# ---------------- helpers ----------------

def ensure(path: str):
    os.makedirs(path, exist_ok=True)


def now_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S")


def pytype(x: Any) -> str:
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, int) and not isinstance(x, bool):
        return "int"
    if isinstance(x, float):
        return "float"
    if isinstance(x, str):
        return "str"
    if isinstance(x, dict):
        return "dict"
    if isinstance(x, list):
        return "list"
    return type(x).__name__


def merge_type_set(s: set) -> str:
    if not s:
        return "str"
    if s <= {"int"}:
        return "int"
    if s <= {"int", "float"}:
        return "float"
    if s <= {"bool"}:
        return "bool"
    return "str"


def sql_type(py_t: str) -> str:
    return {"int": "INTEGER", "float": "REAL", "bool": "BOOLEAN"}.get(py_t, "TEXT")

# ---------------- JSON validation & normalization ----------------

def validate_and_normalize_json(inp: Any) -> Tuple[bool, str, Dict[str, List[Dict]]]:
    if isinstance(inp, dict):
        is_obj_of_arrays = all(isinstance(v, list) for v in inp.values())
        if is_obj_of_arrays and len(inp) > 0:
            for k, v in inp.items():
                if len(v) == 0:
                    return False, f"Collection '{k}' is empty.", {}
                if not all(isinstance(item, dict) for item in v):
                    return False, f"Collection '{k}' must be an array of objects.", {}
            return True, "multi-collection", {k: v for k, v in inp.items()}
        if len(inp) == 0:
            return False, "JSON object is empty.", {}
        return True, "single-object", {"ingest": [inp]}

    if isinstance(inp, list):
        if len(inp) == 0:
            return False, "JSON array is empty.", {}
        if not all(isinstance(item, dict) for item in inp):
            return False, "Every item in the JSON array must be an object.", {}
        return True, "single-collection-array", {"ingest": inp}

    return False, "JSON must be an object or an array of objects.", {}

# ---------------- diagnostics & decisioning ----------------

def analyze_collection(objects: List[Dict]) -> Dict:
    n = len(objects)
    stats = {}
    nesting = False

    for o in objects:
        if not isinstance(o, dict):
            continue
        for k, v in o.items():
            st = stats.setdefault(k, {"count": 0, "nulls": 0, "types": set()})
            st["count"] += 1
            if v is None:
                st["nulls"] += 1
            else:
                st["types"].add(pytype(v))
            if isinstance(v, (dict, list)):
                nesting = True

    metrics = {}
    for k, st in stats.items():
        presence = st["count"] / max(1, n)
        metrics[k] = {
            "presence": presence,
            "null_rate": st["nulls"] / max(1, n),
            "types": sorted(list(st["types"]))
        }

    return {"n": n, "n_keys": len(stats), "nesting": nesting, "metrics": metrics}


def decide_sql_nosql_for_col(objects: List[Dict]) -> Tuple[str, Dict]:
    diag = analyze_collection(objects)
    n_keys = diag["n_keys"]
    nesting = diag["nesting"]

    stable_keys = sum(1 for m in diag["metrics"].values() if m["presence"] >= STABLE_PRESENCE_THRESHOLD)
    stable_ratio = stable_keys / max(1, n_keys)

    if n_keys > HETEROGENEITY_KEY_COUNT:
        return "nosql", diag

    if stable_ratio >= 0.7 and not nesting and n_keys <= MAX_KEYS_FOR_SQL_FLAT:
        return "sql", diag

    if nesting:
        nested_fields = [k for k, m in diag["metrics"].items() if any(t in ("dict", "list") for t in m["types"])]
        if len(nested_fields) <= max(1, n_keys // 5):
            return "sql", diag
        else:
            return "nosql", diag

    if stable_ratio < 0.4:
        return "nosql", diag

    return "sql", diag

# ---------------- schema inference + normalization ----------------

def infer_flat_schema(objects: List[Dict]) -> Dict[str, str]:
    diag = analyze_collection(objects)
    schema = {}
    for k, m in diag["metrics"].items():
        t = merge_type_set(set(m["types"]) - {"null"})
        schema[k] = sql_type(t)
    return schema


def normalize_objects(parent_name: str, objects: List[Dict]) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    collections: Dict[str, List[Dict]] = {parent_name: []}
    schemas_info = {}

    for idx, o in enumerate(objects):
        parent_row = {"__orig_index": idx}

        for k, v in o.items():
            if isinstance(v, dict):
                child_table = f"{parent_name}__{k}"
                child = dict(v)
                child["__parent_index"] = idx
                collections.setdefault(child_table, []).append(child)
            elif isinstance(v, list):
                if len(v) == 0:
                    parent_row[k] = None
                elif all(isinstance(x, dict) for x in v):
                    child_table = f"{parent_name}__{k}"
                    for item in v:
                        child = dict(item)
                        child["__parent_index"] = idx
                        collections.setdefault(child_table, []).append(child)
                else:
                    parent_row[k] = json.dumps(v)
            else:
                parent_row[k] = v

        collections[parent_name].append(parent_row)

    for tname, recs in collections.items():
        schemas_info[tname] = analyze_collection(recs)

    return collections, schemas_info

# ---------------- sqlite builder ----------------

def create_sqlite_with_relationships(db_path: str, collections: Dict[str, List[Dict]]) -> Dict:
    ensure(os.path.dirname(db_path) or ".")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    table_summaries = {}
    id_maps = {}

    for tname, recs in collections.items():
        diag = analyze_collection(recs)
        flat_keys = [k for k in diag["metrics"].keys() if k not in ("__orig_index", "__parent_index")]
        schema = {}
        for k in flat_keys:
            types = set(diag["metrics"][k]["types"]) - {"null"}
            schema[k] = sql_type(merge_type_set(types))
        has_parent_index = any("__parent_index" in r for r in recs)
        if has_parent_index:
            schema["parent_table_index_ref"] = "INTEGER"

        cols_def = ", ".join(f'"{k}" {v}' for k, v in schema.items())
        create_sql = f'CREATE TABLE IF NOT EXISTS "{tname}" (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols_def})'
        cur.execute(create_sql)
        table_summaries[tname] = {"schema": schema, "inserted": 0}

    conn.commit()

    for tname, recs in collections.items():
        diag = analyze_collection(recs)
        cols = [k for k in diag["metrics"].keys() if k != "__orig_index"]
        placeholders = ",".join(["?"] * len(cols)) if cols else ""
        if cols:
            insert_sql = f'INSERT INTO "{tname}" ({",".join(cols)}) VALUES ({placeholders})'
        else:
            insert_sql = f'INSERT INTO "{tname}" DEFAULT VALUES'

        id_map = {}
        for rec in recs:
            values = [rec.get(c) for c in cols] if cols else []
            cur.execute(insert_sql, values)
            rid = cur.lastrowid
            table_summaries[tname]["inserted"] += 1
            orig_idx = rec.get("__orig_index")
            if orig_idx is not None:
                id_map[orig_idx] = rid
        id_maps[tname] = id_map

    conn.commit()

    for tname, summary in table_summaries.items():
        schema = summary["schema"]
        if "parent_table_index_ref" in schema:
            if "__" in tname:
                parent_name = tname.split("__")[0]
                try:
                    cur.execute(f'ALTER TABLE "{tname}" ADD COLUMN parent_id INTEGER')
                except Exception:
                    pass
                cur.execute(f'SELECT id, parent_table_index_ref FROM "{tname}"')
                rows = cur.fetchall()
                for row in rows:
                    row_id, parent_index = row
                    if parent_index is None:
                        continue
                    parent_id = id_maps.get(parent_name, {}).get(parent_index)
                    if parent_id:
                        cur.execute(f'UPDATE "{tname}" SET parent_id = ? WHERE id = ?', (parent_id, row_id))
    conn.commit()
    conn.close()

    return {"db_path": db_path, "tables": table_summaries}

# ---------------- NoSQL saver ----------------

def save_nosql_objects(collections: Dict[str, List[Dict]], outdir: str) -> Dict:
    ensure(outdir)
    saved = []
    for cname, items in collections.items():
        col_dir = os.path.join(outdir, cname)
        ensure(col_dir)
        for i, item in enumerate(items):
            fname = f"{cname}_{now_ts()}_{i}.json"
            fp = os.path.join(col_dir, fname)
            with open(fp, "w") as f:
                json.dump(item, f, indent=2)
            saved.append(fp)
    return {"saved_files": saved}

# ---------------- main ingest pipeline ----------------

def ingest_json_batch_advanced(batch: Any, target_root: Optional[str] = None) -> Dict:
    target_root = target_root or ORGANIZED_ROOT
    ensure(SQL_DIR)
    ensure(SCHEMA_DIR)
    ensure(NOSQL_DIR)

    ok, msg, collections = validate_and_normalize_json(batch)
    if not ok:
        return {"error": msg}

    decisions = {}
    diagnostics = {}

    for cname, items in collections.items():
        dec, diag = decide_sql_nosql_for_col(items)
        decisions[cname] = dec
        diagnostics[cname] = diag

    want_sql = any(d == "sql" for d in decisions.values())
    artifacts = {"db_path": None, "saved_files": [], "schema_files": []}

    if want_sql:
        db_path = os.path.join(SQL_DIR, f"ingest_{now_ts()}.sqlite")
        combined_collections: Dict[str, List[Dict]] = {}
        combined_schema_info = {}

        for cname, items in collections.items():
            if decisions[cname] == "sql":
                norm_name = cname
                cols, sinfo = normalize_objects(norm_name, items)
                for t, recs in cols.items():
                    combined_collections.setdefault(t, []).extend(recs)
                combined_schema_info.update(sinfo)

        try:
            summary = create_sqlite_with_relationships(db_path, combined_collections)
            artifacts["db_path"] = summary["db_path"]
            schema_fp = os.path.join(SCHEMA_DIR, f"schema_diag_{now_ts()}.json")
            with open(schema_fp, "w") as f:
                json.dump({"tables": summary["tables"], "schema_info": combined_schema_info}, f, indent=2)
            artifacts["schema_files"].append(schema_fp)
        except Exception as e:
            return {"error": f"SQL ingestion failed: {e}"}

    nosql_out = os.path.join(NOSQL_DIR, f"ingest_{now_ts()}")
    for cname, items in collections.items():
        if decisions[cname] == "nosql":
            save_res = save_nosql_objects({cname: items}, nosql_out)
            artifacts["saved_files"].extend(save_res["saved_files"])

    result = {
        "decision_summary": decisions,
        "diagnostics": diagnostics,
        "artifacts": artifacts,
        "message": "Ingestion complete"
    }

    return result


def ingest_json_batch(batch: Any, target_root: Optional[str] = None) -> Dict:
    return ingest_json_batch_advanced(batch, target_root)

# ---------------- analyze_path & AI utilities ----------------

def analyze_path(path: Optional[str], batch_json=None):
    """
    If batch_json provided -> ingest
    If path is json file -> parse & ingest
    Otherwise detect file type (image/video/document)
    """
    import os as _os, json as _json
    if batch_json is not None:
        return ingest_json_batch(batch_json)

    if path is None:
        return {"type": "error", "error": "No path provided"}

    ext = str(path).lower().split(".")[-1]

    if ext == "json":
        try:
            parsed = _json.load(open(path, "r"))
            return {
                "type": "json-file-ingested",
                "preview": _json.dumps(parsed, indent=2)[:800],
                "ingest_result": ingest_json_batch(parsed)
            }
        except Exception as e:
            return {"type": "json-error", "error": str(e)}

    IMAGE_EXTS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}
    VIDEO_EXTS = {"mp4", "mov", "avi", "mkv", "webm"}
    DOC_EXTS   = {"pdf", "txt", "csv", "xml", "html", "log", "md", "docx"}

    if ext in IMAGE_EXTS:
        return {"type": "image", "suggested_folder": "Images", "snippet": None}
    if ext in VIDEO_EXTS:
        return {"type": "video", "suggested_folder": "Videos", "snippet": None}
    if ext in DOC_EXTS:
        try:
            txt = open(path, "r", errors="ignore").read()[:800]
        except:
            txt = ""
        return {"type": "document", "suggested_folder": f"Documents/{ext}", "snippet": txt}

    return {"type": "document", "suggested_folder": f"Documents/{ext}", "snippet": ""}

# ---------------- Image analysis utilities (best-effort) ----------------

_ml_available = False
try:
    import torch
    from PIL import Image
    import numpy as np
    from torchvision import models
    from ultralytics import YOLO
    _ml_available = True
except Exception:
    _ml_available = False

def simplify_category(label: str):
    label = label.lower().strip()
    gender_map = {
        "man": "Male", "woman": "Female", "groom": "Male", "bride": "Female",
        "boy": "Male", "girl": "Female"
    }
    categories = {
        "Human": {
            "Adults": ["groom", "bride", "man", "woman", "person"],
            "Children": ["boy", "girl", "kid", "child", "baby"],
            "Profession": ["doctor", "police", "fireman", "chef", "soldier", "athlete"],
        },
        "Animal": {
            "Mammals": ["cat", "dog", "horse", "cow", "sheep", "lion", "tiger", "bear", "elephant", "monkey", "fox"],
            "Birds": ["bird", "eagle", "parrot", "owl", "pigeon", "sparrow", "chicken"],
            "Reptiles": ["snake", "lizard", "crocodile", "turtle"],
            "Aquatic": ["fish", "shark", "whale", "dolphin"],
            "Insects": ["bee", "butterfly", "ant", "spider", "beetle"],
        },
        "Vehicle": {
            "Land": ["car", "bus", "truck", "train", "bike", "motorcycle", "scooter", "bicycle", "tractor"],
            "Air": ["airplane", "jet", "helicopter", "glider"],
            "Water": ["boat", "ship", "yacht", "canoe", "submarine"],
        },
        "Nature": {
            "Plants": ["tree", "flower", "plant", "bush", "grass"],
            "Landscapes": ["mountain", "valley", "hill", "desert", "canyon"],
            "Waterbodies": ["river", "lake", "ocean", "sea", "beach", "waterfall"],
            "Sky": ["cloud", "sun", "moon", "star", "rainbow", "sunset"],
        },
        "Building": {
            "Residential": ["house", "apartment", "villa"],
            "Religious": ["church", "temple", "mosque", "cathedral"],
            "Infrastructure": ["bridge", "tower", "castle", "skyscraper", "monument"],
        },
        "Food": {
            "Fast Food": ["pizza", "burger", "fries", "sandwich", "taco"],
            "Desserts": ["cake", "ice cream", "cookie", "doughnut"],
            "Fruits & Veggies": ["apple", "banana", "orange", "carrot", "tomato", "broccoli"],
            "Drinks": ["coffee", "tea", "juice", "wine", "beer", "bottle"],
        },
        "Electronic": {
            "Computers": ["computer", "laptop", "keyboard", "monitor", "mouse"],
            "Mobile Devices": ["phone", "tablet", "camera", "headphone"],
            "Home Devices": ["tv", "microwave", "fridge", "speaker", "remote"],
        },
        "Sports": {
            "Outdoor": ["ball", "bat", "racket", "skateboard", "surfboard"],
            "Team Sports": ["football", "basketball", "cricket", "baseball"],
            "Equipment": ["helmet", "glove", "net", "jersey"],
        },
        "Tools": {
            "Hand Tools": ["hammer", "knife", "wrench", "screwdriver", "pliers"],
            "Power Tools": ["drill", "saw", "grinder"],
            "Hardware": ["nail", "bolt", "screw", "gear"],
        },
        "Clothing": {
            "Upper Body": ["shirt", "tshirt", "jacket", "coat"],
            "Lower Body": ["pants", "jeans", "shorts", "skirt"],
            "Accessories": ["hat", "shoe", "dress", "tie", "belt"],
        },
        "Art & Culture": {
            "Visual Art": ["painting", "sculpture", "vase", "drawing"],
            "Cultural": ["mask", "statue", "artifact"],
            "Music": ["instrument", "guitar", "piano", "violin", "drum"],
        },
        "Text & Signs": {
            "Documents": ["book", "paper", "newspaper", "menu"],
            "Signs": ["sign", "board", "poster", "label"],
        },
        "Appliances": {
            "Kitchen": ["oven", "stove", "blender", "toaster"],
            "Home": ["washing machine", "vacuum", "fan"],
        },
        "Furniture": {
            "Seating": ["chair", "sofa", "bench", "stool"],
            "Storage": ["table", "desk", "cabinet", "shelf"],
            "Bedding": ["bed", "mattress", "pillow"],
        },
        "Miscellaneous": {
            "Objects": ["bag", "box", "bottle", "watch", "key"],
        }
    }

    for main_cat, subcats in categories.items():
        for subcat, keywords in subcats.items():
            for kw in sorted(keywords, key=lambda x: len(x), reverse=True):
                if kw in label:
                    if main_cat == "Human" and kw in gender_map:
                        return main_cat, subcat, gender_map[kw]
                    return main_cat, subcat, kw.title()
    return "Uncategorized", "Unknown", "Unknown"

def load_models():
    """
    Loads YOLO + classifier if available. Returns (yolo_model, clf_model, preprocess, labels).
    If ML libs not available, returns (None, None, None, None).
    """
    if not _ml_available:
        return None, None, None, None

    yolo_model = None
    clf_model = None
    preprocess = None
    labels = []

    try:
        yolo_model = YOLO("yolov8s.pt")
    except Exception:
        yolo_model = None

    try:
        clf_model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        clf_model.eval()
        preprocess = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms()
        labels = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1.meta["categories"]
    except Exception:
        try:
            clf_model = models.efficientnet_v2_m(pretrained=True)
            clf_model.eval()
            preprocess = None
            labels = []
        except Exception:
            clf_model = None
            preprocess = None
            labels = []

    return yolo_model, clf_model, preprocess, labels

def analyze_image(img_path, yolo_model, clf_model, preprocess, labels):
    """
    Returns: (detected_labels_list, mapped_categories_list, annotated_numpy_image)
    mapped_categories_list is list of triples (Main, Sub, Specific)
    """
    detected_labels = set()
    mapped_categories = []

    if yolo_model is not None:
        try:
            results = yolo_model(img_path, verbose=False)
            for r in results:
                for box in r.boxes:
                    try:
                        cls_id = int(box.cls[0])
                        label = yolo_model.names.get(cls_id, str(cls_id))
                        detected_labels.add(label)
                    except Exception:
                        continue
        except Exception:
            pass

    if not detected_labels and clf_model is not None and preprocess is not None and labels:
        try:
            image = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                output = clf_model(img_tensor)
                topk = torch.topk(output, k=3).indices[0].tolist()
            for idx in topk:
                if idx < len(labels):
                    detected_labels.add(labels[idx])
        except Exception:
            pass

    for lbl in detected_labels:
        mapped_categories.append(simplify_category(lbl))

    annotated = None
    try:
        if yolo_model is not None:
            results = yolo_model(img_path, verbose=False)
            annotated = results[0].plot()
    except Exception:
        annotated = None

    if annotated is None:
        try:
            image = Image.open(img_path).convert("RGB")
            annotated = __import__("numpy").array(image)
        except Exception:
            annotated = None

    return list(detected_labels), mapped_categories, annotated

# ---------------- save_item (AI-only image saving) ----------------

def save_item(path: str, custom_folder: Optional[str] = None):
    """
    - If JSON file -> ingest
    - If image -> run analyze_image -> save copies to AI category folders under ORGANIZED_ROOT
        Example folder: output/Human/Adults/Male/
    - If ML not available or mapping empty -> fallback to output/Images/
    - If video/document -> copy to suggested folder or custom_folder
    Returns: dict or path(s) describing saved artifact(s)
    """
    IMAGE_EXTS = {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}
    VIDEO_EXTS = {"mp4", "mov", "avi", "mkv", "webm"}

    p_lower = path.lower()
    if p_lower.endswith(".json"):
        try:
            parsed = json.load(open(path, "r"))
            return ingest_json_batch(parsed)
        except Exception as e:
            return {"error": f"JSON ingest error: {e}"}

    ext = p_lower.split(".")[-1]
    # IMAGE case -> AI-only categorization + save
    if ext in IMAGE_EXTS:
        # Try to load models (best-effort)
        try:
            yolo_model, clf_model, preprocess, labels = load_models()
        except Exception:
            yolo_model = clf_model = preprocess = labels = None

        try:
            detected, mapped_categories, _ = analyze_image(path, yolo_model, clf_model, preprocess, labels)
        except Exception:
            detected, mapped_categories = [], []

        saved_paths = []
        if mapped_categories:
            # Each mapped category is (Main, Sub, Specific)
            for main, sub, specific in mapped_categories:
                # Build folder path
                if specific and specific.lower() not in ("unknown", ""):
                    folder_path = os.path.join(ORGANIZED_ROOT, main, sub, specific)
                else:
                    folder_path = os.path.join(ORGANIZED_ROOT, main, sub)
                ensure(folder_path)
                dst = os.path.join(folder_path, os.path.basename(path))
                # avoid overwriting by appending timestamp if exists
                if os.path.exists(dst):
                    name, ce = os.path.splitext(os.path.basename(path))
                    dst = os.path.join(folder_path, f"{name}_{now_ts()}{ce}")
                shutil.copy2(path, dst)
                saved_paths.append(dst)
        else:
            # Fallback single folder
            fallback_dir = os.path.join(ORGANIZED_ROOT, "Images")
            ensure(fallback_dir)
            dst = os.path.join(fallback_dir, os.path.basename(path))
            if os.path.exists(dst):
                name, ce = os.path.splitext(os.path.basename(path))
                dst = os.path.join(fallback_dir, f"{name}_{now_ts()}{ce}")
            shutil.copy2(path, dst)
            saved_paths.append(dst)

        return {"saved_paths": saved_paths, "detected": detected, "mapped": mapped_categories}

    # VIDEO / Document / Other -> use suggested folder or custom
    folder = custom_folder or analyze_path(path).get("suggested_folder", "Documents/other")
    dest_dir = os.path.join(ORGANIZED_ROOT, folder)
    ensure(dest_dir)
    dest = os.path.join(dest_dir, os.path.basename(path))
    # avoid overwrite
    if os.path.exists(dest):
        name, ce = os.path.splitext(os.path.basename(path))
        dest = os.path.join(dest_dir, f"{name}_{now_ts()}{ce}")
    shutil.copy2(path, dest)
    return dest

# ------------------------------------------------------------------
# HYBRID QUERY ENGINE (SQL + JSON + IMAGE PATHS)
# ------------------------------------------------------------------

def _list_sqlite_dbs() -> List[str]:
    ensure(SQL_DIR)
    pattern = os.path.join(SQL_DIR, "*.sqlite")
    return sorted(glob(pattern))


def _walk_json_files() -> List[str]:
    roots = []
    if os.path.isdir(NOSQL_DIR):
        roots.append(NOSQL_DIR)
    if os.path.isdir(SQL_DIR):
        roots.append(SQL_DIR)
    if os.path.isdir(JSON_BASE):
        roots.append(JSON_BASE)

    seen = set()
    json_files: List[str] = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".json"):
                    full = os.path.join(dirpath, fn)
                    if full not in seen:
                        seen.add(full)
                        json_files.append(full)
    return json_files


def parse_simple_filter(expr: str) -> Optional[Tuple[str, str, Any]]:
    """
    Parse expressions like:
      age > 25
      score >= 10
      country = "India"
      user_id: 123
    Returns (field, op, value) or None if not a filter.
    """
    ops = ["<=", ">=", "!=", ">", "<", "=", ":"]
    for op in ops:
        if op in expr:
            left, right = expr.split(op, 1)
            field = left.strip()
            raw_val = right.strip().strip('"').strip("'")
            if not field:
                return None

            # Try numeric conversion
            if re.fullmatch(r"-?\d+\.\d+", raw_val):
                val: Any = float(raw_val)
            elif re.fullmatch(r"-?\d+", raw_val):
                val = int(raw_val)
            else:
                val = raw_val
            return field, op, val
    return None


def _compare(a: Any, op: str, b: Any) -> bool:
    try:
        # numeric vs numeric
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            av = a
            bv = b
        else:
            av = str(a)
            bv = str(b)
    except Exception:
        av = str(a)
        bv = str(b)

    if op in ("=", ":"):
        return av == bv
    if op == "!=":
        return av != bv
    if op == ">":
        return av > bv
    if op == "<":
        return av < bv
    if op == ">=":
        return av >= bv
    if op == "<=":
        return av <= bv
    return False


def json_query(raw_query: str) -> Dict:
    """
    If raw_query can be parsed as simple filter -> apply field comparison.
    Otherwise -> substring match against JSON text.
    """
    files = _walk_json_files()
    matches: List[Dict[str, Any]] = []
    q_lower = raw_query.lower()
    filt = parse_simple_filter(raw_query)

    for fp in files:
        try:
            with open(fp, "r", errors="ignore") as f:
                data = json.load(f)
        except Exception:
            continue

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = [data]
        else:
            continue

        for idx, obj in enumerate(items):
            if not isinstance(obj, dict):
                continue

            ok = False
            if filt:
                field, op, val = filt
                if field in obj:
                    ok = _compare(obj[field], op, val)
            else:
                try:
                    txt = json.dumps(obj, default=str).lower()
                except Exception:
                    txt = str(obj).lower()
                ok = q_lower in txt

            if ok:
                matches.append({
                    "file": fp,
                    "index": idx,
                    "object": obj
                })

    return {
        "type": "json",
        "query": raw_query,
        "is_filter": bool(filt),
        "matches": matches,
        "total_files_scanned": len(files)
    }


def keyword_search_sql(keyword: str, limit_per_table: int = 50) -> Dict:
    """
    Simple keyword search across all tables in all SQLite DBs.
    For each table, casts all columns to TEXT and applies LIKE.
    """
    dbs_info: List[Dict[str, Any]] = []
    kw = f"%{keyword}%"

    for db in _list_sqlite_dbs():
        db_entry = {"db_path": db, "tables": []}
        try:
            conn = sqlite3.connect(db)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            table_names = [row[0] for row in cur.fetchall()]

            for tname in table_names:
                try:
                    cur.execute(f'PRAGMA table_info("{tname}")')
                    cols_info = cur.fetchall()
                    cols = [c[1] for c in cols_info]
                    if not cols:
                        continue
                    where = " OR ".join([f'CAST("{c}" AS TEXT) LIKE ?' for c in cols])
                    sql = f'SELECT * FROM "{tname}" WHERE {where} LIMIT ?'
                    params = [kw] * len(cols) + [limit_per_table]
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                    if not rows:
                        continue
                    colnames = [d[0] for d in cur.description] if cur.description else cols
                    dict_rows = [dict(zip(colnames, r)) for r in rows]
                    db_entry["tables"].append({
                        "table": tname,
                        "columns": colnames,
                        "rows": dict_rows
                    })
                except Exception:
                    continue

            conn.close()
        except Exception:
            continue

        if db_entry["tables"]:
            dbs_info.append(db_entry)

    return {
        "type": "sql_keyword",
        "keyword": keyword,
        "dbs": dbs_info
    }


def search_images(keyword: str) -> Dict:
    """
    Search image file paths under ORGANIZED_ROOT by substring match.
    """
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    matches: List[str] = []
    q = keyword.lower()

    if not os.path.isdir(ORGANIZED_ROOT):
        return {"type": "images", "keyword": keyword, "matches": []}

    for dirpath, _, filenames in os.walk(ORGANIZED_ROOT):
        for fn in filenames:
            if fn.lower().endswith(IMAGE_EXTS):
                full = os.path.join(dirpath, fn)
                if q in full.lower():
                    matches.append(full)

    return {"type": "images", "keyword": keyword, "matches": matches}


def query_sql_dbs(query: str, page: int = 1, page_size: int = 100) -> Dict:
    """
    Execute a SELECT query across all SQLite DBs in SQL_DIR.
    Uses pagination: wraps query as subquery and applies LIMIT/OFFSET.
    """
    q_clean = query.strip().rstrip(";")
    # Disallow multiple statements for safety
    if ";" in q_clean:
        return {"type": "sql", "error": "Multiple SQL statements are not allowed. Use a single SELECT query."}

    db_results: List[Dict[str, Any]] = []

    for db in _list_sqlite_dbs():
        entry = {
            "db_path": db,
            "rows": [],
            "columns": [],
            "total_rows": 0,
            "error": None
        }
        try:
            conn = sqlite3.connect(db)
            cur = conn.cursor()

            # Count total rows
            count_sql = f"SELECT COUNT(*) FROM ({q_clean}) AS sub"
            cur.execute(count_sql)
            total = cur.fetchone()[0] or 0
            entry["total_rows"] = int(total)

            offset = max(0, (page - 1) * page_size)
            data_sql = f"SELECT * FROM ({q_clean}) AS sub LIMIT ? OFFSET ?"
            cur.execute(data_sql, (page_size, offset))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            entry["columns"] = cols
            entry["rows"] = [dict(zip(cols, r)) for r in rows]

            conn.close()
        except Exception as e:
            entry["error"] = str(e)

        db_results.append(entry)

    return {
        "type": "sql",
        "query": query,
        "page": page,
        "page_size": page_size,
        "db_results": db_results
    }


def hybrid_query(raw_query: str, page: int = 1, page_size: int = 100) -> Dict:
    """
    Main entry point for Streamlit:
    - If query starts with SELECT -> run SQL query with pagination
    - Else -> JSON search + simple SQL keyword search + image path search
    """
    q = (raw_query or "").strip()
    if not q:
        return {"error": "Empty query"}

    if q.lower().startswith("select"):
        sql_res = query_sql_dbs(q, page=page, page_size=page_size)
        if "error" in sql_res:
            return {"mode": "sql", **sql_res}
        return {"mode": "sql", **sql_res}

    # Non-SQL mode: JSON + SQL keyword + images
    json_res = json_query(q)
    sql_kw_res = keyword_search_sql(q)
    img_res = search_images(q)

    return {
        "mode": "filter" if json_res.get("is_filter") else "text",
        "raw_query": q,
        "json_results": json_res,
        "sql_keyword_results": sql_kw_res,
        "image_results": img_res
    }

# ---------------- simple test runner ----------------
if __name__ == "__main__":
    ensure(ORGANIZED_ROOT)
    print("Backend module loaded. ORGANIZED_ROOT=", ORGANIZED_ROOT)
