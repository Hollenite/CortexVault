"""
Improved organizer_backend — JSON ingestion + SQL/NoSQL decision + automatic schema + relationships

Features added / changed:
- Accept single JSON object, JSON array of objects, or an object-of-arrays (multiple collections).
- More robust validation and normalization of JSON inputs.
- Heuristics to decide SQL vs NoSQL: considers nesting, key stability, number of distinct keys, and schema depth.
- When suitable for SQL, automatically normalizes nested objects/lists into additional tables and creates FOREIGN KEY relationships.
- Generates SQLite DB with multiple tables (parent + child tables) and inserts records, returning DB path and schema diagnostics.
- Falls back to NoSQL (file export) when JSON is highly heterogeneous or contains data types unsuitable for relational storage.

Notes:
- This file intentionally focuses on correctness and clarity. It uses only Python stdlib + sqlite3.
- For large JSONs, consider streaming/ chunking; this implementation loads data into memory.
"""

import os
import json
import sqlite3
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

# ---------------- CONFIG ----------------
ORGANIZED_ROOT = "organized_storage"
JSON_BASE = os.path.join(ORGANIZED_ROOT, "JSON")
SQL_DIR = os.path.join(JSON_BASE, "SQL")
NOSQL_DIR = os.path.join(JSON_BASE, "NoSQL")
SCHEMA_DIR = os.path.join(SQL_DIR, "schemas")

# thresholds
STABLE_PRESENCE_THRESHOLD = 0.7  # field present in >= 70% of rows considered "stable"
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

# ---------------- validation & normalization ----------------

def validate_and_normalize_json(inp: Any) -> Tuple[bool, str, Dict[str, List[Dict]]]:
    """
    Accepts:
      - single object -> {'ingest': [obj]}
      - array of objects -> {'ingest': array}
      - object of arrays -> {'colname': array, ...}

    Returns (valid, message, normalized_collections)
    """
    if isinstance(inp, dict):
        # Distinguish between object-of-arrays (multiple collections) and a single object
        is_obj_of_arrays = all(isinstance(v, list) for v in inp.values())
        if is_obj_of_arrays and len(inp) > 0:
            # ensure each collection contains dicts
            for k, v in inp.items():
                if len(v) == 0:
                    return False, f"Collection '{k}' is empty.", {}
                if not all(isinstance(item, dict) for item in v):
                    return False, f"Collection '{k}' must be an array of objects.", {}
            return True, "multi-collection", {k: v for k, v in inp.items()}

        # otherwise treat as a single object
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

# ---------------- diagnostics ----------------

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
    """
    Heuristic:
      - If many stable keys (>=70% presence) and not deeply nested, prefer SQL.
      - If heavy nesting or extremely heterogeneous keys, prefer NoSQL.
      - But if nesting is present but can be normalized (dicts/lists of dicts), still allow SQL by creating child tables.
    Returns ('sql'|'nosql', diag)
    """
    diag = analyze_collection(objects)
    n_keys = diag["n_keys"]
    nesting = diag["nesting"]

    stable_keys = sum(1 for m in diag["metrics"].values() if m["presence"] >= STABLE_PRESENCE_THRESHOLD)
    stable_ratio = stable_keys / max(1, n_keys)

    # Extremely heterogeneous -> NoSQL
    if n_keys > HETEROGENEITY_KEY_COUNT:
        return "nosql", diag

    # If stable and not too many keys -> SQL
    if stable_ratio >= 0.7 and not nesting and n_keys <= MAX_KEYS_FOR_SQL_FLAT:
        return "sql", diag

    # If nesting present but nesting looks like dict/list of dicts (normalizable), still allow SQL
    if nesting:
        # check approximate normalizability: look for dict/list types in metrics
        nested_fields = [k for k, m in diag["metrics"].items() if any(t in ("dict", "list") for t in m["types"])]
        if len(nested_fields) <= max(1, n_keys // 5):
            # limited nesting, try SQL
            return "sql", diag
        else:
            return "nosql", diag

    # fallback: if not nested but unstable keys -> nosql
    if stable_ratio < 0.4:
        return "nosql", diag

    return "sql", diag

# ---------------- schema inference + normalization ----------------

def infer_flat_schema(objects: List[Dict]) -> Dict[str, str]:
    # infer type per field among stable keys
    diag = analyze_collection(objects)
    schema = {}
    for k, m in diag["metrics"].items():
        t = merge_type_set(set(m["types"]) - {"null"})
        schema[k] = sql_type(t)
    return schema


def normalize_objects(parent_name: str, objects: List[Dict]) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    """
    Convert nested structures to separate tables.
    Returns (collections, schemas_info) where collections is dict: table_name -> list[records]
    and schemas_info contains diagnostics.

    Rules:
      - Parent table will be named parent_name.
      - Nested dict fields become child tables: parentName_fieldName
      - List of dicts become child tables with one-to-many relationship.
      - Scalar fields stay in parent table.
    """
    collections: Dict[str, List[Dict]] = {parent_name: []}
    schemas_info = {}

    for idx, o in enumerate(objects):
        parent_row = {}
        # assign a temporary __orig_index to help with linking
        parent_row["__orig_index"] = idx

        for k, v in o.items():
            if isinstance(v, dict):
                child_table = f"{parent_name}__{k}"
                child = dict(v)  # shallow copy
                # link back: parent_index
                child["__parent_index"] = idx
                collections.setdefault(child_table, []).append(child)
            elif isinstance(v, list):
                # list may be list of scalars or list of dicts
                if len(v) == 0:
                    # store empty list as null in parent
                    parent_row[k] = None
                elif all(isinstance(x, dict) for x in v):
                    child_table = f"{parent_name}__{k}"
                    for item in v:
                        child = dict(item)
                        child["__parent_index"] = idx
                        collections.setdefault(child_table, []).append(child)
                else:
                    # list of scalars -> store as JSON string in parent
                    parent_row[k] = json.dumps(v)
            else:
                parent_row[k] = v

        collections[parent_name].append(parent_row)

    # produce basic schema diagnostics
    for tname, recs in collections.items():
        schemas_info[tname] = analyze_collection(recs)

    return collections, schemas_info

# ---------------- sqlite builder ----------------

def create_sqlite_with_relationships(db_path: str, collections: Dict[str, List[Dict]]) -> Dict:
    """
    Collections is a dict of table_name -> list of dict records.
    This function will:
      - infer schema for each table
      - add INTEGER PRIMARY KEY autoinc 'id' column
      - if child records contain '__parent_index' or other parent id markers, we'll create a parent_id INTEGER column
      - insert rows and keep mapping from __orig_index to inserted row id to patch foreign keys if needed
    Returns summary with table schemas and inserted counts.
    """
    ensure(os.path.dirname(db_path) or ".")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    table_summaries = {}
    id_maps = {}  # table_name -> mapping from __orig_index -> row_id

    # First pass: create tables with inferred schema (excluding foreign key columns that depend on IDs)
    for tname, recs in collections.items():
        diag = analyze_collection(recs)
        flat_keys = [k for k in diag["metrics"].keys() if k not in ("__orig_index", "__parent_index")]
        schema = {}
        for k in flat_keys:
            types = set(diag["metrics"][k]["types"]) - {"null"}
            schema[k] = sql_type(merge_type_set(types))
        # add parent_id if records carry __parent_index
        has_parent_index = any("__parent_index" in r for r in recs)
        if has_parent_index:
            schema["parent_table_index_ref"] = "INTEGER"  # temporary link column

        cols_def = ", ".join(f'"{k}" {v}' for k, v in schema.items())
        create_sql = f'CREATE TABLE IF NOT EXISTS "{tname}" (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols_def})'
        cur.execute(create_sql)
        table_summaries[tname] = {"schema": schema, "inserted": 0}

    conn.commit()

    # Second pass: insert rows and capture id maps
    for tname, recs in collections.items():
        diag = analyze_collection(recs)
        cols = [k for k in diag["metrics"].keys() if k != "__orig_index"]
        placeholders = ",".join(["?"] * len(cols))
        insert_sql = f'INSERT INTO "{tname}" ({",".join(cols)}) VALUES ({placeholders})'

        id_map = {}
        for rec in recs:
            values = [rec.get(c) for c in cols]
            cur.execute(insert_sql, values)
            rid = cur.lastrowid
            table_summaries[tname]["inserted"] += 1
            orig_idx = rec.get("__orig_index")
            if orig_idx is not None:
                id_map[orig_idx] = rid
        id_maps[tname] = id_map

    conn.commit()

    # Third pass: where child records had parent references stored in 'parent_table_index_ref',
    # we can convert parent_table_index_ref (which stored the parent's __orig_index) into a parent_id that references parent's id.
    # We'll look for patterns: child table names like parent__field indicate parent table named before '__'.

    for tname, summary in table_summaries.items():
        schema = summary["schema"]
        if "parent_table_index_ref" in schema:
            # find parent name
            if "__" in tname:
                parent_name = tname.split("__")[0]
                # rename column and add foreign key semantics (SQLite limited enforcement unless PRAGMA used)
                # We'll add a new parent_id column, copy values, then drop the temp column.
                try:
                    cur.execute(f'ALTER TABLE "{tname}" ADD COLUMN parent_id INTEGER')
                except Exception:
                    pass

                # update parent_id by mapping parent_table_index_ref (which currently holds parent __orig_index)
                # we need to fetch all rows and perform updates
                cur.execute(f'SELECT id, parent_table_index_ref FROM "{tname}"')
                rows = cur.fetchall()
                for row in rows:
                    row_id, parent_index = row
                    if parent_index is None:
                        continue
                    parent_id = id_maps.get(parent_name, {}).get(parent_index)
                    if parent_id:
                        cur.execute(f'UPDATE "{tname}" SET parent_id = ? WHERE id = ?', (parent_id, row_id))

                # optional: leave parent_table_index_ref as audit column
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
    """
    Main entry. Accepts raw parsed JSON and will:
      - validate & normalize
      - decide SQL vs NoSQL per collection
      - if SQL: normalize nested structures into child tables and build an SQLite DB with relationships
      - if NoSQL: write JSON files into NoSQL directory
    Returns a detailed result dict with decisions, diagnostics, and artifact paths.
    """
    target_root = target_root or ORGANIZED_ROOT
    ensure(SQL_DIR)
    ensure(SCHEMA_DIR)
    ensure(NOSQL_DIR)

    ok, msg, collections = validate_and_normalize_json(batch)
    if not ok:
        return {"error": msg}

    decisions = {}
    diagnostics = {}

    # first pass: decide per collection
    for cname, items in collections.items():
        dec, diag = decide_sql_nosql_for_col(items)
        decisions[cname] = dec
        diagnostics[cname] = diag

    # if any sql, we will create a single sqlite DB housing multiple tables.
    want_sql = any(d == "sql" for d in decisions.values())
    artifacts = {"db_path": None, "saved_files": [], "schema_files": []}

    if want_sql:
        db_path = os.path.join(SQL_DIR, f"ingest_{now_ts()}.sqlite")
        # build combined collections after normalization
        combined_collections: Dict[str, List[Dict]] = {}
        combined_schema_info = {}

        for cname, items in collections.items():
            if decisions[cname] == "sql":
                # normalize nested into separate tables
                norm_name = cname
                cols, sinfo = normalize_objects(norm_name, items)
                # merge into combined_collections
                for t, recs in cols.items():
                    combined_collections.setdefault(t, []).extend(recs)
                combined_schema_info.update(sinfo)
            else:
                # store as nosql under a NoSQL folder; we'll still save as files
                pass

        # create sqlite and insert
        try:
            summary = create_sqlite_with_relationships(db_path, combined_collections)
            artifacts["db_path"] = summary["db_path"]
            # write schema diagnostics
            schema_fp = os.path.join(SCHEMA_DIR, f"schema_diag_{now_ts()}.json")
            with open(schema_fp, "w") as f:
                json.dump({"tables": summary["tables"], "schema_info": combined_schema_info}, f, indent=2)
            artifacts["schema_files"].append(schema_fp)
        except Exception as e:
            return {"error": f"SQL ingestion failed: {e}"}

    # handle NoSQL collections (either entirely nosql ones, or those we didn't include in SQL)
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

# ---------------- legacy compatibility wrappers ----------------

def ingest_json_batch(batch: Any, target_root: Optional[str] = None) -> Dict:
    # keep backwards compat; call advanced implementation
    return ingest_json_batch_advanced(batch, target_root)

# ---------------- analyze_path & save_item wrappers ----------------

def analyze_path(path: Optional[str], batch_json=None):
    """
    Compatibility wrapper so Streamlit can still call analyze_path.
    If batch_json is provided, run advanced ingest.
    If path is a JSON file, load and ingest it.
    Otherwise classify the file type (image/video/document).
    """
    import os
    import json
    from pathlib import Path

    # JSON pasted case
    if batch_json is not None:
        return ingest_json_batch(batch_json)

    if path is None:
        return {"type": "error", "error": "No path provided"}

    ext = str(path).lower().split(".")[-1]

    # JSON file case
    if ext == "json":
        try:
            parsed = json.load(open(path, "r"))
            return {
                "type": "json-file-ingested",
                "preview": json.dumps(parsed, indent=2)[:800],
                "ingest_result": ingest_json_batch(parsed)
            }
        except Exception as e:
            return {"type": "json-error", "error": str(e)}

    # Media / Document classification fallback
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

    # fallback
    return {"type": "document", "suggested_folder": f"Documents/{ext}", "snippet": ""}


def save_item(path: str, custom_folder: Optional[str] = None):
    """
    Compatibility wrapper to maintain existing Streamlit calls.
    JSON files → run ingestion
    Media/Docs → copy to proper folder
    """
    import shutil, json

    # JSON ingestion
    if path.lower().endswith('.json'):
        try:
            parsed = json.load(open(path, "r"))
            ingest_json_batch(parsed)
            return "JSON ingestion complete"
        except Exception as e:
            return f"JSON ingest error: {e}"

    # Normal files
    folder = custom_folder or analyze_path(path).get("suggested_folder", "Documents/other")
    dest_dir = os.path.join(ORGANIZED_ROOT, folder)
    ensure(dest_dir)
    dest = os.path.join(dest_dir, os.path.basename(path))
    shutil.copy2(path, dest)
    return dest

# ---------------- simple test runner ---------------- (manual) ----------------
if __name__ == "__main__":
    # quick ad-hoc test: run on sample structures
    sample_single = {"id": 1, "name": "Alice", "meta": {"age": 30, "city": "NY"}, "tags": ["x", "y"]}
    sample_array = [{"id": 1, "a": 10}, {"id": 2, "a": 20}]
    sample_multi = {"users": [{"id": 1, "name": "A"}], "orders": [{"order_id": 9, "user_id": 1, "items": [{"sku": "X", "qty": 1}]}]}

    ensure(ORGANIZED_ROOT)
    print(json.dumps(ingest_json_batch(sample_single), indent=2))
    print(json.dumps(ingest_json_batch(sample_array), indent=2))
    print(json.dumps(ingest_json_batch(sample_multi), indent=2))
