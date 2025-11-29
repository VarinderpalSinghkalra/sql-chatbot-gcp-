
import os
import re
import json
import logging
import requests

import google.auth
import google.auth.transport.requests
from flask import Request, jsonify
from google.cloud import bigquery
from google.api_core.exceptions import BadRequest

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO)

# ------------------------------
# Configuration
# ------------------------------
PROJECT_ID = os.environ.get("PROJECT_ID", "data-engineering-479617")

# Vertex AI publisher REST (use current GA models; global endpoint)
LOCATION = os.environ.get("VERTEX_AI_LOCATION", "global")
MODEL = os.environ.get("VERTEX_AI_MODEL", "gemini-2.5-pro")  # or gemini-2.5-flash

# BigQuery table (your setup)
BQ_DATASET = os.environ.get("BQ_DATASET", "conversational_demo")
BQ_TABLE = os.environ.get("BQ_TABLE", "sales_data")
MAX_ROWS = int(os.environ.get("MAX_ROWS", "1000"))

# Known column names in `sales_data`
ALLOWED_COLUMNS = ["customer", "amount", "purchase_date"]

# Optional synonym mapping via env var JSON (e.g. {"customer_name":"customer"})
COLUMN_SYNONYMS = os.environ.get("COLUMN_SYNONYMS", "")
try:
    SYNONYMS_MAP = json.loads(COLUMN_SYNONYMS) if COLUMN_SYNONYMS else {}
except Exception:
    SYNONYMS_MAP = {}

FQ_TABLE = f"`{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`"

# ------------------------------
# Lazy BigQuery client
# ------------------------------
_bq_client = None

def _get_bq_client() -> bigquery.Client:
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client(project=PROJECT_ID)
    return _bq_client

# ------------------------------
# Vertex AI REST with ADC
# ------------------------------
def _get_access_token() -> str:
    """
    Get OAuth2 access token via ADC (Cloud Functions service account).
    """
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    request = google.auth.transport.requests.Request()
    creds.refresh(request)  # correct refresh flow
    return creds.token

def _vertex_endpoint() -> str:
    """
    Vertex AI publisher-models REST endpoint for generateContent (v1).
    """
    return (
        f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
        f"/locations/{LOCATION}/publishers/google/models/{MODEL}:generateContent"
    )

def _call_vertex(prompt_text: str, timeout_sec: int = 60) -> requests.Response:
    headers = {
        "Authorization": f"Bearer {_get_access_token()}",
        "Content-Type": "application/json",
    }
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt_text}]}
        ]
    }
    return requests.post(_vertex_endpoint(), headers=headers, json=body, timeout=timeout_sec)

# ------------------------------
# Schema-aware prompt
# ------------------------------
def _build_prompt(user_input: str) -> str:
    cols_str = ", ".join(ALLOWED_COLUMNS)
    prompt = f"""
You are a BigQuery SQL generator.
Target table: {FQ_TABLE}
Available columns: {cols_str}

Rules:
- Generate exactly one BigQuery SELECT statement (no DDL/DML).
- Use ONLY the exact column names listed above; do not invent new columns.
- Use {FQ_TABLE} as the only table reference.
- If no LIMIT is specified, append LIMIT {MAX_ROWS}.
- Return ONLY raw SQL. No comments, no prose, no markdown, no code fences.
- The first character of the response must be 'S' from 'SELECT'.

Request: "{user_input}"
    """.strip()
    return prompt

# ------------------------------
# SQL normalization & guards
# ------------------------------
def _normalize_sql(raw: str) -> str:
    """
    Normalize model output to bare SQL:
    - Removes markdown fences (```sql ... ```)
    - Skips leading comment lines
    - Extracts the first SELECT statement if there is leading prose
    """
    if not raw:
        return ""

    s = raw.strip()

    # Remove leading markdown fence like ```sql or ```
    s = re.sub(r"^```(?:sql)?\s*", "", s, flags=re.IGNORECASE)
    # Remove trailing fence ```
    s = re.sub(r"\s*```$", "", s)

    # Remove leading SQL single-line comments
    lines = s.splitlines()
    cleaned = []
    for ln in lines:
        if ln.strip().startswith("--"):
            continue
        cleaned.append(ln)
    s = "\n".join(cleaned).strip()

    # If there is prose before SQL, extract from first 'select' onwards
    m = re.search(r"(?is)\bselect\b[\s\S]+", s)
    if m:
        s = m.group(0).strip()

    return s

def _ensure_limit(sql: str) -> str:
    if " limit " not in sql.lower():
        sql = f"{sql}\nLIMIT {MAX_ROWS}"
    return sql

def _ensure_table(sql: str) -> str:
    # If the SQL does not reference the fully-qualified table, enforce it via FROM clause replacement.
    if FQ_TABLE.strip("`").lower() not in sql.lower() and FQ_TABLE.lower() not in sql.lower():
        sql = re.sub(r"\bfrom\b\s+[^\s]+", f"FROM {FQ_TABLE}", sql, flags=re.IGNORECASE)
    return sql

def _enforce_select_only(sql: str) -> str:
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Generated SQL is not a SELECT statement.")
    return sql

def _apply_synonyms(sql: str, synonyms: dict) -> str:
    # Replace known misnamed columns using provided synonyms map.
    for wrong, right in (synonyms or {}).items():
        if right in ALLOWED_COLUMNS:
            sql = re.sub(rf"\b{re.escape(wrong)}\b", right, sql, flags=re.IGNORECASE)
    return sql

# ------------------------------
# Validation via dry-run
# ------------------------------
def _dry_run_validate(sql: str, bq: bigquery.Client) -> tuple[str, str]:
    """
    BigQuery dry-run. If it fails with 'Unrecognized name', attempt safe auto-correction.
    Returns (final_sql, error_message_or_empty).
    """
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    try:
        job = bq.query(sql, job_config=job_config)
        _ = job.result()  # dry-run parse only
        return sql, ""  # valid
    except BadRequest as e:
        msg = str(e)
        m = re.search(r"Unrecognized name: (\w+)", msg)
        if m:
            missing = m.group(1)
            logging.warning("Dry-run: missing column '%s'. Attempting auto-correction.", missing)
            # 1) Synonyms map
            target = SYNONYMS_MAP.get(missing)
            # 2) Simple heuristics based on known columns
            if not target:
                name_l = missing.lower()
                if ("cust" in name_l or "customer" in name_l) and ("customer" in ALLOWED_COLUMNS):
                    target = "customer"
                elif "amount" in name_l and ("amount" in ALLOWED_COLUMNS):
                    target = "amount"
                elif "date" in name_l and ("purchase_date" in ALLOWED_COLUMNS):
                    target = "purchase_date"

            if target and target in ALLOWED_COLUMNS:
                fixed_sql = re.sub(rf"\b{re.escape(missing)}\b", target, sql, flags=re.IGNORECASE)
                # re-validate
                try:
                    job = bq.query(fixed_sql, job_config=job_config)
                    _ = job.result()
                    return fixed_sql, ""
                except BadRequest as e2:
                    return sql, f"Validation failed after auto-correction: {e2}"
            else:
                return sql, f"Unknown column '{missing}'. Allowed columns: {ALLOWED_COLUMNS}"
        return sql, msg  # other parse error

# ------------------------------
# NL → SQL using Vertex
# ------------------------------
def generate_sql_from_nl(user_input: str) -> str:
    prompt = _build_prompt(user_input)
    resp = _call_vertex(prompt)
    if not resp.ok:
        raise RuntimeError(f"Vertex AI error: {resp.status_code} {resp.text}")

    payload = resp.json()
    try:
        text = payload["candidates"][0]["content"]["parts"][0].get("text", "")
    except Exception:
        text = ""

    sql = (text or "").strip()
    sql = _normalize_sql(sql)       # normalize model output first
    sql = _enforce_select_only(sql) # safety check
    sql = _ensure_table(sql)
    sql = _ensure_limit(sql)
    sql = _apply_synonyms(sql, SYNONYMS_MAP)
    return sql

# ------------------------------
# Cloud Function entry point
# ------------------------------
def app(request: Request):
    """
    HTTP Cloud Function (Gen2):
    - GET  -> health check
    - POST -> JSON {"query": "..."} → generate SQL (schema-aware), validate, execute, return rows.
    """
    if request.method == "GET":
        return ("OK", 200, {"Content-Type": "text/plain"})

    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json(silent=True) or {}
        user_input = (data.get("query") or "").strip()
        if not user_input:
            return jsonify({"error": "Query text is required"}), 400

        # 1) Generate SQL
        sql_query = generate_sql_from_nl(user_input)

        # 2) Validate via dry-run (and auto-correct if needed)
        bq = _get_bq_client()
        final_sql, validation_err = _dry_run_validate(sql_query, bq)
        if validation_err:
            return jsonify({
                "error": "SQL validation failed",
                "details": validation_err,
                "sql_suggested": final_sql,
                "model": MODEL,
                "location": LOCATION
            }), 400

        # 3) Execute
        logging.info("Executing SQL: %s", final_sql)
        job = bq.query(final_sql)
        rows = [dict(row) for row in job.result(max_results=MAX_ROWS)]

        return jsonify({
            "results": rows,
            "sql_used": final_sql,
            "model": MODEL,
            "location": LOCATION
        })

    except Exception as e:
        logging.exception("Error handling request")
        return jsonify({"error": str(e)}), 500
