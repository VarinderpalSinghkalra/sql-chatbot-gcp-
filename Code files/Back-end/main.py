
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py — Cloud Function (Gen2) / Cloud Run: NL → BigQuery across ALL tables in a dataset (BA-friendly)

Key features:
- Vertex AI (Gemini) prompts to generate BigQuery SQL.
- Parameterized filters for amount/customer/time.
- DATE/TIMESTAMP guardrails based on schema.
- Dry-run validation before execution.
- Business-friendly JSON (summary, clarify, quick choices, suggested prompts, meta).
- Modes: choose (single table) and fanout (multi-table).
- CORS enabled for browser UI.

Fixes:
- Do NOT inject any purchase_date filter unless the user explicitly mentions a time window.
- Hardened customer parsing (no fallback to guessing title-case names like "Show").
- Year-aware amount parsing: 4-digit tokens in 1900–2100 are treated as years (time), not amounts.
"""

import os
import re
import json
import logging
import requests
import google.auth
import google.auth.transport.requests
from flask import Request, jsonify, make_response
from google.cloud import bigquery
from google.api_core.exceptions import BadRequest
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

# ----------------------------------------------------------------
# Logging
# ----------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
PROJECT_ID = os.environ.get("PROJECT_ID", "data-engineering-479617")
LOCATION = os.environ.get("VERTEX_AI_LOCATION", "global")
MODEL = os.environ.get("VERTEX_AI_MODEL", "gemini-2.5-pro")
BQ_DATASET = os.environ.get("BQ_DATASET", "conversational_demo")
BQ_TABLE = os.environ.get("BQ_TABLE", "sales_data")  # default table
MAX_ROWS = int(os.environ.get("MAX_ROWS", "1000"))
BQ_LOCATION = os.environ.get("BQ_LOCATION", "")  # e.g., "US"
TABLE_SELECTION_MODE = os.environ.get("TABLE_SELECTION_MODE", "choose").lower()  # choose | fanout

# --- CORS config ---
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")  # Set to your UI origin in prod

# Known filter columns (extendable)
FILTER_COLUMNS = ["customer", "amount", "purchase_date"]  # purchase_date may be DATE or TIMESTAMP

# Optional synonyms via env (JSON) — e.g. {"customer_name":"customer","revenue":"amount"}
COLUMN_SYNONYMS = os.environ.get("COLUMN_SYNONYMS", "")
try:
    SYNONYMS_MAP: Dict[str, str] = json.loads(COLUMN_SYNONYMS) if COLUMN_SYNONYMS else {}
except Exception:
    SYNONYMS_MAP = {}

# ----------------------------------------------------------------
# BigQuery client & schema cache
# ----------------------------------------------------------------
_bq_client: Optional[bigquery.Client] = None
_DATASET_SCHEMA: Optional[Dict[str, Dict[str, str]]] = None  # {table: {col: type, ...}}

def _get_bq_client() -> bigquery.Client:
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client(project=PROJECT_ID)
    return _bq_client

def _load_dataset_schema() -> Dict[str, Dict[str, str]]:
    """Loads and caches dataset schema: {table: {col: TYPE, ...}}"""
    global _DATASET_SCHEMA
    if _DATASET_SCHEMA is not None:
        return _DATASET_SCHEMA
    bq = _get_bq_client()
    schema_map: Dict[str, Dict[str, str]] = {}
    for t in bq.list_tables(f"{PROJECT_ID}.{BQ_DATASET}"):
        try:
            table_ref = bq.get_table(t.reference)
            cols = {f.name.lower(): f.field_type.upper() for f in table_ref.schema}
            schema_map[t.table_id] = cols
        except Exception as e:
            logging.warning("Failed to fetch schema for table %s: %s", t.table_id, e)
            continue
    if not schema_map:
        schema_map[BQ_TABLE] = {c: "UNKNOWN" for c in FILTER_COLUMNS}
    _DATASET_SCHEMA = schema_map
    logging.info("Loaded dataset schema: %s", json.dumps(_DATASET_SCHEMA, indent=2))
    return _DATASET_SCHEMA

def _candidate_tables(required_cols: List[str]) -> List[str]:
    """Return tables in dataset that contain all required filter columns (after synonyms)."""
    schema_map = _load_dataset_schema()

    def normalize(col: str) -> str:
        return SYNONYMS_MAP.get(col, col).lower()

    req = [normalize(c) for c in required_cols]
    cands: List[str] = []
    for tbl, cols in schema_map.items():
        colset = set(cols.keys())
        if all(c in colset for c in req):
            cands.append(tbl)
    return cands or [BQ_TABLE]

# ----------------------------------------------------------------
# Vertex AI REST using ADC
# ----------------------------------------------------------------
def _get_access_token() -> str:
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    req = google.auth.transport.requests.Request()
    creds.refresh(req)
    return creds.token

def _vertex_endpoint() -> str:
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
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generation_config": {"temperature": 0.0, "max_output_tokens": 768, "top_p": 0.0, "top_k": 1},
    }
    return requests.post(_vertex_endpoint(), headers=headers, json=body, timeout=timeout_sec)

# ----------------------------------------------------------------
# Prompt builder
# ----------------------------------------------------------------
def _build_catalog_prompt(user_input: str, tables: List[str]) -> str:
    schema_map = _load_dataset_schema()
    lines: List[str] = []
    for t in tables:
        cols = schema_map.get(t, {})
        show_cols = sorted(list(set(list(cols.keys()))))[:20]
        lines.append(f"- {t}: {', '.join(show_cols)}")
    catalog_str = "\n".join(lines)
    prompt = f"""
You are a BigQuery SQL generator.
Dataset: `{PROJECT_ID}.{BQ_DATASET}`
Candidate tables (choose exactly one; DO NOT USE JOIN):
{catalog_str}
Rules:
- Output exactly one BigQuery Standard SQL SELECT statement.
- Use ONLY one table from the above catalog; do not reference tables outside the dataset.
- Fully qualify the table as `{PROJECT_ID}.{BQ_DATASET}.<table>`.
- Use real column names shown in the catalog (no invented names).
- Reflect user filters mentioned in the request:
 • Names like "for/by Alice" → WHERE customer = 'Alice'
 • Numeric amounts → WHERE amount … (support =, >, >=, <, <=, BETWEEN)
 • Time windows on purchase_date → last N days/weeks/months, between dates, before/after date
- NEVER use a bare `DATE`. Prefer TIMESTAMP functions or let backend inject parameters (@start_ts, @end_ts).
- Sorting: support ORDER BY amount and LIMIT when the user asks “top”, “highest/lowest”, or “sorted by amount …”.
- Prefer explicit aliases for computed fields (e.g., SUM(amount) AS total_amount).
- If no LIMIT is specified, append LIMIT {MAX_ROWS}.
- Return ONLY raw SQL. No prose, no comments, no code fences.
- The first character must be 'S' from 'SELECT'.
User request:
"{user_input}"
""".strip()
    return prompt

# ----------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------
MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}

def _parse_number_with_units(token: str):
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([kmb])?\s*$", token, flags=re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "").lower()
    mult = 1
    if unit == "k": mult = 1_000
    elif unit == "m": mult = 1_000_000
    elif unit == "b": mult = 1_000_000_000
    return int(round(val * mult))

# ----------------------------------------------------------------
# SQL normalization & guards
# ----------------------------------------------------------------
def _normalize_sql(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip()
    s = re.sub(r"^```(?:sql)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    lines = s.splitlines()
    cleaned = []
    for ln in lines:
        if ln.strip().startswith("--"):
            continue
        cleaned.append(ln)
    s = "\n".join(cleaned).strip()
    m = re.search(r"(?is)\bselect\b[\s\S]+", s)
    if m:
        s = m.group(0).strip()
    return s

def _ensure_limit(sql: str) -> str:
    if " limit " not in sql.lower():
        sql = f"{sql}\nLIMIT {MAX_ROWS}"
    return sql

def _enforce_select_only(sql: str) -> str:
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Generated SQL is not a SELECT statement.")
    return sql

def _ensure_aliases_for_aggregates(sql: str) -> str:
    rules = [
        (r"(?i)\bcount\s*\(\s*\*\s*\)(?!\s+as\s+\w+)", "COUNT(*) AS total_count"),
        (r"(?i)\bsum\s*\(\s*amount\s*\)(?!\s+as\s+\w+)", "SUM(amount) AS total_amount"),
        (r"(?i)\bavg\s*\(\s*amount\s*\)(?!\s+as\s+\w+)", "AVG(amount) AS avg_amount"),
    ]
    out = sql
    for pattern, replacement in rules:
        out = re.sub(pattern, replacement, out)
    return out

# ----------------------------------------------------------------
# Filter extraction (customer + amount + time + sorting)
# ----------------------------------------------------------------
_STOPWORDS = {
    "total","amount","purchased","purchase","sales","sum","revenue",
    "by","for","of","in","on","from","and","or","the","a","an",
    "customer","count","avg","average","minimum","maximum","max","min",
    "top","lowest","highest","sorted","ascending","descending",
    # Command verbs (to avoid mis-parsing as customer names)
    "show","list","display","give","find","fetch","return","get","view","see","tell","present","reveal"
}

def _extract_customer_from_request(user_input: str):
    """Only extract customer when explicitly indicated—no title-case guessing."""
    pats = [
        r"(?i)\bcustomer\s*(?:=|is|:)\s*['\"]?([A-Za-z][A-Za-z0-9 _\-]{0,50})['\"]?",
        r"(?i)\bfor\s+['\"]?([A-Za-z][A-Za-z0-9 _\-]{0,50})['\"]?",
        r"(?i)\bby\s+['\"]?([A-Za-z][A-Za-z0-9 _\-]{0,50})['\"]?",
        r"(?i)\bpurchased\s+by\s+['\"]?([A-Za-z][A-Za-z0-9 _\-]{0,50})['\"]?",
    ]
    for pat in pats:
        m = re.search(pat, user_input)
        if m:
            return m.group(1).strip()
    return None  # no fallback guessing

def _guess_title_case_name(user_input: str):
    # retained for other potential uses (not called from _extract_customer_from_request)
    candidates = re.findall(r"\b([A-Z][a-zA-Z0-9_\-]{1,50})\b", user_input)
    candidates = [c for c in candidates if c.lower() not in _STOPWORDS]
    return candidates[-1] if candidates else None

_AMT_PATS = [
    (r"(?i)\bbetween\s+([0-9kmb\.]+)\s+and\s+([0-9kmb\.]+)\b", "between"),
    (r"(?i)\b(at least|\>=|greater than|more than|over)\s+([0-9kmb\.]+)\b", "ge"),
    (r"(?i)\b(at most|\<=|less than|under|below)\s+([0-9kmb\.]+)\b", "le"),
    (r"(?i)\bspent\s+([0-9kmb\.]+)\b", "eq"),
    (r"(?i)\bamount\s*=\s*([0-9kmb\.]+)\b", "eq"),
    (r"(?i)\b>\s*([0-9kmb\.]+)\b", "gt"),
    (r"(?i)\b<\s*([0-9kmb\.]+)\b", "lt"),
    (r"(?i)\b([0-9kmb\.]+)\b", "eq"),
]

def _extract_amount_filters(user_input: str):
    """
    Parse amount intent from user_input.
    Disambiguation rule: if the captured numeric token looks like a year (1900–2100),
    treat it as NOT an amount (let time parser handle it) and skip returning an amount filter.
    """
    for pat, kind in _AMT_PATS:
        m = re.search(pat, user_input)
        if not m:
            continue

        # Handle BETWEEN first (two-sided range)
        if kind == "between" and len(m.groups()) >= 2:
            low = _parse_number_with_units(m.group(1))
            high = _parse_number_with_units(m.group(2))
            if low is None or high is None:
                continue
            # Year disambiguation: if either boundary looks like a year, skip amount
            if (1900 <= low <= 2100) or (1900 <= high <= 2100):
                return None
            if low > high:
                low, high = high, low
            return {"type": "between", "low": low, "high": high}

        # For single-value patterns, pick the last captured group
        captured = m.group(len(m.groups()))
        val = _parse_number_with_units(captured)
        if val is None:
            continue

        # Year disambiguation (single value): if token looks like a year, don't treat it as amount
        if 1900 <= val <= 2100:
            return None

        # Map kind to normalized type
        if kind == "ge":
            return {"type": "ge", "value": val}
        elif kind == "le":
            return {"type": "le", "value": val}
        elif kind == "gt":
            return {"type": "gt", "value": val}
        elif kind == "lt":
            return {"type": "lt", "value": val}
        else:
            # default: equality
            return {"type": "eq", "value": val}

    return None

def _parse_iso_or_dmy(date_text: str):
    date_text = date_text.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            dt = datetime.strptime(date_text, fmt)
            return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
        except ValueError:
            pass
    return None

def _month_year_to_range(month_name: str, year_str: str):
    month = MONTH_MAP.get(month_name.lower())
    try:
        year = int(year_str)
    except Exception:
        return None, None
    if not month:
        return None, None
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = datetime(year + (1 if month == 12 else 0), (1 if month == 12 else month + 1), 1, tzinfo=timezone.utc)
    return start, end

def _extract_time_filters(user_input: str):
    text = user_input.lower().strip()
    now = datetime.now(timezone.utc)
    if re.search(r"\btoday\b", text):
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        return {"type": "range", "start": start, "end": end}
    if re.search(r"\byesterday\b", text):
        end = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        start = end - timedelta(days=1)
        return {"type": "range", "start": start, "end": end}
    m = re.search(r"\blast\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\b", text)
    if m:
        n = int(m.group(1)); unit = m.group(2)
        if "day" in unit: start = now - timedelta(days=n)
        elif "week" in unit: start = now - timedelta(days=7*n)
        elif "month" in unit: start = now - timedelta(days=30*n)
        else: start = now - timedelta(days=365*n)
        return {"type": "range", "start": start, "end": now}
    m = re.search(r"\bafter\s+([0-9]{4}\-[0-9]{2}\-[0-9]{2}|[0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})\b", text)
    if m:
        start = _parse_iso_or_dmy(m.group(1))
        if start: return {"type": "range", "start": start, "end": None}
    m = re.search(r"\bbefore\s+([0-9]{4}\-[0-9]{2}\-[0-9]{2}|[0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})\b", text)
    if m:
        end = _parse_iso_or_dmy(m.group(1))
        if end: return {"type": "range", "start": None, "end": end}
    m = re.search(r"\bbetween\s+([0-9]{4}\-[0-9]{2}\-[0-9]{2}|[0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})\s+and\s+([0-9]{4}\-[0-9]{2}\-[0-9]{2}|[0-9]{2}[\/\-][0-9]{2}[\/\-][0-9]{4})\b", text)
    if m:
        start = _parse_iso_or_dmy(m.group(1)); end = _parse_iso_or_dmy(m.group(2))
        if start and end:
            if start > end: start, end = end, start
            return {"type": "range", "start": start, "end": end}
    m = re.search(r"\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b", text)
    if m:
        start, end = _month_year_to_range(m.group(1), m.group(2))
        if start and end: return {"type": "range", "start": start, "end": end}
    if re.search(r"\bthis month\b", text):
        start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        end = (start + timedelta(days=32)).replace(day=1)
        return {"type": "range", "start": start, "end": end}
    if re.search(r"\bthis year\b", text):
        start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        end = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
        return {"type": "range", "start": start, "end": end}
    if re.search(r"\bthis quarter\b", text):
        q = (now.month - 1) // 3 + 1
        q_start_month = (q - 1) * 3 + 1
        start = datetime(now.year, q_start_month, 1, tzinfo=timezone.utc)
        end_month = q_start_month + 3
        end_year = now.year + (1 if end_month > 12 else 0)
        end_month = (end_month - 1) % 12 + 1
        end = datetime(end_year, end_month, 1, tzinfo=timezone.utc)
        return {"type": "range", "start": start, "end": end}
    return None

def _extract_sort_topn(user_input: str):
    order, limit = None, None
    text = user_input.lower()
    m = re.search(r"\btop\s+(\d+)\b", text)
    if m:
        limit = int(m.group(1))
        order = order or "DESC"
    if re.search(r"\bhighest\b", text):
        order = "DESC"
    if re.search(r"\blowest\b|\bbottom\b", text):
        order = "ASC"
    m = re.search(r"\bsorted\s+by\s+amount\s+(ascending|asc|descending|desc)\b", text)
    if m:
        order = "ASC" if m.group(1).startswith("asc") else "DESC"
    return {"order": order, "limit": limit}

# ----------------------------------------------------------------
# WHERE insertion/tidy helpers
# ----------------------------------------------------------------
def _append_where(sql: str, where_no_prefix: str) -> str:
    if not where_no_prefix:
        return sql
    if re.search(r"(?is)\bwhere\b", sql):
        return re.sub(r"(?is)\bwhere\b", f"WHERE {where_no_prefix} AND ", sql, count=1)
    pattern = r"(?is)(FROM\s+`\w[\w\-\._]+`\.\w[\w\-\._]+`\.\w[\w\-\._]+`)"
    return re.sub(pattern, r"\1 WHERE " + where_no_prefix, sql, count=1)

def _tidy_where(sql: str) -> str:
    sql = re.sub(r"(?is)\bAND\s+(ORDER\s+BY|LIMIT)", r"\n\1", sql)
    sql = re.sub(r"(?is)(WHERE\s+[^\n]*?)\s*(AND|OR)\s*(\n|\Z)", r"\1\n", sql)
    sql = re.sub(r"(?is)WHERE\s*(\n)*?(ORDER\s+BY|LIMIT)", r"\2", sql)
    sql = re.sub(r"\s{2,}", " ", sql)
    return sql

# ----------------------------------------------------------------
# DATE guardrail
# ----------------------------------------------------------------
def _fix_malformed_date_clause(sql: str, date_col: str) -> str:
    if not date_col:
        return sql
    sql = re.sub(rf"(?is)\b{re.escape(date_col)}\s*>=\s*DATE\b", f"{date_col} >= DATE(@start_ts)", sql)
    sql = re.sub(rf"(?is)\b{re.escape(date_col)}\s*>\s*DATE\b", f"{date_col} > DATE(@start_ts)", sql)
    sql = re.sub(rf"(?is)\b{re.escape(date_col)}\s*<=\s*DATE\b", f"{date_col} <= DATE(@end_ts)", sql)
    sql = re.sub(rf"(?is)\b{re.escape(date_col)}\s*<\s*DATE\b", f"{date_col} < DATE(@end_ts)", sql)
    sql = re.sub(rf"(?is)\b{re.escape(date_col)}\s*=\s*DATE\b", f"{date_col} BETWEEN DATE(@start_ts) AND DATE(@end_ts)", sql)
    return sql

# ----------------------------------------------------------------
# Ensure filters from NL + collect parameters + build summary
# ----------------------------------------------------------------
def _ensure_filters_from_request(user_input: str, sql: str):
    """Return (sql, query_params, summary_parts, clarify, quick_choices)."""
    query_params: List[bigquery.ScalarQueryParameter] = []
    summary_parts: List[str] = []
    clarify: List[str] = []
    quick_choices: List[str] = []

    # Customer (explicit cues only)
    customer_val = _extract_customer_from_request(user_input)
    if customer_val and "customer" in FILTER_COLUMNS:
        has_customer = re.search(r"(?is)\bcustomer\s*=\s*(['\"]).+?\1", sql) is not None
        if not has_customer:
            sql = _append_where(sql, "customer = @customer_name")
            query_params.append(bigquery.ScalarQueryParameter("customer_name", "STRING", customer_val))
        summary_parts.append(f"Customer = '{customer_val}'")

    # Amount
    amt = _extract_amount_filters(user_input)
    if amt and "amount" in FILTER_COLUMNS:
        has_amount = re.search(r"(?is)\bamount\s*(=|>=|<=|>|<|between)\b", sql) is not None
        if not has_amount:
            if amt["type"] == "between":
                sql = _append_where(sql, "amount BETWEEN @amount_low AND @amount_high")
                query_params.append(bigquery.ScalarQueryParameter("amount_low", "INT64", amt["low"]))
                query_params.append(bigquery.ScalarQueryParameter("amount_high", "INT64", amt["high"]))
                summary_parts.append(f"Amount between {amt['low']} and {amt['high']}")
            elif amt["type"] == "ge":
                sql = _append_where(sql, "amount >= @amount")
                query_params.append(bigquery.ScalarQueryParameter("amount", "INT64", amt["value"]))
                summary_parts.append(f"Amount ≥ {amt['value']}")
            elif amt["type"] == "le":
                sql = _append_where(sql, "amount <= @amount")
                query_params.append(bigquery.ScalarQueryParameter("amount", "INT64", amt["value"]))
                summary_parts.append(f"Amount ≤ {amt['value']}")
            elif amt["type"] == "gt":
                sql = _append_where(sql, "amount > @amount")
                query_params.append(bigquery.ScalarQueryParameter("amount", "INT64", amt["value"]))
                summary_parts.append(f"Amount > {amt['value']}")
            elif amt["type"] == "lt":
                sql = _append_where(sql, "amount < @amount")
                query_params.append(bigquery.ScalarQueryParameter("amount", "INT64", amt["value"]))
                summary_parts.append(f"Amount < {amt['value']}")
            else:
                sql = _append_where(sql, "amount = @amount")
                query_params.append(bigquery.ScalarQueryParameter("amount", "INT64", amt["value"]))
                summary_parts.append(f"Amount = {amt['value']}")
        # Remove conflicting literal equality
        sql = re.sub(r"(?is)\bAND\s+amount\s*=\s*\d+\b", "", sql)
    else:
        # Suggest quick choices for amount
        quick_choices += ["Exactly 200", "≥ 200", "Between 100–300"]

    # Time — ONLY if the prompt explicitly has a time window
    tf = _extract_time_filters(user_input)
    if tf and "purchase_date" in FILTER_COLUMNS:
        # Determine column type of purchase_date from schema of the referenced table
        table_name = _current_table_name(sql)
        schema_map = _load_dataset_schema()
        col_type = None
        if table_name:
            tbl = table_name.split('.')[-1].strip('`')
            cols = schema_map.get(tbl, {})
            col_type = cols.get('purchase_date', cols.get('purchase_date'.lower(), None))
        use_date_cast = (col_type == 'DATE')

        clauses: List[str] = []
        if tf.get("start"):
            clauses.append('purchase_date >= ' + ("DATE(@start_ts)" if use_date_cast else "@start_ts"))
            query_params.append(bigquery.ScalarQueryParameter("start_ts", "TIMESTAMP", tf["start"].isoformat()))
            summary_parts.append(f"From {tf['start'].date().isoformat()}")
        if tf.get("end"):
            clauses.append('purchase_date < ' + ("DATE(@end_ts)" if use_date_cast else "@end_ts"))
            query_params.append(bigquery.ScalarQueryParameter("end_ts", "TIMESTAMP", tf["end"].isoformat()))
            summary_parts.append(f"Until {tf['end'].date().isoformat()}")

        if clauses:
            sql = _append_where(sql, " AND ".join(clauses))
    else:
        # No implicit date filter; only suggest time quick choices
        quick_choices += ["Last 7 days", "Last 30 days", "This month"]

    # Sorting / Top-N
    st = _extract_sort_topn(user_input)
    if st.get("order"):
        if "order by" not in sql.lower():
            sql += f"\nORDER BY amount {st['order']}"
        summary_parts.append(f"Sorted by amount {st['order'].lower()}")
    if st.get("limit"):
        sql = re.sub(r"(?is)\blimit\s+\d+\b", "", sql)
        sql += f"\nLIMIT {min(st['limit'], MAX_ROWS)}"
        summary_parts.append(f"Top {min(st['limit'], MAX_ROWS)}")

    if not summary_parts:
        clarify.append("Try including a number or time window, e.g., 'spent 200', 'last 30 days', or 'top 10'.")

    sql = _tidy_where(sql)
    return sql, query_params, summary_parts, clarify, quick_choices

# ----------------------------------------------------------------
# Table handling (extract & qualify)
# ----------------------------------------------------------------
def _extract_first_table(sql: str) -> Optional[str]:
    m = re.search(r"(?is)\bFROM\b\s+`?([\w\-\._]+)`?", sql)
    return m.group(1) if m else None

def _fully_qualify_table(table_name: str) -> str:
    parts = table_name.split(".")
    if len(parts) == 1:
        return f"`{PROJECT_ID}.{BQ_DATASET}.{parts[0]}`"
    if len(parts) == 2:
        ds, tbl = parts
        return f"`{PROJECT_ID}.{ds}.{tbl}`"
    return f"`{parts[0]}.{parts[1]}.{parts[2]}`"

def _enforce_single_dataset_table(sql: str, candidates: List[str]) -> str:
    first = _extract_first_table(sql)
    if not first:
        fq = f"`{PROJECT_ID}.{BQ_DATASET}.{candidates[0]}`"
        sql = re.sub(r"(?is)\bFROM\b\s+[`\"A-Za-z0-9\.\-\_]+", f"FROM {fq}", sql, count=1)
        if " FROM " not in sql.upper():
            sql = re.sub(r"(?is)^SELECT", f"SELECT\nFROM {fq}\n", sql, count=1)
        return sql
    fq = _fully_qualify_table(first)
    sql = re.sub(rf"(?is)\bFROM\b\s+`?{re.escape(first)}`?", f"FROM {fq}", sql, count=1)
    parts = fq.strip("`").split(".")
    if len(parts) == 3 and parts[1] != BQ_DATASET:
        sql = re.sub(
            r"(?is)\bFROM\b\s+`[A-Za-z0-9\-\_]+\.(?:[A-Za-z0-9\-\_]+)\.(?:[A-Za-z0-9\-\_]+)`",
            f"FROM `{PROJECT_ID}.{BQ_DATASET}.\\2`",
            sql,
            count=1
        )
    return sql

# ----------------------------------------------------------------
# Dry-run validation
# ----------------------------------------------------------------
def _dry_run_validate(sql: str, bq: bigquery.Client, job_config: Optional[bigquery.QueryJobConfig]):
    job_config = job_config or bigquery.QueryJobConfig()
    job_config.dry_run = True
    job_config.use_query_cache = False
    if BQ_LOCATION:
        job_config.location = BQ_LOCATION
    try:
        job = bq.query(sql, job_config=job_config)
        _ = job.result()
        return sql, ""
    except BadRequest as e:
        return sql, str(e)

# ----------------------------------------------------------------
# LLM path: choose one table
# ----------------------------------------------------------------
def _generate_sql_choose_mode(user_input: str):
    required = FILTER_COLUMNS
    candidates = _candidate_tables(required)
    prompt = _build_catalog_prompt(user_input, candidates)
    resp = _call_vertex(prompt)
    if not resp.ok:
        raise RuntimeError(f"Vertex AI error: {resp.status_code} {resp.text}")
    payload = resp.json()
    try:
        text = payload["candidates"][0]["content"]["parts"][0].get("text", "")
    except Exception:
        text = ""
    sql = (text or "").strip()
    sql = _normalize_sql(sql)
    # Strip hard-coded numeric amount equality to avoid conflicts with parameterized filters
    sql = re.sub(r"(?is)\bamount\s*=\s*\d+", "", sql)
    sql = _enforce_select_only(sql)
    sql = _enforce_single_dataset_table(sql, candidates)
    sql = _ensure_aliases_for_aggregates(sql)
    sql = _ensure_limit(sql)
    # Remove invalid bare TIMESTAMP RHS introduced by LLM
    sql = re.sub(r"(?is)\bpurchase_date\s*(=|>=|<=|>|<)\s*TIMESTAMP(?!\()", "", sql)
    sql = _tidy_where(sql)
    # Inject filters and collect parameters + BA-friendly fields
    sql, query_params, summary_parts, clarify, quick_choices = _ensure_filters_from_request(user_input, sql)
    return sql, query_params, summary_parts, clarify, quick_choices

# ----------------------------------------------------------------
# Fan-out path: rule-based SQL across multiple tables
# ----------------------------------------------------------------
def _build_rule_sql_for_table(table_name: str, user_input: str):
    fq = f"`{PROJECT_ID}.{BQ_DATASET}.{table_name}`"
    base = f"SELECT customer, amount, purchase_date FROM {fq}"
    sql = base
    query_params: List[bigquery.ScalarQueryParameter] = []

    # Amount
    amt = _extract_amount_filters(user_input)
    if amt:
        if amt["type"] == "between":
            sql = _append_where(sql, "amount BETWEEN @amount_low AND @amount_high")
            query_params.append(bigquery.ScalarQueryParameter("amount_low", "INT64", amt["low"]))
            query_params.append(bigquery.ScalarQueryParameter("amount_high", "INT64", amt["high"]))
        elif amt["type"] == "ge":
            sql = _append_where(sql, "amount >= @amount")
            query_params.append(bigquery.ScalarQueryParameter("amount", "INT64", amt["value"]))
        elif amt["type"] == "le":
            sql = _append_where(sql, "amount <= @amount")
            query_params.append(bigquery.ScalarQueryParameter("amount", "INT64", amt["value"]))
        elif amt["type"] == "gt":
            sql = _append_where(sql, "amount > @amount")
            query_params.append(bigquery.ScalarQueryParameter("amount", "INT64", amt["value"]))
        elif amt["type"] == "lt":
            sql = _append_where(sql, "amount < @amount")
            query_params.append(bigquery.ScalarQueryParameter("amount", "INT64", amt["value"]))
        else:
            sql = _append_where(sql, "amount = @amount")
            query_params.append(bigquery.ScalarQueryParameter("amount", "INT64", amt["value"]))
    # Cleanup any conflicting literal equality
    sql = re.sub(r"(?is)\bAND\s+amount\s*=\s*\d+\b", "", sql)

    # Customer (explicit only)
    cust = _extract_customer_from_request(user_input)
    if cust:
        sql = _append_where(sql, "customer = @customer_name")
        query_params.append(bigquery.ScalarQueryParameter("customer_name", "STRING", cust))

    # Time (only when explicitly present)
    tf = _extract_time_filters(user_input)
    if tf:
        clauses: List[str] = []
        schema_map = _load_dataset_schema()
        cols = schema_map.get(table_name, {})
        col_type = cols.get('purchase_date')
        use_date_cast = (col_type == 'DATE')
        if tf.get('start'):
            clauses.append('purchase_date >= ' + ("DATE(@start_ts)" if use_date_cast else "@start_ts"))
            query_params.append(bigquery.ScalarQueryParameter("start_ts", "TIMESTAMP", tf['start'].isoformat()))
        if tf.get('end'):
            clauses.append('purchase_date < ' + ("DATE(@end_ts)" if use_date_cast else "@end_ts"))
            query_params.append(bigquery.ScalarQueryParameter("end_ts", "TIMESTAMP", tf['end'].isoformat()))
        if clauses:
            sql = _append_where(sql, " AND ".join(clauses))

    # Sorting/limit
    st = _extract_sort_topn(user_input)
    if st.get("order"):
        sql += f"\nORDER BY amount {st['order']}"
    sql = re.sub(r"(?is)\blimit\s+\d+\b", "", sql)
    sql += f"\nLIMIT {MAX_ROWS}"

    sql = _tidy_where(sql)
    return sql, query_params

def _generate_sql_fanout_mode(user_input: str):
    cands = _candidate_tables(FILTER_COLUMNS)
    bq = _get_bq_client()
    all_results: List[Dict[str, Any]] = []
    summaries: List[str] = []
    clarify: List[str] = []

    for t in cands:
        sql, params = _build_rule_sql_for_table(t, user_input)
        job_conf = bigquery.QueryJobConfig(query_parameters=params)
        if BQ_LOCATION:
            job_conf.location = BQ_LOCATION
        _, err = _dry_run_validate(sql, bq, job_conf)
        if err:
            logging.warning("Skip %s due to validation error: %s", t, err)
            continue
        job = bq.query(sql, job_config=job_conf)
        rows = [dict(row) for row in job.result(max_results=MAX_ROWS)]
        for r in rows:
            r["table"] = t
        all_results.extend(rows)

    if not all_results:
        clarify.append("No matching rows found. Try a different amount or time window.")

    amt = _extract_amount_filters(user_input)
    tf = _extract_time_filters(user_input)
    if amt: summaries.append("Amount filter applied")
    if tf: summaries.append("Time window applied")
    if cands: summaries.append(f"Searched {len(cands)} tables in `{BQ_DATASET}`")

    rep_sql, rep_params = _build_rule_sql_for_table(cands[0], user_input)
    quick_choices = []
    if not amt: quick_choices += ["Exactly 200", "≥ 200", "Between 100–300"]
    if not tf: quick_choices += ["Last 7 days", "Last 30 days", "This month"]

    return rep_sql, rep_params, summaries, clarify, all_results, quick_choices

# ----------------------------------------------------------------
# Helpers for response enrichment
# ----------------------------------------------------------------
def _current_table_name(sql: str) -> Optional[str]:
    m = re.search(r"(?is)\bFROM\b\s+`?([\w\-\._]+)`?", sql)
    return m.group(1) if m else None

def _build_explain(summary_parts: List[str], table_used: Optional[str], rows_count: int) -> str:
    summary_str = " → ".join(summary_parts) if summary_parts else "No explicit filters"
    table_str = table_used or "(model-chosen)"
    return f"Applied: {summary_str}. Source table: {table_str}. Returned {rows_count} rows."

def _suggested_prompts() -> List[str]:
    return [
        "Top 10 customers by amount in the last 30 days",
        "Customers who spent ≥ 200k this quarter",
        "Alice’s purchases after 2025-01-01",
        "Average amount for customers with amount > 1000 this year",
        "Customers who spent between 100 and 300 in February 2024"
    ]

# ----------------------------------------------------------------
# CORS helpers
# ----------------------------------------------------------------
def _cors_response(payload, status: int = 200):
    resp = make_response(payload, status)
    resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Access-Control-Max-Age"] = "3600"
    return resp

def _json_response(obj: dict, status: int = 200):
    return _cors_response(jsonify(obj), status)

# ----------------------------------------------------------------
# Entry point (UI-ready with CORS)
# ----------------------------------------------------------------
def app(request: Request):
    """
    HTTP entry: GET -> OK; POST -> JSON with BA-friendly fields and results.
    CORS enabled for browser-based UI.
    """
    if request.method == "OPTIONS":
        return _cors_response("", 204)

    if request.method == "GET":
        return _cors_response("OK", 200)

    try:
        if not request.is_json:
            return _json_response({"error": "Content-Type must be application/json"}, 400)

        data = request.get_json(silent=True) or {}
        user_input = (data.get("query") or "").strip()
        if not user_input:
            return _json_response({"error": "Query text is required"}, 400)

        if TABLE_SELECTION_MODE == "fanout":
            (sql_generated, query_params, summary_parts,
             clarify, precomputed_rows, quick_choices) = _generate_sql_fanout_mode(user_input)

            return _json_response({
                "results": precomputed_rows,
                "summary": " → ".join(summary_parts) if summary_parts else "",
                "clarify": clarify,
                "quick_choices": quick_choices,
                "explain": _build_explain(summary_parts, None, len(precomputed_rows)),
                "suggested_prompts": _suggested_prompts(),
                "sql_generated": sql_generated,
                "sql_final": sql_generated,
                "sql_used": sql_generated,
                "meta": {
                    "rows_returned": len(precomputed_rows),
                    "dataset": BQ_DATASET,
                    "mode": "fanout"
                },
                "model": MODEL,
                "location": LOCATION
            }, 200)

        (sql_generated, query_params, summary_parts,
         clarify, quick_choices) = _generate_sql_choose_mode(user_input)

        bq = _get_bq_client()
        dry_job_config = bigquery.QueryJobConfig(query_parameters=query_params if query_params else [])
        if BQ_LOCATION:
            dry_job_config.location = BQ_LOCATION

        sql_final, validation_err = _dry_run_validate(sql_generated, bq, dry_job_config)
        table_used = _current_table_name(sql_final)

        if validation_err:
            return _json_response({
                "error": "SQL validation failed",
                "details": validation_err,
                "sql_generated": sql_generated,
                "sql_final": sql_final,
                "sql_used": sql_final,
                "summary": " → ".join(summary_parts) if summary_parts else "",
                "clarify": clarify,
                "quick_choices": quick_choices,
                "suggested_prompts": _suggested_prompts(),
                "explain": _build_explain(summary_parts, table_used, 0),
                "meta": {
                    "rows_returned": 0,
                    "dataset": BQ_DATASET,
                    "table_used": table_used,
                    "mode": "choose"
                },
                "model": MODEL,
                "location": LOCATION
            }, 400)

        logging.info("Executing SQL (choose mode):\n%s", sql_final)
        exec_job_config = bigquery.QueryJobConfig(query_parameters=query_params if query_params else [])
        if BQ_LOCATION:
            exec_job_config.location = BQ_LOCATION

        job = bq.query(sql_final, job_config=exec_job_config)
        rows = [dict(row) for row in job.result(max_results=MAX_ROWS)]

        return _json_response({
            "results": rows,
            "summary": " → ".join(summary_parts) if summary_parts else "",
            "clarify": clarify,
            "quick_choices": quick_choices,
            "suggested_prompts": _suggested_prompts(),
            "explain": _build_explain(summary_parts, table_used, len(rows)),
            "sql_generated": sql_generated,
            "sql_final": sql_final,
            "sql_used": sql_final,
            "meta": {
                "rows_returned": len(rows),
                "dataset": BQ_DATASET,
                "table_used": table_used,
                "mode": "choose"
            },
            "model": MODEL,
            "location": LOCATION
        }, 200)

    except Exception as e:
        logging.exception("Error handling request")
        return _json_response({"error": "Internal server error", "details": str(e)}, 500)
