# app.py
import os
import io
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# OpenAI client (supports both old/new SDK styles)
# -----------------------------
def get_openai_client():
    api_key = st.secrets.get("openai_api_key", None) or os.environ.get("OPENAI_API_KEY", None)
    if not api_key:
        return None, None

    # Try new SDK first (openai>=1.0)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        mode = "v1"
        return client, mode
    except Exception:
        pass

    # Fallback to legacy
    try:
        import openai  # legacy
        openai.api_key = api_key
        client = openai
        mode = "legacy"
        return client, mode
    except Exception:
        return None, None


def ai_chat(messages: List[Dict], model: str = "gpt-4o-mini", temperature: float = 0.3, max_tokens: int = 1200) -> Optional[str]:
    client, mode = get_openai_client()
    if client is None:
        return None

    try:
        if mode == "v1":
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        else:
            # legacy
            resp = client.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp["choices"][0]["message"]["content"]
    except Exception as e:
        st.warning(f"OpenAI error: {e}")
        return None

# -----------------------------
# Utility: session + simple schema profiling
# -----------------------------
def init_state():
    st.session_state.setdefault("challenge", {
        "title": "",
        "theme": "",
        "sponsor": "",
        "due": "",
        "context": "",
        "questions": "",
        "constraints": ""
    })
    st.session_state.setdefault("dfs", {})  # name -> DataFrame
    st.session_state.setdefault("profiles", pd.DataFrame())
    st.session_state.setdefault("table_roles", {})  # name->role
    st.session_state.setdefault("ai_notes", "")
    st.session_state.setdefault("base_measures", [])
    st.session_state.setdefault("ai_measures", [])
    st.session_state.setdefault("palette", [])
    st.session_state.setdefault("pbi_theme_json", "")
    st.session_state.setdefault("rel_suggestions", pd.DataFrame())

def save_session() -> bytes:
    payload = {
        "challenge": st.session_state["challenge"],
        "palette": st.session_state["palette"],
        "pbi_theme_json": st.session_state["pbi_theme_json"],
    }
    # Dataframes as CSV strings for portability
    payload["data"] = {name: df.to_csv(index=False) for name, df in st.session_state["dfs"].items()}
    return json.dumps(payload).encode("utf-8")

def load_session(file):
    content = json.load(file)
    st.session_state["challenge"] = content.get("challenge", st.session_state["challenge"])
    st.session_state["palette"] = content.get("palette", [])
    st.session_state["pbi_theme_json"] = content.get("pbi_theme_json", "")
    data = content.get("data", {})
    newdfs = {}
    for name, csv_str in data.items():
        newdfs[name] = pd.read_csv(io.StringIO(csv_str))
    st.session_state["dfs"] = newdfs

def profile_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        nonnull = series.notna().sum()
        nulls = series.isna().sum()
        unique = series.nunique(dropna=True)
        ex_vals = series.dropna().astype(str).head(3).tolist()
        # Detect semantic hints
        is_date_like = False
        if np.issubdtype(series.dtype, np.datetime64):
            is_date_like = True
        else:
            try:
                pd.to_datetime(series.dropna().astype(str).head(10), errors="raise")
                is_date_like = True
            except Exception:
                pass

        numeric_like = np.issubdtype(series.dtype, np.number)

        rows.append({
            "table": name,
            "column": col,
            "dtype": dtype,
            "is_date_like": bool(is_date_like),
            "is_numeric_like": bool(numeric_like),
            "non_null": int(nonnull),
            "nulls": int(nulls),
            "unique": int(unique),
            "examples": ", ".join(ex_vals)
        })
    return pd.DataFrame(rows)

def guess_table_role(df: pd.DataFrame) -> str:
    num_numeric = sum(np.issubdtype(dt, np.number) for dt in df.dtypes)
    has_date = any(np.issubdtype(dt, np.datetime64) for dt in df.dtypes)
    num_rows = len(df)
    if (num_numeric >= 3 and has_date) or num_rows > 2000:
        return "fact (heuristic)"
    if num_numeric <= 2:
        return "dimension (heuristic)"
    return "unknown"

# -----------------------------
# DAX suggestion engine (heuristics + optional AI)
# -----------------------------
def find_candidate_columns(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, List[str]]]:
    hints = {t: {"date":[], "amount":[], "qty":[], "price":[], "id":[], "cat":[]} for t in dfs}
    for t, df in dfs.items():
        for col in df.columns:
            lc = col.lower()
            if any(k in lc for k in ["date", "day", "period"]):
                hints[t]["date"].append(col)
            if any(k in lc for k in ["amount", "amt", "revenue", "sales", "cost", "price_total", "value"]):
                hints[t]["amount"].append(col)
            if any(k in lc for k in ["qty", "quantity", "units", "# sold", "count"]):
                hints[t]["qty"].append(col)
            if any(k in lc for k in ["price", "unit price", "rate"]):
                hints[t]["price"].append(col)
            if any(k in lc for k in ["id", "key", "code", "number", "no."]) and not any(k in lc for k in ["date"]):
                hints[t]["id"].append(col)
            if any(k in lc for k in ["category", "type", "class", "segment", "group"]):
                hints[t]["cat"].append(col)
    return hints

def heuristic_dax(table: str, hints: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    measures = []
    amt = hints.get("amount", [])
    qty = hints.get("qty", [])
    price = hints.get("price", [])
    date_cols = hints.get("date", [])
    id_cols = hints.get("id", [])

    if amt:
        measures.append((
            "Total Amount",
            f"Total Amount = SUM('{table}'[{amt[0]}])"
        ))
        measures.append((
            "Amount YoY",
            "Amount YoY = \nVAR Curr = [Total Amount]\nVAR Prev = CALCULATE([Total Amount], DATEADD('Dates'[Date], -1, YEAR))\nRETURN Prev"
        ))
        measures.append((
            "Amount YoY % Change",
            "Amount YoY % Change = DIVIDE([Total Amount] - [Amount YoY], [Amount YoY])"
        ))
        measures.append((
            "Amount MTD",
            f"Amount MTD = TOTALMTD([Total Amount], 'Dates'[Date])"
        ))
        measures.append((
            "Amount YTD",
            f"Amount YTD = TOTALYTD([Total Amount], 'Dates'[Date])"
        ))
        measures.append((
            "Amount % of Total (Context)",
            "Amount % of Total (Context) = \nDIVIDE([Total Amount], CALCULATE([Total Amount], A
