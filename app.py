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
            "Amount % of Total (Context) = \nDIVIDE([Total Amount], CALCULATE([Total Amount], ALLSELECTED()))"
        ))

    if qty:
        measures.append((
            "Total Quantity",
            f"Total Quantity = SUM('{table}'[{qty[0]}])"
        ))
        measures.append((
            "Average Quantity",
            f"Average Quantity = AVERAGE('{table}'[{qty[0]}])"
        ))

    if amt and qty:
        measures.append((
            "Average Selling Price",
            "Average Selling Price = DIVIDE([Total Amount], [Total Quantity])"
        ))

    if id_cols:
        measures.append((
            "Distinct IDs",
            f"Distinct IDs = DISTINCTCOUNT('{table}'[{id_cols[0]}])"
        ))

    if date_cols:
        measures.append((
            "Amount Rolling 30 Days",
            "Amount Rolling 30 Days = \nCALCULATE([Total Amount], DATESINPERIOD('Dates'[Date], MAX('Dates'[Date]), -30, DAY))"
        ))

    return measures

def ai_dax_refinement(schema_summary: str, base_measures: List[Tuple[str,str]], theme: str) -> Optional[List[Tuple[str,str]]]:
    prompt = f"""
You are a senior Power BI DAX expert. Given the schema summary and base measures, propose 5-10 additional *high-quality* measures (with full DAX) tailored to the challenge theme. 
- Assume a properly marked Dates table named 'Dates'[Date].
- Prefer robust patterns (e.g., ALLSELECTED, time intelligence, segmentation, ranking).
- Avoid duplicate measures from the base list.
- Use table and column names exactly as provided; if unsure, add a short TODO comment with your assumption.

SCHEMA SUMMARY:
{schema_summary}

BASE MEASURES:
{json.dumps([m[0] for m in base_measures], indent=2)}

THEME: {theme}

Return as a numbered list: MEASURE NAME: DAX (only).
"""
    msg = ai_chat([
        {"role": "system", "content": "You are an expert Power BI/DAX consultant."},
        {"role": "user", "content": prompt}
    ], model="gpt-4o-mini", temperature=0.2, max_tokens=1600)
    if not msg:
        return None

    extracted = []
    for line in msg.splitlines():
        if ":" in line:
            name, code = line.split(":", 1)
            name = name.strip().lstrip("0123456789). ").strip()
            code = code.strip()
            if name and code:
                if "=" not in code:
                    code = f"{name} = {code}"
                extracted.append((name, code))
    return extracted or None

# -----------------------------
# Palette generator + PBI theme JSON
# -----------------------------
def simple_palette_from_theme(theme: str) -> List[str]:
    import hashlib, random
    h = hashlib.sha256(theme.encode("utf-8")).hexdigest()
    rng = random.Random(int(h[:8], 16))
    colors = []
    for _ in range(8):
        r = rng.randint(30, 220)
        g = rng.randint(30, 220)
        b = rng.randint(30, 220)
        colors.append("#{0:02X}{1:02X}{2:02X}".format(r, g, b))
    return colors

def ai_palette(theme: str) -> Optional[List[str]]:
    prompt = f"""
Suggest exactly 8 HEX colors (e.g., #123ABC) for a Power BI palette that fits this theme: "{theme}".
Return ONLY the 8 hex codes separated by commas. Ensure accessible contrast variety and avoid near-duplicates.
"""
    msg = ai_chat(
        [{"role": "system", "content": "You are a color theorist and data viz designer."},
         {"role": "user", "content": prompt}],
        model="gpt-4o-mini",
        temperature=0.6,
        max_tokens=200
    )
    if not msg:
        return None
    hexes = re.findall(r"#[0-9A-Fa-f]{6}", msg)
    uniq = []
    for h in hexes:
        u = h.upper()
        if u not in uniq:
            uniq.append(u)
    return uniq[:8] if len(uniq) >= 8 else None

def build_pbi_theme_json(name: str, colors: List[str], bg: str = "#FFFFFF", fg: str = "#000000") -> str:
    theme = {
        "name": name,
        "dataColors": colors,
        "background": bg,
        "foreground": fg,
        "visualStyles": {
            "*": {
                "*": {
                    "title": [{"fontSize": 12}],
                    "labels": [{"color": fg}],
                }
            }
        }
    }
    return json.dumps(theme, indent=2)

# -----------------------------
# Relationship inference helpers
# -----------------------------
def _col_compat(df_left: pd.DataFrame, df_right: pd.DataFrame, c_left: str, c_right: str) -> bool:
    lt = df_left[c_left].dtype
    rt = df_right[c_right].dtype
    if lt == rt:
        return True
    if np.issubdtype(lt, np.number) and np.issubdtype(rt, np.number):
        return True
    if (lt == object) or (rt == object):
        return True
    if (np.issubdtype(lt, np.datetime64) and np.issubdtype(rt, np.datetime64)):
        return True
    return False

def _name_match_score(c_left: str, c_right: str) -> float:
    a = c_left.lower().strip()
    b = c_right.lower().strip()
    if a == b:
        return 1.0
    def norm(s):
        for tok in [" id", "_id", "id_", " key", "_key", " code", "_code", " number", "_number", " no", "_no"]:
            s = s.replace(tok, "")
        return s
    if norm(a) == norm(b):
        return 0.9
    if a in b or b in a:
        return 0.75
    return 0.0

def _sample_jaccard(series_left: pd.Series, series_right: pd.Series, max_samples: int = 5000) -> float:
    try:
        lvals = pd.Series(series_left.dropna().unique())
        rvals = pd.Series(series_right.dropna().unique())
        if len(lvals) > max_samples:
            lvals = lvals.sample(max_samples, random_state=42)
        if len(rvals) > max_samples:
            rvals = rvals.sample(max_samples, random_state=42)
        set_l = set(map(str, lvals))
        set_r = set(map(str, rvals))
        if not set_l or not set_r:
            return 0.0
        inter = len(set_l & set_r)
        union = len(set_l | set_r)
        return inter / union if union else 0.0
    except Exception:
        return 0.0

def _cardinality(left: pd.Series, right: pd.Series) -> str:
    dl = left.nunique(dropna=True)
    dr = right.nunique(dropna=True)
    nl = left.shape[0]
    nr = right.shape[0]
    left_is_keyish = dl >= 0.95 * nl
    right_is_keyish = dr >= 0.95 * nr
    if left_is_keyish and not right_is_keyish:
        return "one-to-many (left→right)"
    if right_is_keyish and not left_is_keyish:
        return "one-to-many (right→left)"
    if left_is_keyish and right_is_keyish:
        return "one-to-one"
    return "many-to-many (check dimensions)"

def infer_relationships(dfs: Dict[str, pd.DataFrame], max_pairs_per_table: int = 10) -> pd.DataFrame:
    suggestions = []
    tables = list(dfs.keys())
    for i in range(len(tables)):
        for j in range(i+1, len(tables)):
            lt, rt = tables[i], tables[j]
            ldf, rdf = dfs[lt], dfs[rt]
            candidates = []
            for lc in ldf.columns:
                for rc in rdf.columns:
                    if not _col_compat(ldf, rdf, lc, rc):
                        continue
                    name_score = _name_match_score(lc, rc)
                    if name_score < 0.5:
                        continue
                    overlap = _sample_jaccard(ldf[lc], rdf[rc])
                    if overlap < 0.05:
                        continue
                    card = _cardinality(ldf[lc], rdf[rc])

                    dl = ldf[lc].nunique(dropna=True)
                    dr = rdf[rc].nunique(dropna=True)
                    dim_side = lt if dl >= dr else rt
                    fact_side = rt if dim_side == lt else lt
                    direction = f"{dim_side} (one) → {fact_side} (many)"

                    score = 0.6 * name_score + 0.4 * overlap
                    candidates.append((lc, rc, score, card, direction))

            candidates.sort(key=lambda x: x[2], reverse=True)
            for (lc, rc, sc, card, direction) in candidates[:max_pairs_per_table]:
                suggestions.append({
                    "left_table": lt,
                    "left_col": lc,
                    "right_table": rt,
                    "right_col": rc,
                    "score": round(sc, 3),
                    "cardinality": card,
                    "direction": direction
                })

    if not suggestions:
        return pd.DataFrame(columns=["left_table","left_col","right_table","right_col","score","cardinality","direction"])

    df = pd.DataFrame(suggestions).sort_values(by=["score"], ascending=False).reset_index(drop=True)
    return df

def relationships_to_json(plan_df: pd.DataFrame) -> str:
    payload = []
    for _, r in plan_df.iterrows():
        payload.append({
            "left_table": r["left_table"],
            "left_col": r["left_col"],
            "right_table": r["right_table"],
            "right_col": r["right_col"],
            "cardinality": r["cardinality"],
            "direction": r["direction"],
            "confidence": r.get("score", None),
        })
    return json.dumps({"relationships": payload}, indent=2)

def m_merge_snippet(left_table: str, left_col: str, right_table: str, right_col: str) -> str:
    """
    Power Query (M) template to type key columns and demo a merge step.
    NOTE: Relationships are created in Power BI Model view; this is for typing/validation.
    """
    return f"""// ---- Begin helper for {left_table} ⇄ {right_table} ----
let
    // Ensure key column types (adjust type if needed)
    {left_table}_Typed = Table.TransformColumnTypes({left_table},{{{{"{left_col}", type text}}}}),
    {right_table}_Typed = Table.TransformColumnTypes({right_table},{{{{"{right_col}", type text}}}}),

    // Example merge (Left Outer) – usually keep tables separate; this checks overlaps
    {left_table}_Merged_{right_table} = Table.NestedJoin(
        {left_table}_Typed, {{"{left_col}"}},
        {right_table}_Typed, {{"{right_col}"}},
        "{right_table}",
        JoinKind.LeftOuter
    ),
    {left_table}_Expanded = Table.ExpandTableColumn({left_table}_Merged_{right_table}, "{right_table}", Table.ColumnNames({right_table}_Typed))
in
    {left_table}_Expanded
// ---- End helper ----
"""

# -----------------------------
# Page config + init
# -----------------------------
st.set_page_config(page_title="Power BI Challenge Helper", page_icon="📊", layout="wide")
init_state()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("📊 Challenge Helper")
    st.caption("Streamline Power BI data challenges")
    st.divider()

    st.subheader("Session")
    st.download_button("💾 Download session JSON", data=save_session(), file_name="pbi_challenge_session.json", mime="application/json", use_container_width=True)

    sess_up = st.file_uploader("Load session JSON", type=["json"], label_visibility="visible", key="session_loader")
    if sess_up:
        try:
            load_session(sess_up)
            st.success("Session loaded.")
        except Exception as e:
            st.error(f"Load failed: {e}")

    st.divider()
    st.subheader("API Keys")
    if get_openai_client()[0] is None:
        st.info("No OpenAI key found. Set `OPENAI_API_KEY` env var or add `openai_api_key` in `.streamlit/secrets.toml`.")
    else:
        st.success("OpenAI connected.")

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "1) Challenge Setup",
    "2) Data",
    "3) AI Review",
    "4) DAX Suggestions",
    "5) Relationships",
    "6) Palette & Theme",
    "7) Inspiration"
])

# -----------------------------
# Tab 1: Challenge Setup
# -----------------------------
with tabs[0]:
    st.header("Challenge Setup")
    c = st.session_state["challenge"]
    col1, col2, col3 = st.columns([2,2,1])
    c["title"] = col1.text_input("Challenge Title", c["title"])
    c["theme"] = col2.text_input("Theme (short phrase, e.g., 'Coastal Birds at Sunrise')", c["theme"])
    c["sponsor"] = col3.text_input("Sponsor", c["sponsor"])
    c["due"] = st.text_input("Due Date (optional)", c["due"])
    c["context"] = st.text_area("Context / Brief", c["context"], height=150, placeholder="Paste the scenario or notes here…")
    c["questions"] = st.text_area("Key Questions", c["questions"], height=120)
    c["constraints"] = st.text_area("Constraints (data/privacy/rules/tools)", c["constraints"], height=100)
    st.session_state["challenge"] = c

    if st.button("🧠 Generate Problem Statement & Plan", use_container_width=True):
        prompt = f"""
You are helping me prepare for a Power BI data challenge.

TITLE: {c['title']}
THEME: {c['theme']}
SPONSOR: {c['sponsor']}
DUE: {c['due']}
CONTEXT: {c['context']}
KEY QUESTIONS: {c['questions']}
CONSTRAINTS: {c['constraints']}

Write:
1) A concise problem statement (2-3 sentences).
2) A prioritized plan of attack (5-10 bullet steps).
3) Risks & mitigations (3-5 bullets).
Keep it practical and focused on delivering a strong Power BI entry.
"""
        out = ai_chat(
            [{"role": "system", "content": "You are a pragmatic analytics lead and data-visualization coach."},
             {"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=800
        )
        if out:
            st.markdown(out)
            st.session_state["ai_notes"] = out
        else:
            st.warning("AI not available. Consider setting your API key.")

# -----------------------------
# Tab 2: Data
# -----------------------------
with tabs[1]:
    st.header("Upload & Inspect Data")
    up = st.file_uploader("Upload CSV or Excel files", type=["csv","xlsx"], accept_multiple_files=True)
    if up:
        for f in up:
            try:
                if f.name.lower().endswith(".csv"):
                    df = pd.read_csv(f)
                else:
                    xls = pd.ExcelFile(f)
                    sheet = st.selectbox(f"Select sheet for {f.name}", xls.sheet_names, key=f"sheet_{f.name}")
                    df = pd.read_excel(xls, sheet_name=sheet)
                st.session_state["dfs"][f.name] = df
                st.success(f"Loaded {f.name} ({df.shape[0]} rows, {df.shape[1]} cols)")
            except Exception as e:
                st.error(f"Failed to load {f.name}: {e}")

    dfs = st.session_state["dfs"]
    if not dfs:
        st.info("No data yet. Upload one or more CSV/XLSX files.")
    else:
        for name, df in dfs.items():
            with st.expander(f"👁 Preview: {name}  |  {df.shape[0]}×{df.shape[1]}  |  role: {guess_table_role(df)}"):
                st.dataframe(df.head(50), use_container_width=True)

        if st.button("🔍 Profile Data", help="Compute quick schema profile for all uploads"):
            profs = []
            roles = {}
            for name, df in dfs.items():
                for col in df.columns:
                    if df[col].dtype == object:
                        try:
                            parsed = pd.to_datetime(df[col], errors="raise")
                            if parsed.notna().mean() > 0.7:
                                df[col] = parsed
                        except Exception:
                            pass
                p = profile_dataframe(df, name)
                profs.append(p)
                roles[name] = guess_table_role(df)
            if profs:
                allp = pd.concat(profs, ignore_index=True)
                st.session_state["profiles"] = allp
                st.session_state["table_roles"] = roles
                st.success("Profile completed.")
        if not st.session_state["profiles"].empty:
            st.subheader("Data Dictionary (Quick)")
            st.dataframe(st.session_state["profiles"], use_container_width=True)

# -----------------------------
# Tab 3: Preliminary AI Review
# -----------------------------
with tabs[2]:
    st.header("Preliminary AI Review")
    if not st.session_state["dfs"]:
        st.info("Upload data first.")
    else:
        prof = st.session_state["profiles"]
        if prof.empty:
            frames = [profile_dataframe(df, name) for name, df in st.session_state["dfs"].items()]
            prof = pd.concat(frames, ignore_index=True)

        schema_summary = prof.to_csv(index=False)
        theme = st.session_state["challenge"]["theme"]
        context = st.session_state["challenge"]["context"]
        questions = st.session_state["challenge"]["questions"]

        if st.button("🧪 Get AI Dataset Review", use_container_width=True):
            prompt = f"""
You are a senior analytics consultant. Review the dataset profile below and produce:
- A concise summary of what data appears to be available.
- Data quality flags or caveats (missingness, duplicates, likely dirty fields).
- Suggested joins and likely grain per table.
- Potential KPIs, dimensions, and cuts.
- 5-10 hypotheses or story angles to explore for a Power BI challenge themed "{theme}".
CONTEXT: {context}
KEY QUESTIONS: {questions}

DATA PROFILE (CSV of columns):
{schema_summary[:200000]}  # truncated if huge
"""
            out = ai_chat(
                [{"role": "system", "content": "You are a precise yet practical analytics advisor."},
                 {"role": "user", "content": prompt}],
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=1400
            )
            if out:
                st.markdown(out)
                st.session_state["ai_notes"] = (st.session_state.get("ai_notes","") + "\n\n" + out).strip()
            else:
                st.warning("AI not available; set your key in the sidebar.")

# -----------------------------
# Tab 4: DAX Suggestions
# -----------------------------
with tabs[3]:
    st.header("DAX Suggestions")
    if not st.session_state["dfs"]:
        st.info("Upload data first.")
    else:
        hints = find_candidate_columns(st.session_state["dfs"])
        all_measures = []
        for table, table_hints in hints.items():
            base = heuristic_dax(table, table_hints)
            if base:
                st.markdown(f"### Table: `{table}`")
                for name, dax in base:
                    with st.expander(name):
                        st.code(dax, language="DAX")
                all_measures.extend([(table, m, d) for (m,d) in base])

        st.session_state["base_measures"] = all_measures

        if st.button("🤖 Ask AI for advanced measures", use_container_width=True):
            schema_lines = []
            for t, df in st.session_state["dfs"].items():
                schema_lines.append(f"TABLE {t}: cols = {list(df.columns)}")
            schema_str = "\n".join(schema_lines)
            base_pairs = [(name, dax) for (_, name, dax) in all_measures]
            refined = ai_dax_refinement(schema_str, base_pairs, st.session_state["challenge"]["theme"])
            if refined:
                st.success("AI suggested additional measures:")
                ai_list = []
                for name, dax in refined:
                    with st.expander(f"AI: {name}"):
                        st.code(dax, language="DAX")
                    ai_list.append(("", name, dax))
                st.session_state["ai_measures"] = ai_list
            else:
                st.warning("No AI measures returned or API not available.")

        if st.session_state["base_measures"] or st.session_state["ai_measures"]:
            rows = []
            for t, name, dax in (st.session_state["base_measures"] + st.session_state["ai_measures"]):
                rows.append({"table": t, "measure": name, "dax": dax})
            out_df = pd.DataFrame(rows)
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download measures (CSV)", csv_bytes, file_name="dax_measures.csv", mime="text/csv")

# -----------------------------
# Tab 5: Relationships
# -----------------------------
with tabs[4]:
    st.header("Relationships (Star-Schema Hints)")
    dfs = st.session_state["dfs"]
    if not dfs:
        st.info("Upload data first in the Data tab.")
    else:
        colA, colB = st.columns([1,1])
        with colA:
            if st.button("🔎 Infer relationships", use_container_width=True):
                rels = infer_relationships(dfs)
                if rels.empty:
                    st.warning("No plausible relationships found. You may need to clean/standardize key columns.")
                else:
                    st.session_state["rel_suggestions"] = rels
        with colB:
            if not st.session_state.get("rel_suggestions", pd.DataFrame()).empty:
                st.success(f"Found {len(st.session_state['rel_suggestions'])} candidate joins")

        if not st.session_state.get("rel_suggestions", pd.DataFrame()).empty:
            st.subheader("Suggested Joins")
            rels = st.session_state["rel_suggestions"].copy()

            min_score = st.slider("Minimum confidence score", 0.0, 1.0, 0.55, 0.05)
            show = rels[rels["score"] >= min_score].reset_index(drop=True)
            if show.empty:
                st.info("No suggestions above this score threshold.")
            else:
                approved_flags = []
                for i, row in show.iterrows():
                    with st.expander(
                        f"({row['score']}) {row['left_table']}[{row['left_col']}] ⇄ {row['right_table']}[{row['right_col']}] • {row['cardinality']} • {row['direction']}"
                    ):
                        approved = st.checkbox("Approve this relationship", key=f"rel_ok_{i}", value=("one-to-many" in row["cardinality"]))
                        approved_flags.append(approved)

                        st.write("**Details**")
                        st.json({
                            "left_table": row["left_table"],
                            "left_col": row["left_col"],
                            "right_table": row["right_table"],
                            "right_col": row["right_col"],
                            "cardinality": row["cardinality"],
                            "direction": row["direction"],
                            "confidence": row["score"],
                        })

                        snippet = m_merge_snippet(row["left_table"], row["left_col"], row["right_table"], row["right_col"])
                        st.code(snippet, language="powerquery")

                plan_df = show.loc[[i for i, ok in enumerate(approved_flags) if ok]].reset_index(drop=True)

                if not plan_df.empty:
                    st.markdown("### Export")
                    rel_json = relationships_to_json(plan_df)
                    st.download_button(
                        "⬇️ Download Relationship Plan (JSON)",
                        rel_json.encode("utf-8"),
                        file_name="relationships_plan.json",
                        mime="application/json"
                    )

                    m_all = []
                    for _, r in plan_df.iterrows():
                        m_all.append(m_merge_snippet(r["left_table"], r["left_col"], r["right_table"], r["right_col"]))
                    st.download_button(
                        "⬇️ Download M Merge Helpers (TXT)",
                        "\n\n".join(m_all).encode("utf-8"),
                        file_name="relationships_merge_helpers.txt",
                        mime="text/plain"
                    )

                    st.info("Tip: In Power BI, create relationships in **Model view**. Use the M snippets to ensure key columns are correctly typed, and to validate overlaps if needed.")
                else:
                    st.caption("Approve one or more suggestions to enable exports.")
        else:
            st.caption("Click **Infer relationships** to scan for likely joins.")

# -----------------------------
# Tab 6: Palette & Theme
# -----------------------------
with tabs[5]:
    st.header("Palette & Power BI Theme")
    theme_phrase = st.text_input("Theme phrase for palette", value=st.session_state["challenge"]["theme"])
    colA, colB = st.columns(2)
    with colA:
        if st.button("🎨 Generate with AI", use_container_width=True):
            cols = ai_palette(theme_phrase)
            if not cols:
                st.warning("AI unavailable; using deterministic fallback.")
                cols = simple_palette_from_theme(theme_phrase)
            st.session_state["palette"] = cols
    with colB:
        if st.button("🎲 Generate fallback palette (no AI)", use_container_width=True):
            st.session_state["palette"] = simple_palette_from_theme(theme_phrase)

    if st.session_state["palette"]:
        st.subheader("Preview (8 colors)")
        cols = st.columns(8)
        for i, col in enumerate(cols):
            hexv = st.session_state["palette"][i]
            with col:
                st.color_picker(f"{i+1}", hexv, key=f"pal_{i}")
        palette = [st.session_state[f"pal_{i}"].upper() for i in range(8)]
        st.session_state["palette"] = palette

        theme_name = st.text_input("Power BI Theme Name", value=(st.session_state["challenge"]["title"] or "PBI Challenge Theme"))
        bg = st.text_input("Background (hex)", value="#FFFFFF")
        fg = st.text_input("Foreground (hex)", value="#000000")
        pbi_json = build_pbi_theme_json(theme_name, palette, bg, fg)
        st.session_state["pbi_theme_json"] = pbi_json

        st.download_button("⬇️ Download Power BI Theme JSON", pbi_json.encode("utf-8"), file_name="powerbi_theme.json", mime="application/json")
        with st.expander("View JSON"):
            st.code(pbi_json, language="json")

# -----------------------------
# Tab 7: Inspiration
# -----------------------------
with tabs[6]:
    st.header("Inspiration Hub")
    st.caption("Brainstorm narrative, layout, visual ideas, and title options.")
    brief = st.text_area("Optional prompt or angle", placeholder="e.g., Focus on migration patterns and seasonality with a clean editorial look.")
    if st.button("💡 Brainstorm ideas", use_container_width=True):
        c = st.session_state["challenge"]
        prompt = f"""
Help brainstorm strong directions for a Power BI challenge.

TITLE: {c['title']}
THEME: {c['theme']}
CONTEXT: {c['context']}
KEY QUESTIONS: {c['questions']}
ANGLE: {brief}

Provide:
- 3 narrative concepts (with what the audience should feel/learn).
- Recommended page plan (2–4 pages, each with suggested visuals).
- Specific chart suggestions tied to measures.
- 5 report title options.
Keep it concise and practical.
"""
        out = ai_chat(
            [{"role": "system", "content": "You are an award-winning data storyteller and Power BI designer."},
             {"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=1000
        )
        if out:
            st.markdown(out)
        else:
            st.warning("AI not available; set your key in the sidebar.")

# -----------------------------
# Footer
# -----------------------------
st.write("---")
st.caption("Built for Timmer’s Power BI data challenges • Streamlit app template")
