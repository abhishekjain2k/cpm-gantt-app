# cpm_app.py
# Streamlit app: CPM in HOURS with Upload-only, Baseline, optional Delayed, and Parallel Cap~Fill (same start)

import json, base64, re
from pathlib import Path
from collections import deque
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Manufacturing CPM (Hours)", layout="wide")
st.title("Manufacturing Gantt – Baseline, Delayed & Parallel Cap~Fill")

# ----------------- Core helpers -----------------
def parse_preds(s: str):
    s = (str(s) or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.replace(";", ",").split(",") if x.strip().isdigit()]

def load_schedule_uploaded(uploaded_file) -> pd.DataFrame:
    """Accepts a Streamlit UploadedFile; reads Excel (first sheet) or CSV."""
    if uploaded_file is None:
        raise ValueError("No file uploaded.")
    # Try Excel
    try:
        raw = pd.read_excel(uploaded_file, sheet_name=None, engine="openpyxl")
        if isinstance(raw, dict):
            first = list(raw.keys())[0]
            return raw[first]
        return raw
    except Exception:
        uploaded_file.seek(0)
        # Try CSV
        return pd.read_csv(uploaded_file)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {
        "TASK": "Task", "Task": "Task", "task": "Task",
        "Duration": "Duration", "duration": "Duration",
        "previous task": "Predecessors", "Previous Task": "Predecessors", "PREVIOUS TASK": "Predecessors",
    }
    df = df.rename(columns=rename_map)
    for col in ["ID", "Task", "Duration", "Predecessors"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df["ID"] = df["ID"].astype(int)
    df["Duration"] = df["Duration"].astype(float)  # HOURS
    df["Predecessors"] = df["Predecessors"].astype(str).replace({"nan": ""})
    return df

def build_graph(frame: pd.DataFrame):
    ids = frame["ID"].tolist()
    preds = {r.ID: set(parse_preds(r.Predecessors)) for r in frame.itertuples(index=False)}
    return ids, preds

def topo_sort(ids, preds):
    indeg = {i: 0 for i in ids}
    for j in ids:
        indeg[j] = len(preds[j])
    q = deque([i for i in ids if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in ids:
            if u in preds[v]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
    if len(order) != len(ids):
        raise ValueError("Cycle or invalid dependencies found.")
    return order

def cpm(frame: pd.DataFrame):
    ids, preds = build_graph(frame)
    order = topo_sort(ids, preds)
    dmap = dict(zip(frame["ID"], frame["Duration"]))
    nmap = dict(zip(frame["ID"], frame["Task"]))

    ES, EF, DUR, NAME = {}, {}, {}, {}
    for i in order:
        NAME[i] = nmap[i]
        DUR[i]  = float(dmap[i])
        ES[i]   = max([EF[p] for p in preds[i]], default=0.0)
        EF[i]   = ES[i] + DUR[i]
    project_finish = max(EF.values()) if EF else 0.0

    # successors
    succs = {i: [] for i in ids}
    for j in ids:
        for p in preds[j]:
            succs[p].append(j)

    LS, LF = {}, {}
    sinks = [i for i in ids if not succs[i]]
    for i in sinks:
        LF[i] = project_finish
        LS[i] = LF[i] - DUR[i]
    for i in reversed(order):
        if i not in LF:
            nexts = succs[i]
            LF[i] = min(LS[s] for s in nexts) if nexts else project_finish
            LS[i] = LF[i] - DUR[i]

    Float = {i: round(LS[i] - ES[i], 6) for i in ids}
    critical = {i: abs(Float[i]) < 1e-9 for i in ids}

    sched = pd.DataFrame({
        "ID": ids,
        "Task": [NAME[i] for i in ids],
        "Duration": [DUR[i] for i in ids],
        "ES": [ES[i] for i in ids],
        "EF": [EF[i] for i in ids],
        "LS": [LS[i] for i in ids],
        "LF": [LF[i] for i in ids],
        "Float": [Float[i] for i in ids],
        "Critical": ["Critical" if critical[i] else "Non-Critical" for i in ids],
        "Predecessors": [", ".join(map(str, sorted(list(preds[i])))) for i in ids],
    }).sort_values(["ES","EF","ID"])
    return sched, project_finish

def gantt_figure(schedule_df: pd.DataFrame, title: str):
    import re
    FILL_RX = re.compile(r"\s*Fill\s*#\s*(\d+)\b.*", re.IGNORECASE)
    CAP_RX  = re.compile(r"\s*Cap\s*#\s*(\d+)\b.*",  re.IGNORECASE)

    # Build a tuple sort key. We want:
    #  - primarily by ES (earliest start)
    #  - within the same “family” number, Fill (rank 0) before Cap (rank 1)
    #  - then by EF and ID to keep ordering deterministic for ties
    def _key(rec):
        name = str(rec["Task"])
        es   = float(rec["ES"])
        ef   = float(rec["EF"])
        tid  = int(rec["ID"])
        m_fill = FILL_RX.fullmatch(name)
        m_cap  = CAP_RX.fullmatch(name)

        if m_fill:
            fam = int(m_fill.group(1))
            return (es, fam, 0, ef, tid)  # Fill first
        if m_cap:
            fam = int(m_cap.group(1))
            return (es, fam, 1, ef, tid)  # Cap second
        # Non Fill/Cap tasks stay ordered by ES → EF → ID after the families
        return (es, 10_000_000, 0, ef, tid)

    # Sort using Python's sorted on records; no DataFrame.sort_values
    recs_sorted = sorted(schedule_df.to_dict("records"), key=_key)
    ordered = pd.DataFrame(recs_sorted, columns=schedule_df.columns)

    fig = go.Figure()
    for row in ordered.itertuples(index=False):
        color = "#d62728" if row.Critical == "Critical" else "#1f77b4"
        fig.add_trace(go.Bar(
            x=[row.EF - row.ES],
            y=[f"{int(row.ID)}: {row.Task}"],
            base=[row.ES],
            orientation="h",
            marker=dict(color=color),
            hovertemplate=(
                f"Task: {row.Task}<br>"
                f"ES: {row.ES:.1f} h  EF: {row.EF:.1f} h<br>"
                f"LS: {row.LS:.1f} h  LF: {row.LF:.1f} h<br>"
                f"Float: {row.Float:.1f} h<extra></extra>"
            ),
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Hours",
        yaxis_title="Task",
        barmode="stack",
        height=max(800, 22 * len(ordered)),
        showlegend=False,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


# ---- Parallel Cap~Fill (cycle-safe union of predecessors) ----
_CAP_REGEX  = re.compile(r"\s*Cap\s*#\s*(\d+)\b.*", re.IGNORECASE)
_FILL_REGEX = re.compile(r"\s*Fill\s*#\s*(\d+)\b.*", re.IGNORECASE)

def _build_pred_succ(frame: pd.DataFrame):
    preds = {r.ID: set(parse_preds(r.Predecessors)) for r in frame.itertuples(index=False)}
    succ  = {i: [] for i in frame["ID"].tolist()}
    for j in frame["ID"].tolist():
        for p in preds[j]:
            succ[p].append(j)
    return preds, succ

def _descendants(node, succ):
    seen, q = set(), deque([node])
    while q:
        u = q.popleft()
        for v in succ.get(u, []):
            if v not in seen:
                seen.add(v); q.append(v)
    return seen

def build_parallel_cap_fill_same_start_union(frame: pd.DataFrame) -> pd.DataFrame:
    dfp = frame.copy()
    preds, succ = _build_pred_succ(dfp)
    name_map = dict(zip(dfp["ID"], dfp["Task"]))

    fill_by_n, cap_by_n = {}, {}
    for _id, name in name_map.items():
        s = str(name)
        m_fill = _FILL_REGEX.fullmatch(s)
        m_cap  = _CAP_REGEX.fullmatch(s)
        if m_fill:
            fill_by_n[int(m_fill.group(1))] = int(_id)
        if m_cap:
            cap_by_n[int(m_cap.group(1))]  = int(_id)

    common = sorted(set(fill_by_n) & set(cap_by_n))
    if not common:
        return dfp

    for n in common:
        fid = fill_by_n[n]; cid = cap_by_n[n]
        p_fill = set(preds.get(fid, set()))
        p_cap  = set(preds.get(cid, set()))
        p_fill.discard(fid); p_fill.discard(cid)
        p_cap.discard(fid);  p_cap.discard(cid)

        p_union = p_fill | p_cap
        bad_for_f = _descendants(fid, succ) | {fid, cid}
        bad_for_c = _descendants(cid, succ) | {fid, cid}
        p_union_safe = {p for p in p_union if p not in bad_for_f and p not in bad_for_c}

        preds[fid] = set(p_union_safe)
        preds[cid] = set(p_union_safe)

    # write back
    new_pred = []
    id_list = dfp["ID"].tolist()
    for _id in id_list:
        ps = sorted(preds.get(_id, set()))
        new_pred.append(", ".join(map(str, ps)))
    dfp["Predecessors"] = new_pred
    return dfp

# ----------------- UI: Upload + Delay controls -----------------
with st.sidebar:
    st.header("Upload schedule")
    uploaded = st.file_uploader("Excel or CSV (durations in HOURS)", type=["xlsx", "csv"])

    st.header("Delays (optional)")
    has_delays = st.toggle("Any delays?", value=False)
    selected_ids = []
    new_durations = {}

# ----------------- Run once a file is uploaded -----------------
if uploaded is None:
    st.info("Upload your schedule file to begin.")
    st.stop()

try:
    with st.spinner("Loading schedule…"):
        df = load_schedule_uploaded(uploaded)
        df = normalize_columns(df)
        # For delays UI: build a task lookup
        task_lookup = {int(r.ID): f"{int(r.ID)} — {r.Task}" for r in df.itertuples(index=False)}

    # Delays UI (only if toggled)
    df_working = df.copy()
    delayed_applied = False
    if has_delays:
        st.subheader("Delay details")
        # Multi-select by label but store IDs
        selected_labels = st.multiselect(
            "Select one or more tasks to update duration (hours):",
            options=[f"{tid}: {task_lookup[tid]}" for tid in sorted(task_lookup)],
        )
        selected_ids = []
        for lbl in selected_labels:
            tid = int(lbl.split(":")[0].strip())
            selected_ids.append(tid)

        cols = st.columns(2)
        for idx, tid in enumerate(selected_ids):
            with cols[idx % 2]:
                current = float(df_working.loc[df_working["ID"] == tid, "Duration"].iloc[0])
                nd = st.number_input(
                    f"New duration for Task {tid} ({task_lookup[tid].split('—',1)[-1].strip()})",
                    min_value=0.0, value=current, step=0.5, format="%.2f"
                )
                new_durations[tid] = nd

        if selected_ids:
            apply_btn = st.button("Apply delays")
            if apply_btn:
                for tid in selected_ids:
                    df_working.loc[df_working["ID"] == tid, "Duration"] = float(new_durations[tid])
                delayed_applied = True
                st.success("Delays applied.")

    # Compute Baseline
    with st.spinner("Computing Baseline CPM…"):
        baseline, base_finish = cpm(df)
    st.success(f"Baseline duration: {base_finish:.1f} hours")
    st.plotly_chart(gantt_figure(baseline, "Baseline Gantt (Critical in Red)"), use_container_width=True)

    # Compute Delayed (if applied)
    delayed = None
    delayed_finish = None
    if delayed_applied:
        with st.spinner("Computing Delayed CPM…"):
            delayed, delayed_finish = cpm(df_working)
        slip = delayed_finish - base_finish
        st.info(f"Delayed duration: {delayed_finish:.1f} hours (slip {slip:+.1f} h)")
        st.plotly_chart(gantt_figure(delayed, "Delayed Gantt (Critical in Red)"), use_container_width=True)

    # Parallel Cap~Fill (same start) from whichever is active (delayed or baseline)
    use_df = df_working if delayed_applied else df
    try:
        with st.spinner("Computing Parallel Cap~Fill what-if…"):
            df_parallel = build_parallel_cap_fill_same_start_union(use_df)
            parallel_sched, parallel_finish = cpm(df_parallel)
        msg = f"{parallel_finish - base_finish:+.1f} h vs Baseline"
        if delayed_finish is not None:
            msg += f", {parallel_finish - delayed_finish:+.1f} h vs Delayed"
        st.success(f"Parallel Cap~Fill duration: {parallel_finish:.1f} hours ({msg})")
        st.plotly_chart(
            gantt_figure(parallel_sched, "Parallel What-if: Cap starts with Fill (Critical in Red)"),
            use_container_width=True
        )
    except Exception as e:
        st.warning(f"Parallel Cap~Fill failed: {e}")

    # Optional: export HTMLs in working directory
    outdir = Path.cwd()
    def write_html(df_sched, title, fname):
        fig = gantt_figure(df_sched, title)
        fig.write_html(str(outdir / fname))
    write_html(baseline, "Baseline", "baseline_gantt.html")
    if delayed_applied and delayed is not None:
        write_html(delayed, "Delayed", "delayed_gantt.html")
    write_html(parallel_sched, "Parallel", "parallel_cap_fill_gantt.html")
    st.caption("HTML files also written in the current folder.")

except Exception as e:
    import traceback
    st.error(f"App error: {e}")
    st.code(traceback.format_exc())
