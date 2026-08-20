"""
VaR Monitor — All Commodities
Parametric 1-Day VaR at 99% confidence (rolling window: 20D / 60D / 120D)
Run: streamlit run var_monitor.py
"""

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ── Data paths ───────────────────────────────────────────────────────────────
_DB_DIR       = Path(__file__).resolve().parents[1] / "Database"
_POSITIONS_FILE = Path(__file__).resolve().parent / "saved_positions.json"

def _load_positions(comm_order):
    if _POSITIONS_FILE.exists():
        try:
            return json.loads(_POSITIONS_FILE.read_text())
        except Exception:
            pass
    return {c: 0 for c in comm_order}

def _save_positions(positions: dict):
    _POSITIONS_FILE.write_text(json.dumps(positions))

def _rx_load(comm: str) -> pd.DataFrame:
    """Read rollex parquet directly — no rollex_utils dependency."""
    alias = {"LRC": "RC"}
    c = alias.get(comm.upper(), comm.upper())
    df = pd.read_parquet(_DB_DIR / f"rollex_{c}.parquet")
    df.index.name = "Date"
    return df

# commodity code → parquet filename in VaR/Database/
_PARQUET_MAP = {
    "KC":  "kc_futures.parquet",  "LRC": "rc_futures.parquet",  "CC":  "cc_futures.parquet",
    "LCC": "lcc_futures.parquet", "SB":  "sb_futures.parquet",  "CT":  "ct_futures.parquet",
    "LSU": "lsu_futures.parquet",
}

def _load_front_price(comm: str) -> pd.DataFrame:
    """Return a Date-indexed DataFrame with settlement price and active contract name."""
    path = _DB_DIR / _PARQUET_MAP[comm]
    raw  = pd.read_parquet(path, columns=["Date", "FND", "settlement", "ice_symbol"])
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["FND"]  = pd.to_datetime(raw["FND"])
    raw = raw.dropna(subset=["settlement"])
    # Front contract on each date = earliest FND that is still >= that date
    active = (
        raw[raw["FND"] >= raw["Date"]]
        .sort_values(["Date", "FND"])
        .groupby("Date")[["settlement", "ice_symbol"]]
        .first()
        .rename(columns={"ice_symbol": "base_ric"})
    )
    return active  # DatetimeIndex → {settlement, base_ric}

st.set_page_config(page_title="VaR Monitor", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
  [data-testid="stAppViewContainer"],[data-testid="stMain"],.main{background:#fafafa!important;color:#1d1d1f!important}
  [data-testid="stHeader"]{background:transparent!important}
  .block-container{padding-top:2rem!important;padding-bottom:1.5rem;max-width:1440px}
  hr{border:none!important;border-top:1px solid #e8e8ed!important;margin:.4rem 0!important}
  [data-testid="stRadio"] label,[data-testid="stRadio"] label p,[data-testid="stRadio"] label div{font-size:.78rem!important;color:#1d1d1f!important}
  [data-testid="stExpander"]{border:1px solid #e8e8ed!important;border-radius:8px!important;background:#fff!important}
  h1,h2,h3{color:#1d1d1f!important;font-weight:500!important}
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
NAVY  = "#0a2463"
BLACK = "#1d1d1f"

LOT_SIZES = {"KC": 375, "LRC": 10, "CC": 10, "LCC": 10, "SB": 1120, "CT": 500, "LSU": 50}
COLORS    = {"KC": "#0a2463", "LRC": "#8b1a00", "CC": "#e8a020", "LCC": "#4a7fb5",
             "SB": "#1a6b1a", "CT": "#7b2d8b", "LSU": "#c0392b"}
NAMES     = {"KC": "Arabica", "LRC": "Robusta", "CC": "NYC Cocoa", "LCC": "London Cocoa",
             "SB": "Sugar #11", "CT": "Cotton", "LSU": "White Sugar"}

COMBINED = {
    "Coffee (KC+LRC)": ["KC",  "LRC"],
    "Cocoa (CC+LCC)":  ["CC",  "LCC"],
    "Sugar (SB+LSU)":  ["SB",  "LSU"],
}
COMBINED_COLORS = {
    "Coffee (KC+LRC)": "#2a4a7a",
    "Cocoa (CC+LCC)":  "#c87010",
    "Sugar (SB+LSU)":  "#2a8a2a",
}

CONF_Z  = 2.3263  # one-tailed 99% VaR (N^-1(0.99)); 2.5758 is two-tailed 99% and overstates
WINDOWS = {"20D": 20, "60D": 60, "120D": 120}
MONTHS  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

_D = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system,Helvetica Neue,sans-serif", color=BLACK, size=10),
)

def lbl(text):
    return (f"<div style='background:{NAVY};padding:5px 13px;border-radius:5px;"
            f"margin-bottom:8px'><span style='font-size:.78rem;font-weight:500;"
            f"letter-spacing:.07em;text-transform:uppercase;color:#dde4f0'>{text}</span></div>")

# ── Data loading & VaR computation ───────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all():
    data = {}
    for comm in LOT_SIZES:
        # ── Returns & vol: rollex continuous price ────────────────────────────
        rx = _rx_load(comm)[["rollex_px"]].reset_index()
        rx.columns = ["Date", "Close"]
        rx["Date"] = pd.to_datetime(rx["Date"])
        rx = rx.sort_values("Date").dropna(subset=["Close"]).reset_index(drop=True)
        full_idx = pd.bdate_range(rx["Date"].min(), rx["Date"].max())
        rx = rx.set_index("Date").reindex(full_idx).ffill()
        rx["log_ret"] = np.log(rx["Close"] / rx["Close"].shift(1))
        for w_name, w in WINDOWS.items():
            rx[f"vol_{w_name}"] = rx["log_ret"].rolling(w).std()

        # ── Price for VaR: active front contract settlement ───────────────────
        front = _load_front_price(comm)
        df = rx.join(front, how="left")
        df["settlement"] = df["settlement"].ffill()
        df["base_ric"]   = df["base_ric"].ffill()

        for w_name in WINDOWS:
            df[f"VaR_{w_name}"] = df["settlement"] * LOT_SIZES[comm] * df[f"vol_{w_name}"] * CONF_Z

        data[comm] = df.reset_index().rename(columns={"index": "Date"})
    return data

data = load_all()

# ── Helpers ───────────────────────────────────────────────────────────────────
INDIV_OPTIONS = {f"{comm} — {NAMES[comm]}": comm for comm in LOT_SIZES}
ALL_OPTIONS   = list(INDIV_OPTIONS.keys()) + list(COMBINED.keys())

def _var_series(label: str, var_col: str) -> pd.DataFrame:
    if label in INDIV_OPTIONS:
        comm = INDIV_OPTIONS[label]
        return data[comm][["Date", var_col, "base_ric"]].copy().rename(columns={var_col: "VaR", "base_ric": "contract"})
    else:
        comms  = COMBINED[label]
        frames = [data[c].set_index("Date")[var_col] for c in comms]
        combined = pd.concat(frames, axis=1).ffill()
        s = combined.sum(axis=1, min_count=len(comms)).reset_index()
        s.columns = ["Date", "VaR"]
        # For combined, show both active contracts e.g. "KCH6 + RCH6"
        contracts = pd.concat([data[c].set_index("Date")["base_ric"] for c in comms], axis=1).ffill()
        contracts.columns = comms
        s["contract"] = contracts.apply(lambda r: " + ".join(r.dropna().values), axis=1)
        return s

def _label_meta(label: str):
    if label in INDIV_OPTIONS:
        comm = INDIV_OPTIONS[label]
        return NAMES[comm], COLORS[comm]
    return label, COMBINED_COLORS[label]

# ── App header ────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='font-family:\"Playfair Display\",Georgia,serif;color:#0a2463;"
    "font-weight:400;letter-spacing:-.01em;margin-bottom:2px'>VaR Monitor</h2>",
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Global date bounds ────────────────────────────────────────────────────────
all_dates     = pd.concat([data[c]["Date"] for c in data]).dropna()
min_d         = all_dates.min().date()
max_d         = all_dates.max().date()
default_start = (all_dates.max() - pd.DateOffset(years=5)).date()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab2, tab1 = st.tabs(["Portfolio VaR — Monte Carlo", "Parametric VaR"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — VaR Monitor
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        f"<i style='font-size:.75rem;color:#888'>Data as of {max_d.strftime('%b %d, %Y')}</i>",
        unsafe_allow_html=True,
    )

    # ── Section 1 — Collapsible Filters + Line Chart ─────────────────────────
    with st.expander("Controls", expanded=True):
        c1, c2, c3 = st.columns([3, 4, 2])
        with c1:
            selected_labels = st.multiselect(
                "Commodities", ALL_OPTIONS,
                default=["KC — Arabica", "LRC — Robusta"],
                key="ms_main",
            )
        with c2:
            date_range = st.slider(
                "Date range", min_value=min_d, max_value=max_d,
                value=(default_start, max_d), key="sl_main",
            )
        with c3:
            window_label = st.radio(
                "VaR Window", list(WINDOWS.keys()), index=1,
                horizontal=True, key="radio_window",
            )

    var_col        = f"VaR_{window_label}"
    start_d, end_d = date_range

    st.markdown(lbl(f"1-Day VaR · 99% Confidence · {window_label} Rolling Window · Per Lot"), unsafe_allow_html=True)

    fig_line = go.Figure()
    for label in selected_labels:
        name, color = _label_meta(label)
        s = _var_series(label, var_col)
        s = s[(s["Date"].dt.date >= start_d) & (s["Date"].dt.date <= end_d)]
        fig_line.add_trace(go.Scatter(
            x=s["Date"], y=s["VaR"].round(0),
            name=name, mode="lines",
            line=dict(color=color, width=1.8),
            customdata=s["contract"],
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>VaR: $%{y:,.0f}<br>Contract: %{customdata}<extra>" + name + "</extra>",
        ))

    fig_line.update_layout(
        height=380,
        xaxis=dict(showgrid=False, tickfont=dict(size=9, color=BLACK)),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0",
                   tickfont=dict(size=9, color=BLACK), title="VaR (USD / lot)"),
        legend=dict(orientation="h", y=1.02, x=0,
                    font=dict(size=8, color=BLACK), bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(t=10, b=10, l=4, r=4), **_D,
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Section 2 — Vol Percentile Bar ───────────────────────────────────────
    st.markdown(lbl(f"Current Volatility Percentile vs Full History · {window_label}"), unsafe_allow_html=True)

    pct_rows = []
    for comm in LOT_SIZES:
        vol_col = f"vol_{window_label}"
        hist    = data[comm][vol_col].dropna()
        if hist.empty:
            continue
        cur_vol = hist.iloc[-1]
        pct     = float((hist < cur_vol).mean() * 100)
        cur_var = data[comm][var_col].dropna().iloc[-1]
        pct_rows.append({
            "Commodity":   NAMES[comm],
            "Percentile":  round(pct, 1),
            "Current VaR": f"${cur_var:,.0f}",
            "color":       COLORS[comm],
        })

    pct_df = pd.DataFrame(pct_rows).sort_values("Percentile", ascending=True)

    fig_pct = go.Figure(go.Bar(
        x=pct_df["Percentile"], y=pct_df["Commodity"],
        orientation="h",
        marker_color=pct_df["color"],
        text=pct_df.apply(lambda r: f"{r['Percentile']:.0f}th  |  {r['Current VaR']}", axis=1),
        textposition="outside",
        textfont=dict(size=9, color=BLACK),
    ))
    fig_pct.add_vline(x=50, line_dash="dot", line_color="#aaaaaa", line_width=1,
                      annotation_text="50th", annotation_font=dict(size=8, color="#aaaaaa"),
                      annotation_position="top")
    fig_pct.add_vline(x=80, line_dash="dot", line_color="#e07b39", line_width=1,
                      annotation_text="80th", annotation_font=dict(size=8, color="#e07b39"),
                      annotation_position="top")
    fig_pct.update_layout(
        height=300,
        xaxis=dict(range=[0, 120], showgrid=False,
                   tickfont=dict(size=9, color=BLACK), title="Percentile"),
        yaxis=dict(showgrid=False, tickfont=dict(size=9, color=BLACK)),
        margin=dict(t=10, b=10, l=4, r=120), **_D,
    )
    st.plotly_chart(fig_pct, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Section 3 — Heatmap (Years × Months) ────────────────────────────────
    hm_label = st.selectbox("Commodity for Heatmap", ALL_OPTIONS, index=0, key="hm_sel")
    st.markdown(lbl(f"Monthly Avg VaR Heatmap · {window_label} · {_label_meta(hm_label)[0]}"), unsafe_allow_html=True)

    hm_s = _var_series(hm_label, var_col)
    hm_s = hm_s[(hm_s["Date"].dt.date >= start_d) & (hm_s["Date"].dt.date <= end_d)].dropna(subset=["VaR"])
    hm_s["Year"]  = hm_s["Date"].dt.year
    hm_s["Month"] = hm_s["Date"].dt.month

    pivot = (
        hm_s.groupby(["Year", "Month"])["VaR"]
        .mean()
        .reset_index()
        .pivot(index="Year", columns="Month", values="VaR")
    )
    pivot.columns = [MONTHS[m - 1] for m in pivot.columns]
    pivot = pivot.sort_index(ascending=False)

    z         = pivot.values
    years     = [str(y) for y in pivot.index]
    months    = list(pivot.columns)
    text_vals = [[f"${v:,.0f}" if not pd.isna(v) else "" for v in row] for row in z]

    fig_hm = go.Figure(go.Heatmap(
        z=z, x=months, y=years,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=8, color=BLACK),
        colorscale=[
            [0.0, "#d4edda"],
            [0.4, "#fff3cd"],
            [0.7, "#f8d7a0"],
            [1.0, "#f5c6cb"],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="VaR (USD)", font=dict(size=9, color=BLACK)),
            tickfont=dict(size=8, color=BLACK),
            thickness=12, len=0.8,
        ),
        hoverongaps=False,
        hovertemplate="<b>%{y} %{x}</b><br>Avg VaR: $%{z:,.0f}<extra></extra>",
    ))
    fig_hm.update_layout(
        height=max(300, len(years) * 28),
        xaxis=dict(side="top", tickfont=dict(size=9, color=BLACK), showgrid=False),
        yaxis=dict(tickfont=dict(size=9, color=BLACK), showgrid=False),
        margin=dict(t=40, b=10, l=60, r=10), **_D,
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Portfolio VaR — Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        f"<i style='font-size:.75rem;color:#888'>Data as of {max_d.strftime('%b %d, %Y')}</i>",
        unsafe_allow_html=True,
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    cc1, cc2, cc3, cc4 = st.columns([2, 2, 2, 2])
    with cc1:
        mc_win  = st.radio("Calibration Window", list(WINDOWS.keys()), index=1,
                           horizontal=True, key="mc_win")
    with cc2:
        n_sims  = st.select_slider("Simulations",
                                   [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 500_000],
                                   value=10_000, key="mc_nsims")
    with cc3:
        mc_conf = st.radio("Confidence", ["95%", "99%"], index=1,
                           horizontal=True, key="mc_conf")
    with cc4:
        use_t  = st.toggle("Fat tails (t-dist)", value=True, key="mc_t")
        t_df_v = st.slider("Degrees of freedom", 3, 30, 6, key="mc_tdf") if use_t else None

    # ── Parameter guide ───────────────────────────────────────────────────────
    with st.expander("Parameter Guide", expanded=False):
        st.markdown("""
**Calibration Window** sets how many recent trading days are used to estimate volatility and correlations.
Use **20D** when you want the model to reflect current market conditions closely.
Use **60D** as the standard balanced view (roughly one quarter).
Use **120D** for a more conservative estimate that smooths over short calm periods and is less likely to understate risk.

**Simulations** controls how many scenarios are generated. More paths means a more stable VaR number with less sampling noise.
1,000 is fine for quick exploration, 10,000 is the working default, and 25,000 is recommended for any formal reporting or comparison.

**Confidence** sets the loss threshold. At 95% the VaR is the loss exceeded on roughly 1 in 20 days. At 99% it is 1 in 100 days.
99% is the standard for professional risk reporting.

**Fat Tails (t-dist)** switches the simulation from a normal distribution to a Student-t distribution, which assigns higher probability
to extreme moves. Commodity markets experience sharp dislocations more often than normal would predict, so this is recommended on by default.

**Degrees of Freedom** controls how fat the tails are. Lower means more extreme moves in the simulation.

| Degrees of Freedom | Tail behaviour | When to use |
|---|---|---|
| 3 to 4 | Very fat tails, frequent extreme moves | Stress testing, crisis scenarios |
| 5 to 7 | Moderate fat tails | Standard soft commodity conditions (default 6) |
| 10 to 15 | Mild fat tails | Calm, range-bound markets |
| Above 30 | Converges to normal distribution | Effectively the same as turning fat tails off |
""")

    # ── Position input table ──────────────────────────────────────────────────
    st.markdown(lbl("Book — Enter Positions"), unsafe_allow_html=True)

    comm_order = list(LOT_SIZES.keys())

    if "saved_positions" not in st.session_state:
        st.session_state["saved_positions"] = _load_positions(comm_order)

    # Harvest any edits from the previous render's diff state, then clear it.
    # This avoids Streamlit conflicting its own diff with the updated base DataFrame.
    if "mc_pos_editor" in st.session_state:
        changed = False
        for _row, _edits in st.session_state["mc_pos_editor"].get("edited_rows", {}).items():
            if "Position (lots)" in _edits:
                st.session_state["saved_positions"][comm_order[int(_row)]] = int(_edits["Position (lots)"])
                changed = True
        if changed:
            _save_positions(st.session_state["saved_positions"])
        del st.session_state["mc_pos_editor"]

    _pos_rows  = []
    for _c in comm_order:
        _last  = data[_c].dropna(subset=["settlement"]).iloc[-1]
        _price = float(_last["settlement"])
        _contr = str(_last.get("base_ric", ""))
        _pos_rows.append({
            "Code":            _c,
            "Name":            NAMES[_c],
            "Contract":        _contr,
            "Price":           _price,
            "$ / Lot":         round(_price * LOT_SIZES[_c], 0),
            "Position (lots)": st.session_state["saved_positions"].get(_c, 0),
        })

    _pos_default = pd.DataFrame(_pos_rows)
    _edited = st.data_editor(
        _pos_default,
        column_config={
            "Code":            st.column_config.TextColumn("Code",      disabled=True, width="small"),
            "Name":            st.column_config.TextColumn("Name",      disabled=True),
            "Contract":        st.column_config.TextColumn("Contract",  disabled=True, width="small"),
            "Price":           st.column_config.NumberColumn("Price",   disabled=True, format="%.2f"),
            "$ / Lot":         st.column_config.NumberColumn("$ / Lot", disabled=True, format="$%,.0f"),
            "Position (lots)": st.column_config.NumberColumn("Position (lots)", step=1,
                                                              min_value=-500_000, max_value=500_000),
        },
        hide_index=True,
        use_container_width=True,
        key="mc_pos_editor",
    )

    positions  = _edited["Position (lots)"].values.astype(float)
    prices_v   = _edited["Price"].values.astype(float)
    lot_arr    = np.array([LOT_SIZES[c] for c in comm_order], dtype=float)
    dollar_exp = positions * lot_arr * prices_v

    if np.all(positions == 0):
        st.info("Enter position sizes above to run the simulation.")
        st.stop()

    # ── Returns matrix & covariance ───────────────────────────────────────────
    w_mc     = WINDOWS[mc_win]
    ret_mx   = pd.concat(
        [data[c].set_index("Date")["log_ret"].rename(c) for c in comm_order], axis=1
    ).dropna()
    recent_r = ret_mx.tail(w_mc)
    cov_mx   = recent_r.cov().values
    corr_mx  = recent_r.corr()

    try:
        L = np.linalg.cholesky(cov_mx)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov_mx + np.eye(len(comm_order)) * 1e-10)

    # ── Simulate ──────────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    if use_t:
        Z = rng.standard_t(t_df_v, size=(n_sims, len(comm_order)))
        Z = Z / np.sqrt(t_df_v / (t_df_v - 2))
    else:
        Z = rng.standard_normal((n_sims, len(comm_order)))

    sim_ret = Z @ L.T
    sim_pnl = sim_ret @ dollar_exp

    # ── VaR / CVaR ────────────────────────────────────────────────────────────
    alpha     = 0.01 if mc_conf == "99%" else 0.05
    z_para    = 2.3263 if mc_conf == "99%" else 1.6449
    cutoff    = float(np.percentile(sim_pnl, alpha * 100))
    port_var  = max(-cutoff, 0.0)
    tail_mask = sim_pnl <= cutoff
    port_cvar = float(-sim_pnl[tail_mask].mean()) if tail_mask.any() else port_var

    indiv_var = np.array([
        abs(dollar_exp[i]) * float(data[c][f"vol_{mc_win}"].dropna().iloc[-1]) * z_para
        for i, c in enumerate(comm_order)
    ])
    sum_indiv   = float(indiv_var.sum())
    div_benefit = sum_indiv - port_var

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Portfolio VaR",        f"${port_var:,.0f}")
    k2.metric("CVaR / Exp Shortfall", f"${port_cvar:,.0f}")
    k3.metric("Sum Indiv VaRs",       f"${sum_indiv:,.0f}")
    k4.metric("Diversification",      f"${div_benefit:,.0f}",
              delta=f"{div_benefit / sum_indiv * 100:.1f}% reduction" if sum_indiv > 0 else None,
              delta_color="inverse")
    k5.metric("Gross $ Exposure",     f"${np.abs(dollar_exp).sum():,.0f}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── P&L Histogram  +  Correlation heatmap ────────────────────────────────
    h_col, c_col = st.columns([3, 2])

    with h_col:
        st.markdown(lbl(f"Simulated 1-Day P&L · {n_sims:,} paths · {mc_conf}"), unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=sim_pnl[~tail_mask], nbinsx=80,
            marker_color="#2a6496", opacity=0.55, name="Within VaR",
            hovertemplate="P&L: $%{x:,.0f}<extra>Within VaR</extra>",
        ))
        fig_hist.add_trace(go.Histogram(
            x=sim_pnl[tail_mask], nbinsx=30,
            marker_color="#c0392b", opacity=0.85, name=f"Tail (>{mc_conf})",
            hovertemplate="P&L: $%{x:,.0f}<extra>Tail loss</extra>",
        ))
        fig_hist.add_vline(x=cutoff,
                           line=dict(color="#c0392b", width=2, dash="dash"),
                           annotation_text=f"VaR  ${port_var:,.0f}",
                           annotation_font=dict(size=9, color="#c0392b"),
                           annotation_position="top right")
        fig_hist.add_vline(x=-port_cvar,
                           line=dict(color="#7b0000", width=1.5, dash="dot"),
                           annotation_text=f"CVaR  ${port_cvar:,.0f}",
                           annotation_font=dict(size=9, color="#7b0000"),
                           annotation_position="top left")
        fig_hist.update_layout(
            barmode="overlay", height=400,
            xaxis=dict(title="1-Day P&L (USD)", tickformat="$,.0f",
                       tickfont=dict(size=9, color=BLACK), showgrid=False),
            yaxis=dict(title="Frequency", tickfont=dict(size=9, color=BLACK),
                       showgrid=True, gridcolor="#f0f0f0"),
            legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=8)),
            margin=dict(t=30, b=60, l=4, r=4), **_D,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c_col:
        st.markdown(lbl(f"Return Correlation · {mc_win} window"), unsafe_allow_html=True)
        _corr_z   = corr_mx.values
        _clbls    = [NAMES[c] for c in comm_order]
        fig_corr  = go.Figure(go.Heatmap(
            z=_corr_z, x=_clbls, y=_clbls,
            text=[[f"{v:.2f}" for v in row] for row in _corr_z],
            texttemplate="%{text}", textfont=dict(size=9, color=BLACK),
            colorscale=[[0, "#c0392b"], [0.5, "#ffffff"], [1, "#2a6496"]],
            zmin=-1, zmax=1, showscale=True,
            colorbar=dict(thickness=10, len=0.8, tickfont=dict(size=8, color=BLACK)),
            hovertemplate="<b>%{y} × %{x}</b><br>ρ = %{z:.2f}<extra></extra>",
        ))
        fig_corr.update_layout(
            height=400,
            xaxis=dict(tickfont=dict(size=8, color=BLACK), tickangle=-30, showgrid=False),
            yaxis=dict(tickfont=dict(size=8, color=BLACK), showgrid=False),
            margin=dict(t=10, b=60, l=90, r=10), **_D,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Component VaR by commodity ────────────────────────────────────────────
    st.markdown(lbl("Component CVaR by Commodity"), unsafe_allow_html=True)
    st.caption("Average per-commodity loss in tail scenarios (portfolio P&L ≤ VaR cutoff). Sums to portfolio CVaR. Negative = hedge.")

    comm_pnl_mx = sim_ret * dollar_exp[np.newaxis, :]
    if tail_mask.any():
        comp_var_arr = np.where(positions != 0, -comm_pnl_mx[tail_mask].mean(axis=0), 0.0)
    else:
        comp_var_arr = np.zeros(len(comm_order))

    fig_comp = go.Figure(go.Bar(
        x=[f"{c}<br>{NAMES[c]}" for c in comm_order],
        y=comp_var_arr,
        marker_color=[COLORS[c] if positions[i] != 0 else "#cccccc"
                      for i, c in enumerate(comm_order)],
        text=[f"${v:,.0f}" if positions[i] != 0 else "—"
              for i, v in zip(range(len(comm_order)), comp_var_arr)],
        textposition="outside",
        textfont=dict(size=9, color=BLACK),
        hovertemplate="<b>%{x}</b><br>Component CVaR: $%{y:,.0f}<extra></extra>",
    ))
    fig_comp.add_hline(y=0, line=dict(color="#aaaaaa", width=1))
    fig_comp.update_layout(
        height=320,
        xaxis=dict(showgrid=False, tickfont=dict(size=9, color=BLACK)),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickformat="$,.0f",
                   tickfont=dict(size=9, color=BLACK), title="Component CVaR (USD)"),
        margin=dict(t=30, b=10, l=4, r=4), **_D,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Full breakdown table ──────────────────────────────────────────────────
    _tbl = []
    for i, c in enumerate(comm_order):
        _tbl.append({
            "Commodity":       f"{c} — {NAMES[c]}",
            "Position (lots)": int(positions[i]),
            "$ Exposure":      f"${dollar_exp[i]:,.0f}",
            "Indiv VaR":       f"${indiv_var[i]:,.0f}",
            "Component CVaR":  f"${comp_var_arr[i]:,.0f}" if positions[i] != 0 else "—",
            "% of CVaR":       f"{comp_var_arr[i] / port_cvar * 100:.1f}%"
                               if port_cvar > 0 and positions[i] != 0 else "—",
        })

    with st.expander("Full Position Breakdown", expanded=False):
        st.dataframe(pd.DataFrame(_tbl), use_container_width=True, hide_index=True)
