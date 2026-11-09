"""
Benford's Law + Two-Period KS Dashboard
---------------------------------------
Upload one or two CSV / Excel files (e.g. the same month in two different years),
filter the data, run a Benford analysis (1st digit and first-two digits), compare the
two periods with the Kolmogorov-Smirnov (KS) statistic (calculated with NumPy, the same
way as the "Data Viz and Two Groups" notebook), and download the results.
"""

import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Benford & KS Dashboard", page_icon="📊", layout="wide")

ALPHA_CONST = 1.36  # KS critical-value constant for alpha = 0.05 (same as the notebook)
DIGITS_1 = np.arange(1, 10)
DIGITS_2 = np.arange(10, 100)


# ----------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading file...")
def load_file(file_bytes: bytes, name: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    if name.lower().endswith(".csv"):
        return pd.read_csv(buf, low_memory=False)
    return pd.read_excel(buf)  # first sheet of .xlsx / .xlsm


# ----------------------------------------------------------------------------------------
# Benford + KS calculations (NumPy)
# ----------------------------------------------------------------------------------------
def clean_amounts(series: pd.Series) -> np.ndarray:
    """Absolute value, numeric only, drop zeros / NaN (Benford needs a non-zero first digit)."""
    v = np.abs(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float))
    return v[np.isfinite(v) & (v > 0)]


def leading_digits(values: np.ndarray, n_digits: int) -> np.ndarray:
    """First 1 (or 2) significant digits of each value, computed numerically."""
    exp = np.floor(np.log10(values))
    lead = values / np.power(10.0, exp)
    lead = np.where(lead >= 10, lead / 10, lead)  # guard against floating-point edge cases
    lead = np.where(lead < 1, lead * 10, lead)
    mult = 10 ** (n_digits - 1)
    d = np.floor(lead * mult + 1e-9).astype(int)
    return np.clip(d, mult, 10 * mult - 1)


def benford_table(values: np.ndarray, n_digits: int) -> pd.DataFrame:
    digits = DIGITS_1 if n_digits == 1 else DIGITS_2
    counts = np.bincount(leading_digits(values, n_digits), minlength=digits.max() + 1)[digits]
    total = counts.sum()
    actual = counts / total if total else np.zeros(len(digits))
    benford = np.log10(1 + 1 / digits)
    return pd.DataFrame(
        {"digit": digits, "count": counts, "actual": actual, "benford": benford, "difference": actual - benford}
    )


def ks_pvalue(lam: float) -> float:
    """Asymptotic Kolmogorov distribution p-value (NumPy only)."""
    if lam <= 0:
        return 1.0
    k = np.arange(1, 101)
    p = 2 * np.sum((-1.0) ** (k - 1) * np.exp(-2 * k**2 * lam**2))
    return float(min(max(p, 0.0), 1.0))


def ks_vs_benford(table: pd.DataFrame) -> dict:
    """One-sample KS: max |cumulative actual - cumulative Benford|, critical value 1.36/sqrt(n)."""
    n = int(table["count"].sum())
    if n == 0:
        return dict(n=0, ks=np.nan, critical=np.nan, p_value=np.nan, reject=False)
    ks = float(np.max(np.abs(np.cumsum(table["actual"]) - np.cumsum(table["benford"]))))
    crit = ALPHA_CONST / np.sqrt(n)
    return dict(n=n, ks=ks, critical=float(crit), p_value=ks_pvalue(ks * np.sqrt(n)), reject=ks > crit)


def ks_two_sample_values(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sample KS on the raw amounts (max gap between the two empirical CDFs)."""
    a, b = np.sort(a), np.sort(b)
    n, m = len(a), len(b)
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / n
    cdf_b = np.searchsorted(b, grid, side="right") / m
    ks = float(np.max(np.abs(cdf_a - cdf_b)))
    crit = ALPHA_CONST * np.sqrt((n + m) / (n * m))
    p = ks_pvalue(ks * np.sqrt(n * m / (n + m)))
    return dict(n1=n, n2=m, ks=ks, critical=float(crit), p_value=p, reject=ks > crit)


def ks_two_sample_digits(t1: pd.DataFrame, t2: pd.DataFrame) -> dict:
    """Two-sample KS on the first-digit / first-two-digit distributions of the two periods."""
    n, m = int(t1["count"].sum()), int(t2["count"].sum())
    ks = float(np.max(np.abs(np.cumsum(t1["actual"]) - np.cumsum(t2["actual"]))))
    crit = ALPHA_CONST * np.sqrt((n + m) / (n * m))
    p = ks_pvalue(ks * np.sqrt(n * m / (n + m)))
    return dict(n1=n, n2=m, ks=ks, critical=float(crit), p_value=p, reject=ks > crit)


def summary_stats(values: np.ndarray) -> pd.DataFrame:
    return pd.Series(values).describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).to_frame("amount")


# ----------------------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------------------
def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def benford_chart(table: pd.DataFrame, title: str, n_digits: int):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(table["digit"], table["actual"], color="steelblue", label="Actual", width=0.8)
    ax.plot(table["digit"], table["benford"], color="crimson", marker="o", markersize=3, label="Benford")
    ax.set_title(title)
    ax.set_xlabel("First digit" if n_digits == 1 else "First two digits")
    ax.set_ylabel("Proportion of transactions")
    if n_digits == 2:
        ax.set_xticks(range(10, 100, 5))
    else:
        ax.set_xticks(DIGITS_1)
    ax.legend()
    fig.tight_layout()
    return fig


def compare_chart(t1, t2, label1, label2, title, n_digits):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t1["digit"], t1["actual"], marker="o", markersize=3, label=label1, color="steelblue")
    ax.plot(t2["digit"], t2["actual"], marker="s", markersize=3, label=label2, color="darkorange")
    ax.plot(t1["digit"], t1["benford"], linestyle="--", color="gray", label="Benford")
    ax.set_title(title)
    ax.set_xlabel("First digit" if n_digits == 1 else "First two digits")
    ax.set_ylabel("Proportion of transactions")
    ax.legend()
    fig.tight_layout()
    return fig


def ecdf_chart(a, b, label1, label2):
    q = np.linspace(0, 1, 500)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(np.quantile(a, q), q, label=label1, color="steelblue")
    ax.plot(np.quantile(b, q), q, label=label2, color="darkorange")
    ax.set_xscale("log")
    ax.set_title("Empirical CDF of payment amounts (absolute value, log scale)")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Cumulative proportion")
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------------------
# Sidebar: upload + filters
# ----------------------------------------------------------------------------------------
st.title("📊 Benford's Law & Two-Period KS Dashboard")
st.caption(
    "Upload a CSV or Excel file (and optionally a second file for comparison), filter the data, "
    "test it against Benford's Law, and download the results."
)

with st.sidebar:
    st.header("1. Upload data")
    file1 = st.file_uploader("Period 1 file (CSV or Excel)", type=["csv", "xlsx", "xlsm"], key="f1")
    label1 = st.text_input("Period 1 label", "Period 1")
    file2 = st.file_uploader("Period 2 file (optional, for comparison)", type=["csv", "xlsx", "xlsm"], key="f2")
    label2 = st.text_input("Period 2 label", "Period 2")

if file1 is None:
    st.info("👈 Upload a CSV or Excel file in the sidebar to get started.")
    st.stop()

df1 = load_file(file1.getvalue(), file1.name)
df2 = load_file(file2.getvalue(), file2.name) if file2 is not None else None

all_cols = list(df1.columns)
numeric_guess = [c for c in all_cols if pd.api.types.is_numeric_dtype(df1[c])]
default_amt = "PAYMENT_AMOUNT" if "PAYMENT_AMOUNT" in all_cols else (numeric_guess[0] if numeric_guess else all_cols[0])

with st.sidebar:
    st.header("2. Choose amount column")
    amount_col = st.selectbox("Column to analyze", all_cols, index=all_cols.index(default_amt))
    if df2 is not None and amount_col not in df2.columns:
        st.error(f"Column '{amount_col}' is not in the Period 2 file.")
        st.stop()

    st.header("3. Filters")
    st.caption("Use any of these. Filters apply to both periods (when the column exists in both).")

    # Filter 1: amount range
    st.subheader("Filter 1 – Amount range")
    min_amt = st.number_input("Minimum amount (absolute value)", min_value=0.0, value=0.0)
    max_amt = st.number_input("Maximum amount (0 = no limit)", min_value=0.0, value=0.0)

    # Filter 2: category
    st.subheader("Filter 2 – Category")
    cat_col = st.selectbox("Category column", ["(none)"] + all_cols, key="cat_col")
    cat_vals = []
    if cat_col != "(none)":
        options = df1[cat_col].astype(str).value_counts().head(500).index.tolist()
        cat_vals = st.multiselect("Keep only these values (top 500 shown)", options)

    # Filter 3: text search
    st.subheader("Filter 3 – Text contains")
    txt_col = st.selectbox("Text column", ["(none)"] + all_cols, key="txt_col")
    txt_val = st.text_input("Contains (not case-sensitive)") if txt_col != "(none)" else ""

    # Filter 4: date range
    st.subheader("Filter 4 – Date range")
    date_guess = [c for c in all_cols if "date" in str(c).lower() or pd.api.types.is_datetime64_any_dtype(df1[c])]
    date_col = st.selectbox("Date column", ["(none)"] + date_guess + [c for c in all_cols if c not in date_guess], key="date_col")
    date_range = None
    if date_col != "(none)":
        parsed = pd.to_datetime(df1[date_col], errors="coerce")
        if parsed.notna().any():
            lo, hi = parsed.min().date(), parsed.max().date()
            date_range = st.date_input("Keep dates between", value=(lo, hi), min_value=lo, max_value=hi)
        else:
            st.warning("That column could not be read as dates.")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    amt = pd.to_numeric(df[amount_col], errors="coerce").abs()
    if min_amt > 0:
        mask &= amt >= min_amt
    if max_amt > 0:
        mask &= amt <= max_amt
    if cat_col != "(none)" and cat_vals and cat_col in df.columns:
        mask &= df[cat_col].astype(str).isin(cat_vals)
    if txt_col != "(none)" and txt_val and txt_col in df.columns:
        mask &= df[txt_col].astype(str).str.contains(txt_val, case=False, na=False, regex=False)
    if date_col != "(none)" and date_range is not None and len(date_range) == 2 and date_col in df.columns:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        mask &= (dt >= start) & (dt < end)
    return df[mask]


f1 = apply_filters(df1)
f2 = apply_filters(df2) if df2 is not None else None
vals1 = clean_amounts(f1[amount_col])
vals2 = clean_amounts(f2[amount_col]) if f2 is not None else None

if len(vals1) == 0:
    st.error("No usable (non-zero numeric) amounts in Period 1 after filtering. Check your column and filters.")
    st.stop()
if f2 is not None and len(vals2) == 0:
    st.error("No usable (non-zero numeric) amounts in Period 2 after filtering. Check your column and filters.")
    st.stop()

# ----------------------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------------------
b1_p1, b2_p1 = benford_table(vals1, 1), benford_table(vals1, 2)
ks1_p1, ks2_p1 = ks_vs_benford(b1_p1), ks_vs_benford(b2_p1)
stats1 = summary_stats(vals1)

if vals2 is not None:
    b1_p2, b2_p2 = benford_table(vals2, 1), benford_table(vals2, 2)
    ks1_p2, ks2_p2 = ks_vs_benford(b1_p2), ks_vs_benford(b2_p2)
    stats2 = summary_stats(vals2)
    ks_amounts = ks_two_sample_values(vals1, vals2)
    ks_d1 = ks_two_sample_digits(b1_p1, b1_p2)
    ks_d2 = ks_two_sample_digits(b2_p1, b2_p2)

# Result tables for display / download
summary_df = stats1.rename(columns={"amount": label1})
if vals2 is not None:
    summary_df[label2] = stats2["amount"]

rows = [
    {"test": "1st digit vs Benford", "period": label1, **ks1_p1},
    {"test": "First 2 digits vs Benford", "period": label1, **ks2_p1},
]
if vals2 is not None:
    rows += [
        {"test": "1st digit vs Benford", "period": label2, **ks1_p2},
        {"test": "First 2 digits vs Benford", "period": label2, **ks2_p2},
    ]
benford_results = pd.DataFrame(rows)[["test", "period", "n", "ks", "critical", "p_value", "reject"]]
benford_results.columns = ["Test", "Period", "N", "KS statistic", "Critical value (5%)", "p-value", "Reject Benford?"]

comparison_results = None
if vals2 is not None:
    comparison_results = pd.DataFrame(
        [
            {"Test": "Amounts (two-sample KS)", "N1": ks_amounts["n1"], "N2": ks_amounts["n2"], "KS statistic": ks_amounts["ks"],
             "Critical value (5%)": ks_amounts["critical"], "p-value": ks_amounts["p_value"], "Periods differ?": ks_amounts["reject"]},
            {"Test": "1st-digit distribution (two-sample KS)", "N1": ks_d1["n1"], "N2": ks_d1["n2"], "KS statistic": ks_d1["ks"],
             "Critical value (5%)": ks_d1["critical"], "p-value": ks_d1["p_value"], "Periods differ?": ks_d1["reject"]},
            {"Test": "First-2-digit distribution (two-sample KS)", "N1": ks_d2["n1"], "N2": ks_d2["n2"], "KS statistic": ks_d2["ks"],
             "Critical value (5%)": ks_d2["critical"], "p-value": ks_d2["p_value"], "Periods differ?": ks_d2["reject"]},
        ]
    )

# Charts
chart_figs = {
    f"benford_1digit_{label1}": benford_chart(b1_p1, f"{label1}: first digit vs Benford", 1),
    f"benford_2digit_{label1}": benford_chart(b2_p1, f"{label1}: first two digits vs Benford", 2),
}
if vals2 is not None:
    chart_figs[f"benford_1digit_{label2}"] = benford_chart(b1_p2, f"{label2}: first digit vs Benford", 1)
    chart_figs[f"benford_2digit_{label2}"] = benford_chart(b2_p2, f"{label2}: first two digits vs Benford", 2)
    chart_figs["compare_1digit"] = compare_chart(b1_p1, b1_p2, label1, label2, "First-digit distribution: two periods", 1)
    chart_figs["compare_2digit"] = compare_chart(b2_p1, b2_p2, label1, label2, "First-two-digit distribution: two periods", 2)
    chart_figs["ecdf_amounts"] = ecdf_chart(vals1, vals2, label1, label2)


def verdict(reject: bool, reject_text: str, keep_text: str) -> str:
    return f"❌ {reject_text}" if reject else f"✅ {keep_text}"


# ----------------------------------------------------------------------------------------
# Report text (for download / sharing)
# ----------------------------------------------------------------------------------------
def build_report() -> str:
    lines = [
        "# Benford & KS Analysis Report",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"Amount column: {amount_col}",
        f"{label1}: {len(df1):,} rows loaded, {len(f1):,} after filters, {len(vals1):,} non-zero amounts analyzed",
    ]
    if vals2 is not None:
        lines.append(f"{label2}: {len(df2):,} rows loaded, {len(f2):,} after filters, {len(vals2):,} non-zero amounts analyzed")
    lines += ["", "## Filters applied", f"- Amount range: min {min_amt:,.2f}, max {'none' if max_amt == 0 else f'{max_amt:,.2f}'}"]
    lines.append(f"- Category: {cat_col} in {cat_vals}" if cat_col != "(none)" and cat_vals else "- Category: none")
    lines.append(f"- Text contains: {txt_col} contains '{txt_val}'" if txt_col != "(none)" and txt_val else "- Text: none")
    lines.append(f"- Date range: {date_col} {date_range}" if date_col != "(none)" and date_range else "- Date range: none")
    lines += ["", "## Conformity to Benford's Law (one-sample KS, alpha = 0.05)", benford_results.to_markdown(index=False)
              if hasattr(benford_results, "to_markdown") and _has_tabulate() else benford_results.to_string(index=False)]
    if comparison_results is not None:
        lines += ["", "## Period comparison (two-sample KS, alpha = 0.05)",
                  comparison_results.to_markdown(index=False) if _has_tabulate() else comparison_results.to_string(index=False)]
    lines += ["", "Note: with very large samples, even tiny differences are statistically significant. "
              "Look at the size of the KS statistic and the charts, not just the pass/fail result."]
    return "\n".join(lines)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def build_excel() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        benford_results.to_excel(xw, sheet_name="Benford_KS_results", index=False)
        if comparison_results is not None:
            comparison_results.to_excel(xw, sheet_name="Period_comparison", index=False)
        stats1.to_excel(xw, sheet_name=f"Summary_{label1}"[:31])
        b1_p1.to_excel(xw, sheet_name=f"1digit_{label1}"[:31], index=False)
        b2_p1.to_excel(xw, sheet_name=f"2digit_{label1}"[:31], index=False)
        if vals2 is not None:
            stats2.to_excel(xw, sheet_name=f"Summary_{label2}"[:31])
            b1_p2.to_excel(xw, sheet_name=f"1digit_{label2}"[:31], index=False)
            b2_p2.to_excel(xw, sheet_name=f"2digit_{label2}"[:31], index=False)
    return buf.getvalue()


# ----------------------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric(f"{label1} rows (after filters)", f"{len(f1):,}", f"of {len(df1):,}", delta_color="off")
if f2 is not None:
    c2.metric(f"{label2} rows (after filters)", f"{len(f2):,}", f"of {len(df2):,}", delta_color="off")
c3.metric("Amount column", amount_col)

tab_data, tab_ben1, tab_ben2, tab_cmp, tab_dl = st.tabs(
    ["Data & summary stats", "Benford: 1st digit", "Benford: first 2 digits", "Compare periods (KS)", "Download & share"]
)

with tab_data:
    st.subheader(f"{label1}: filtered data preview")
    st.dataframe(f1.head(200))
    st.subheader("Summary statistics (absolute, non-zero amounts)")
    st.dataframe(summary_df.style.format("{:,.2f}"))

with tab_ben1:
    st.pyplot(chart_figs[f"benford_1digit_{label1}"])
    if vals2 is not None:
        st.pyplot(chart_figs[f"benford_1digit_{label2}"])
    st.subheader("Does the data follow Benford's Law?")
    sub = benford_results[benford_results["Test"] == "1st digit vs Benford"]
    st.dataframe(sub)
    for _, r in sub.iterrows():
        st.write(f"**{r['Period']}**: " + verdict(bool(r["Reject Benford?"]),
                 "KS statistic exceeds the critical value, so the first digits do NOT conform to Benford's Law (5% level).",
                 "KS statistic is below the critical value, so we cannot reject conformity with Benford's Law (5% level)."))
    st.subheader("Digit table")
    st.dataframe(b1_p1.style.format({"actual": "{:.4f}", "benford": "{:.4f}", "difference": "{:+.4f}"}))

with tab_ben2:
    st.pyplot(chart_figs[f"benford_2digit_{label1}"])
    if vals2 is not None:
        st.pyplot(chart_figs[f"benford_2digit_{label2}"])
    st.subheader("Does the data follow Benford's Law?")
    sub = benford_results[benford_results["Test"] == "First 2 digits vs Benford"]
    st.dataframe(sub)
    for _, r in sub.iterrows():
        st.write(f"**{r['Period']}**: " + verdict(bool(r["Reject Benford?"]),
                 "KS statistic exceeds the critical value, so the first two digits do NOT conform to Benford's Law (5% level).",
                 "KS statistic is below the critical value, so we cannot reject conformity with Benford's Law (5% level)."))
    st.subheader("Digit table")
    st.dataframe(b2_p1.style.format({"actual": "{:.4f}", "benford": "{:.4f}", "difference": "{:+.4f}"}))

with tab_cmp:
    if vals2 is None:
        st.info("Upload a second file in the sidebar to compare two time periods.")
    else:
        st.pyplot(chart_figs["compare_1digit"])
        st.pyplot(chart_figs["compare_2digit"])
        st.pyplot(chart_figs["ecdf_amounts"])
        st.subheader("Are the two periods statistically different?")
        st.dataframe(comparison_results)
        st.write(verdict(ks_amounts["reject"],
                 f"The amount distributions of {label1} and {label2} are statistically different (KS = {ks_amounts['ks']:.4f} > critical {ks_amounts['critical']:.4f}).",
                 f"No significant difference between the amount distributions (KS = {ks_amounts['ks']:.4f} ≤ critical {ks_amounts['critical']:.4f})."))
        st.caption("With millions of rows, the critical value becomes tiny, so almost any difference is 'significant'. "
                   "Pay attention to how large the KS statistic is, not just whether it passes.")

with tab_dl:
    st.subheader("Download results")
    st.write("Download the results to share with others, or share the link to this app.")
    d1, d2, d3 = st.columns(3)
    d1.download_button("📥 Excel workbook (all tables)", build_excel(), "benford_ks_results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("📥 Report (Markdown)", build_report(), "benford_ks_report.md", "text/markdown")
    d3.download_button("📥 Summary stats (CSV)", summary_df.to_csv(), "summary_stats.csv", "text/csv")

    e1, e2 = st.columns(2)
    e1.download_button("📥 Benford 1st-digit table (CSV)", b1_p1.to_csv(index=False), f"benford_1digit_{label1}.csv", "text/csv")
    e2.download_button("📥 Benford 2-digit table (CSV)", b2_p1.to_csv(index=False), f"benford_2digit_{label1}.csv", "text/csv")

    st.subheader("Download charts (PNG)")
    for name, fig in chart_figs.items():
        st.download_button(f"📥 {name}.png", fig_to_png(fig), f"{name}.png", "image/png", key=f"png_{name}")

plt.close("all")
