import pandas as pd
from matplotlib import cm
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="System Observation Boundary Map", layout="wide")

st.title("System Observation Boundary Map")
st.caption("This area indicates time outside the system’s observation.")

# ---- Settings
default_path = r"data\processed\windows_2026-01-20_strict_with_context.csv"
csv_path = st.sidebar.text_input("CSV path", default_path)

# 어떤 컬럼을 관측 기준으로 쓸지 선택 (네 파일에 맞게 자동으로 찾음)
candidate_flags = ["is_observed_strict", "is_observed", "is_outside_observation"]
flag_col = st.sidebar.selectbox("Observation flag column", candidate_flags, index=0)

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    st.error(f"CSV를 읽을 수 없어요: {e}")
    st.stop()

# ---- Validate required columns
required_cols = {"window_start", "window_end"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"필수 컬럼이 없어요: {missing}. (필요: window_start, window_end)")
    st.stop()

if flag_col not in df.columns:
    st.error(f"선택한 플래그 컬럼 '{flag_col}' 이 CSV에 없어요. 사이드바에서 다른 컬럼을 골라줘.")
    st.stop()

# ---- Parse times
df = df.copy()
df["window_start"] = pd.to_datetime(df["window_start"], errors="coerce", utc=True)
df["window_end"] = pd.to_datetime(df["window_end"], errors="coerce", utc=True)

df = df.dropna(subset=["window_start", "window_end"])
df = df.sort_values("window_start")

# ---- If flag_col is "is_outside_observation", invert to observed flag
if flag_col == "is_outside_observation":
    df["_observed"] = ~df[flag_col].astype(bool)
else:
    df["_observed"] = df[flag_col].astype(bool)

# ---- Summary
total = len(df)
observed_n = int(df["_observed"].sum())
outside_n = total - observed_n

st.sidebar.markdown("### Summary")
st.sidebar.write(f"Rows: {total}")
st.sidebar.write(f"Observed: {observed_n}")
st.sidebar.write(f"Outside: {outside_n}")

# ---- Build a single-track timeline plot
t0 = df["window_start"].min()
df["x0"] = (df["window_start"] - t0).dt.total_seconds()
df["x1"] = (df["window_end"] - t0).dt.total_seconds()
df["w"] = (df["x1"] - df["x0"]).clip(lower=0)

# Plot
fig, ax = plt.subplots(figsize=(18, 2.8))

# Observed bars
obs = df[df["_observed"]]
for _, r in obs.iterrows():
    ax.broken_barh([(r["x0"], r["w"])], (0, 10))

# Outside bars (hatched)
out = df[~df["_observed"]]
for _, r in out.iterrows():
    ax.broken_barh([(r["x0"], r["w"])], (0, 10), hatch="////", alpha=0.5)

ax.set_ylim(0, 10)
ax.set_yticks([])
ax.set_xlabel(f"seconds since {t0.isoformat()}")
ax.set_title("Observed vs Outside observation")
st.pyplot(fig, clear_figure=True)


# ===== Window context track (2nd row) =====
context_cols = [c for c in ["process", "title", "app", "window_name"] if c in df.columns]
context_col = st.sidebar.selectbox(
    "Window context column (for 2nd track)",
    ["(none)"] + context_cols,
    index=0 if not context_cols else 1
)

if context_col != "(none)":
    st.subheader("Window Context Track")
    st.caption("Foreground window context only. No behavioural inference.")

    ctx = df[context_col].fillna("unknown").astype(str)

    # 너무 길면 보기 힘드니 title은 줄여서 표시 (선택)
    if context_col == "title":
        ctx = ctx.str.slice(0, 60)

    uniq = ctx.unique().tolist()
    colour_map = {u: cm.tab20(i % 20) for i, u in enumerate(uniq)}

    fig2, ax2 = plt.subplots(figsize=(18, 2.4))
    for x0, w, u in zip(df["x0"], df["w"], ctx):
        ax2.broken_barh([(x0, w)], (0, 10), facecolors=colour_map.get(u, "grey"))

    ax2.set_ylim(0, 10)
    ax2.set_yticks([])
    ax2.set_xlabel(f"seconds since {t0.isoformat()}")
    ax2.set_title(f"Active window segments by {context_col}")
    st.pyplot(fig2, clear_figure=True)

    with st.expander("Context legend (top values)"):
        # 너무 많을 수 있으니 상위 30개만 보여줌
        counts = ctx.value_counts().head(30).reset_index()
        counts.columns = [context_col, "segments_count"]
        st.dataframe(counts)
else:
    st.info("No window context selected (or not available in CSV).")



# Optional table view
with st.expander("Data preview"):
    show_cols = [c for c in ["window_start", "window_end", flag_col, "event_total", "click_count", "key_count", "scroll_count", "move_count"] if c in df.columns]
    st.dataframe(df[show_cols].head(200))
