"""
Stopwatch Repeatability Tester
--------------------------------
Visual cues fire at slightly varying intervals. The user operates a physical
stopwatch and notes the reading at each MARK. Results show accuracy and
consistency (repeatability).
"""

import random
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Stopwatch Repeatability Tester",
    layout="centered",
    page_icon="⏱️",
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_time(val: str) -> float | None:
    """Accept M:SS.xx or decimal seconds."""
    val = val.strip()
    if not val:
        return None
    try:
        if ":" in val:
            parts = val.split(":")
            return float(parts[0]) * 60 + float(parts[1])
        return float(val)
    except ValueError:
        return None


def fmt_elapsed(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:05.2f}"


def reset_state():
    keys = [
        "phase", "cue_intervals", "cumulative_targets", "cue_actual_elapsed",
        "cue_active", "cue_active_time", "next_cue_time", "test_start_time",
        "entered_times", "countdown_start", "go_time", "n_cycles",
        "base_interval", "variation",
    ]
    for k in keys:
        st.session_state.pop(k, None)


# ── State init ────────────────────────────────────────────────────────────────

if "phase" not in st.session_state:
    st.session_state.phase = "setup"

phase = st.session_state.phase

# ── SETUP ─────────────────────────────────────────────────────────────────────
if phase == "setup":
    st.title("⏱️ Stopwatch Repeatability Tester")
    st.markdown("""
Measures how accurately and consistently you can read a stopwatch at live
visual cues — useful practice before any precision timed event.

**Procedure**
1. Configure parameters and press **Start Test**
2. Have your physical stopwatch in hand — **do not start it yet**
3. On the **GO!** flash, start your stopwatch
4. At each **MARK!** flash, note the stopwatch reading (keep the watch running)
5. After all marks, enter the readings you noted
6. Review your accuracy and repeatability
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_cycles = st.number_input("Number of marks", min_value=3, max_value=30, value=10, step=1)
    with col2:
        base_interval = st.number_input("Base interval (s)", min_value=5, max_value=120, value=15, step=5)
    with col3:
        variation = st.number_input("Variation ± (s)", min_value=1, max_value=20, value=4, step=1)

    approx_s = int(n_cycles * base_interval)
    st.info(f"Approximate test duration: **{approx_s // 60}m {approx_s % 60}s**")

    if st.button("▶  Start Test", type="primary", use_container_width=True):
        intervals = [base_interval + random.uniform(-variation, variation) for _ in range(n_cycles)]
        cumulative_targets = [sum(intervals[: i + 1]) for i in range(n_cycles)]

        st.session_state.n_cycles = n_cycles
        st.session_state.base_interval = base_interval
        st.session_state.variation = variation
        st.session_state.cue_intervals = intervals
        st.session_state.cumulative_targets = cumulative_targets
        st.session_state.cue_actual_elapsed = []
        st.session_state.cue_active = False
        st.session_state.phase = "countdown"
        st.session_state.countdown_start = time.time()
        st.rerun()

# ── COUNTDOWN ─────────────────────────────────────────────────────────────────
elif phase == "countdown":
    elapsed = time.time() - st.session_state.countdown_start
    remaining = 3.0 - elapsed

    if remaining > 0:
        digit = int(remaining) + 1
        st.markdown(
            f"<div style='text-align:center;font-size:180px;font-weight:bold;color:#e74c3c;"
            f"line-height:1;'>{digit}</div>"
            f"<div style='text-align:center;font-size:22px;color:#7f8c8d;margin-top:8px;'>"
            f"Get your stopwatch ready — start it on <strong>GO!</strong></div>",
            unsafe_allow_html=True,
        )
        time.sleep(0.1)
        st.rerun()
    else:
        st.session_state.phase = "go"
        st.session_state.go_time = time.time()
        st.rerun()

# ── GO ────────────────────────────────────────────────────────────────────────
elif phase == "go":
    st.markdown(
        "<div style='text-align:center;font-size:180px;font-weight:bold;color:white;"
        "background:#27ae60;border-radius:24px;padding:0 20px;line-height:1.1;'>GO!</div>"
        "<div style='text-align:center;font-size:24px;margin-top:16px;'>"
        "⬆️ Start your stopwatch <strong>NOW</strong></div>",
        unsafe_allow_html=True,
    )
    if time.time() - st.session_state.go_time >= 1.2:
        st.session_state.test_start_time = st.session_state.go_time
        st.session_state.next_cue_time = (
            st.session_state.go_time + st.session_state.cumulative_targets[0]
        )
        st.session_state.phase = "running"
        st.rerun()
    else:
        time.sleep(0.1)
        st.rerun()

# ── RUNNING ───────────────────────────────────────────────────────────────────
elif phase == "running":
    now = time.time()
    test_start = st.session_state.test_start_time
    n_done = len(st.session_state.cue_actual_elapsed)
    n_total = st.session_state.n_cycles
    cumulative_targets = st.session_state.cumulative_targets

    if st.session_state.cue_active:
        # ── Show MARK flash — sleep the full duration, then transition in one rerun ──
        st.markdown(
            f"<div style='text-align:center;font-size:180px;font-weight:bold;color:white;"
            f"background:#c0392b;border-radius:24px;padding:0 20px;line-height:1.1;'>MARK!</div>"
            f"<div style='text-align:center;font-size:22px;margin-top:16px;'>"
            f"Mark <strong>{n_done}</strong> of {n_total}</div>",
            unsafe_allow_html=True,
        )
        time.sleep(0.8)
        st.session_state.cue_active = False
        if n_done >= n_total:
            st.session_state.phase = "entering"
        st.rerun()

    elif now >= st.session_state.next_cue_time:
        # ── Fire cue ──
        st.session_state.cue_actual_elapsed.append(now - test_start)
        new_n_done = n_done + 1
        st.session_state.cue_active = True
        st.session_state.cue_active_time = now
        if new_n_done < n_total:
            st.session_state.next_cue_time = test_start + cumulative_targets[new_n_done]
        st.rerun()

    else:
        # ── Waiting — sleep until just before the next cue, one rerun per interval ──
        time_to_next = st.session_state.next_cue_time - now

        st.markdown(
            f"<div style='text-align:center;font-size:72px;line-height:1;padding:10px;'>⏳</div>"
            f"<div style='text-align:center;font-size:32px;font-weight:bold;'>"
            f"Marks: {n_done} / {n_total}</div>",
            unsafe_allow_html=True,
        )
        time.sleep(max(0.0, time_to_next - 0.05))
        st.rerun()

# ── ENTER TIMES ───────────────────────────────────────────────────────────────
elif phase == "entering":
    st.title("✍️ Enter Your Stopwatch Readings")
    st.markdown(
        "Enter the **lap time** shown on your stopwatch at each **MARK!** — "
        "the time since the previous mark (or since GO for mark 1).  \n"
        "Format: `M:SS.xx` (e.g. `0:14.83`) or decimal seconds (e.g. `14.83`)."
    )

    n = len(st.session_state.cue_actual_elapsed)

    with st.form("entry_form"):
        entries = []
        for i in range(n):
            v = st.text_input(f"Mark {i + 1}", key=f"entry_{i}", placeholder="M:SS.xx")
            entries.append(v)

        submit = st.form_submit_button(
            "Calculate Results →", type="primary", use_container_width=True
        )

    if submit:
        parsed = [parse_time(v) for v in entries]
        failed = [i + 1 for i, v in enumerate(parsed) if v is None]
        if failed:
            st.error(f"Could not parse entries for marks: {failed}. Use M:SS.xx or decimal seconds.")
        else:
            st.session_state.entered_times = parsed
            st.session_state.phase = "results"
            st.rerun()

# ── RESULTS ───────────────────────────────────────────────────────────────────
elif phase == "results":
    cumulative = st.session_state.cue_actual_elapsed
    entered = st.session_state.entered_times
    n = len(cumulative)

    # Convert cumulative cue times to lap intervals
    actual = [cumulative[0]] + [cumulative[i] - cumulative[i - 1] for i in range(1, n)]

    errors = [entered[i] - actual[i] for i in range(n)]
    abs_errors = [abs(e) for e in errors]

    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors, ddof=1)) if n > 1 else 0.0
    mean_abs = float(np.mean(abs_errors))
    max_abs = float(max(abs_errors))

    st.title("📊 Repeatability Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Error", f"{mean_err:+.3f}s", help="Bias: + = reading high (fast), − = reading low (slow/reaction lag)")
    c2.metric("Std Dev (σ)", f"{std_err:.3f}s", help="Consistency — the lower the better")
    c3.metric("Mean |Error|", f"{mean_abs:.3f}s")
    c4.metric("Max |Error|", f"{max_abs:.3f}s")

    # ── Rating banner ──
    if std_err < 0.3:
        rating, msg, color = "Excellent", "Very high consistency — errors are minimal.", "#27ae60"
    elif std_err < 0.7:
        rating, msg, color = "Good", "Solid consistency with minor variation.", "#2980b9"
    elif std_err < 1.5:
        rating, msg, color = "Fair", "Noticeable variation — more practice will help.", "#f39c12"
    else:
        rating, msg, color = "Needs Work", "High variation — focus on reducing reaction-time inconsistency.", "#e74c3c"

    st.markdown(
        f"<div style='text-align:center;padding:14px;background:{color};color:white;"
        f"border-radius:12px;font-size:20px;font-weight:bold;margin:12px 0;'>"
        f"Repeatability: {rating} &mdash; {msg}</div>",
        unsafe_allow_html=True,
    )

    # ── Bias note ──
    if abs(mean_err) > 0.4:
        if mean_err > 0:
            st.warning(
                f"Systematic bias: your readings run **{mean_err:.3f}s fast** on average "
                f"(you may be anticipating the mark slightly, or your stopwatch is ahead of reference)."
            )
        else:
            st.warning(
                f"Systematic bias: your readings run **{abs(mean_err):.3f}s slow** on average "
                f"(likely a consistent reaction / glance delay)."
            )

    # ── Error chart ──
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, n + 1)),
            y=errors,
            mode="lines+markers",
            name="Error",
            marker=dict(size=9, color="#e74c3c"),
            line=dict(color="#e74c3c", width=2),
            hovertemplate="Mark %{x}<br>Error: %{y:+.3f}s<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="#2c3e50", line_width=1)
    fig.add_hline(
        y=mean_err,
        line_dash="dot",
        line_color="#3498db",
        annotation_text=f"Mean {mean_err:+.3f}s",
        annotation_position="bottom right",
    )
    # ±1σ band
    fig.add_hrect(
        y0=mean_err - std_err,
        y1=mean_err + std_err,
        fillcolor="#3498db",
        opacity=0.08,
        line_width=0,
        annotation_text="±1σ",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Error per Mark  (Your Lap Time − Actual Lap Interval)",
        xaxis_title="Mark #",
        yaxis_title="Error (seconds)",
        height=380,
        hovermode="x unified",
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Detail table ──
    df = pd.DataFrame(
        {
            "Mark": range(1, n + 1),
            "Actual lap": [fmt_elapsed(a) for a in actual],
            "Your lap": [fmt_elapsed(e) for e in entered],
            "Error (s)": [f"{e:+.3f}" for e in errors],
            "|Error| (s)": [f"{a:.3f}" for a in abs_errors],
        }
    )
    st.dataframe(df, hide_index=True, use_container_width=True)

    # ── Re-run ──
    st.divider()
    if st.button("🔁 Run Again", type="primary", use_container_width=True):
        reset_state()
        st.rerun()
