"""
Gait Analysis System — Streamlit Frontend
Computer Vision-Based Human Walking Pattern Analysis
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
import tempfile
import os
import time
from pathlib import Path

# ──────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Gait Analysis System",
    page_icon="🦶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL STYLING — Dark Scientific Dashboard
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base & Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #080d14;
        color: #c8d8e8;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #0d1a2e 100%);
        border-right: 1px solid #1a2e45;
    }
    section[data-testid="stSidebar"] * {
        color: #a8c4d8 !important;
    }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stCheckbox label {
        font-size: 0.82rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #5b8fa8 !important;
    }

    /* ── Main container ── */
    .main .block-container {
        padding: 1.5rem 2.5rem 3rem;
        max-width: 1400px;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d1e30 0%, #0a1a28 100%);
        border: 1px solid #1c3a55;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        box-shadow: 0 0 18px rgba(0, 140, 220, 0.06);
        transition: box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 0 28px rgba(0, 180, 255, 0.14);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #4a8fa8 !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        color: #38d9f5 !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
        color: #2eb88a !important;
    }

    /* ── Section headers ── */
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #2a7fa0;
        border-bottom: 1px solid #1a3349;
        padding-bottom: 0.45rem;
        margin-bottom: 1.2rem;
        margin-top: 2rem;
    }

    /* ── Info / explanation box ── */
    .info-box {
        background: linear-gradient(135deg, #0a1d30 0%, #071626 100%);
        border-left: 3px solid #00b4e0;
        border-radius: 6px;
        padding: 1rem 1.4rem;
        font-size: 0.88rem;
        line-height: 1.7;
        color: #8ab8cc;
        margin-bottom: 1.5rem;
    }
    .info-box strong {
        color: #38d9f5;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        letter-spacing: 0.06em;
    }

    /* ── File detail chips ── */
    .file-chip {
        display: inline-block;
        background: #0d2035;
        border: 1px solid #1a3a55;
        border-radius: 20px;
        padding: 0.25rem 0.85rem;
        font-size: 0.78rem;
        font-family: 'IBM Plex Mono', monospace;
        color: #5baec8;
        margin-right: 0.5rem;
        margin-top: 0.4rem;
    }

    /* ── Status badge ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #041520;
        border: 1px solid #1a4060;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.76rem;
        font-family: 'IBM Plex Mono', monospace;
        color: #38d9f5;
    }
    .status-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #38d9f5;
        animation: pulse-dot 1.8s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.25; }
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #0062a3, #004e82);
        color: #e0f4ff;
        border: 1px solid #0080c8;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        letter-spacing: 0.06em;
        padding: 0.55rem 1.4rem;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #007acc, #0062a3);
        box-shadow: 0 0 20px rgba(0, 160, 255, 0.3);
        border-color: #38d9f5;
    }

    /* ── Dividers ── */
    hr {
        border-color: #112233;
    }

    /* ── Matplotlib figure background fix ── */
    .stImage > img {
        border-radius: 8px;
    }

    /* ── Plot containers ── */
    .plot-container {
        background: #080e18;
        border: 1px solid #152840;
        border-radius: 10px;
        padding: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# MATPLOTLIB DARK THEME DEFAULTS
# ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#080e18",
    "axes.facecolor": "#0b1520",
    "axes.edgecolor": "#1a3349",
    "axes.labelcolor": "#6ab0c8",
    "axes.titlecolor": "#38d9f5",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "text.color": "#8ab8cc",
    "xtick.color": "#4a7a94",
    "ytick.color": "#4a7a94",
    "grid.color": "#112233",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.8,
    "font.family": "monospace",
    "figure.dpi": 120,
})

# ──────────────────────────────────────────────
# UTILITY / BACKEND INTERFACE FUNCTIONS
# ──────────────────────────────────────────────

def load_video(uploaded_file) -> dict:
    """
    Saves the uploaded video to a temp file and returns metadata.
    In production, pass the file path to your CV backend.
    """
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    size_mb = os.path.getsize(tmp.name) / (1024 ** 2)
    return {
        "path": tmp.name,
        "name": uploaded_file.name,
        "size_mb": round(size_mb, 2),
        "format": suffix.lstrip(".").upper(),
    }


def process_video(video_info: dict, options: dict) -> dict:
    """
    Hook for your CV backend. Currently synthesises realistic gait
    signals so the UI is fully functional without a connected backend.

    Replace the body of this function with calls to your processing
    pipeline (e.g. import your module and call run_analysis(video_info["path"])).

    Returns
    -------
    dict with keys:
        frames          – list of processed frame arrays (or None if backend absent)
        whole_motion    – 1-D numpy array of whole-body motion signal
        lower_motion    – 1-D numpy array of lower-body motion signal
        smooth_motion   – smoothed version of lower_motion
        centroid_path   – (N,2) numpy array of (x,y) centroid positions
        peaks           – indices of detected step peaks in lower_motion
    """
    # ── Synthesise plausible gait signals ──────────────────────────
    rng = np.random.default_rng(42)
    n = 300  # simulate 300 frames at ~30 fps ≈ 10 s clip

    t = np.linspace(0, 10, n)
    freq = 1.8 + rng.uniform(-0.1, 0.1)         # ~1.8 Hz step cadence
    base_whole  = 18 + 6 * np.sin(2 * np.pi * 0.9 * t) + rng.normal(0, 2, n)
    base_lower  = 22 + 9 * np.sin(2 * np.pi * freq * t) + rng.normal(0, 2.5, n)

    smooth_level = options.get("smoothing", 5)
    kernel = np.ones(smooth_level) / smooth_level
    smooth_motion = np.convolve(base_lower, kernel, mode="same")

    peaks, _ = find_peaks(smooth_motion, distance=int(30 / freq), prominence=4)

    # Centroid path — slightly curved walk across frame
    cx = np.linspace(80, 540, n) + rng.normal(0, 3, n)
    cy = 240 + 20 * np.sin(2 * np.pi * 0.4 * t) + rng.normal(0, 2, n)
    centroid_path = np.column_stack([cx, cy])

    return {
        "frames": None,           # swap with actual frame list when backend connected
        "whole_motion": base_whole,
        "lower_motion": base_lower,
        "smooth_motion": smooth_motion,
        "centroid_path": centroid_path,
        "peaks": peaks,
    }


def compute_metrics(results: dict) -> dict:
    """
    Derives high-level gait metrics from the motion signal analysis.
    In production, replace/augment with values returned by your backend.
    """
    peaks       = results["peaks"]
    smooth      = results["smooth_motion"]
    centroid    = results["centroid_path"]
    n_frames    = len(smooth)
    fps         = 30.0

    # Walking speed — centroid displacement per frame
    displacements = np.linalg.norm(np.diff(centroid, axis=0), axis=1)
    walking_speed = float(np.median(displacements))

    # Gait cycle frequency — steps per second
    if len(peaks) >= 2:
        inter_peak = np.diff(peaks) / fps
        gait_freq  = float(1.0 / np.mean(inter_peak))
    else:
        gait_freq  = 0.0

    # Stride length — average centroid distance between every 2 steps
    if len(peaks) >= 3:
        stride_lengths = []
        for i in range(0, len(peaks) - 2, 2):
            p1, p2 = centroid[peaks[i]], centroid[peaks[i + 2]]
            stride_lengths.append(np.linalg.norm(p2 - p1))
        stride_length = float(np.mean(stride_lengths))
    else:
        stride_length = 0.0

    # Motion smoothness — inverse of normalised jerk (mean abs 2nd derivative)
    jerk = np.abs(np.diff(smooth, n=2))
    smoothness = float(1.0 / (1.0 + np.mean(jerk)))

    return {
        "walking_speed":   round(walking_speed, 2),
        "gait_frequency":  round(gait_freq, 3),
        "stride_length":   round(stride_length, 1),
        "motion_smoothness": round(smoothness, 4),
        "step_count":      len(peaks),
        "duration_s":      round(n_frames / fps, 1),
    }


def plot_gait_graph(results: dict, options: dict) -> plt.Figure:
    """
    Renders the step-detection plot — lower-body motion signal with
    detected step peaks highlighted.
    """
    lower  = results["lower_motion"]
    smooth = results["smooth_motion"]
    peaks  = results["peaks"]
    n      = len(lower)
    frames = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 3.2))

    ax.plot(frames, lower, color="#1a4a6a", linewidth=1.1, alpha=0.55, label="Raw signal")
    ax.plot(frames, smooth, color="#00a8cc", linewidth=1.9, label="Smoothed signal")
    ax.scatter(peaks, smooth[peaks], color="#ff6b35", s=55, zorder=5,
               edgecolors="#ffa07a", linewidths=0.8, label=f"Steps detected ({len(peaks)})")

    ax.set_title("Step Detection via Lower Body Motion", pad=10)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Motion Magnitude (px)")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.25,
              facecolor="#0a1a28", edgecolor="#1a3349")
    ax.grid(True, alpha=0.4)
    fig.tight_layout(pad=1.2)
    return fig


def plot_trajectory(results: dict) -> plt.Figure:
    """
    Plots the smoothed centroid trajectory as a 2-D walking path.
    """
    path = results["centroid_path"]
    x, y = path[:, 0], path[:, 1]
    n    = len(x)

    fig, ax = plt.subplots(figsize=(10, 3.0))

    # Colour-map trajectory by progression (early=dark, late=bright)
    for i in range(n - 1):
        alpha = 0.25 + 0.75 * (i / n)
        ax.plot(x[i:i+2], y[i:i+2], color="#00b4e0", alpha=alpha, linewidth=1.6)

    ax.scatter(x[0],  y[0],  color="#2eb88a", s=80, zorder=5, label="Start")
    ax.scatter(x[-1], y[-1], color="#ff6b35", s=80, zorder=5, label="End")

    ax.set_title("Walking Trajectory  ·  Centroid Path", pad=10)
    ax.set_xlabel("Horizontal Position (px)")
    ax.set_ylabel("Vertical Position (px)")
    ax.invert_yaxis()   # image-coordinate convention
    ax.legend(fontsize=8, framealpha=0.25, facecolor="#0a1a28", edgecolor="#1a3349")
    ax.grid(True, alpha=0.35)
    fig.tight_layout(pad=1.2)
    return fig


# ──────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ──────────────────────────────────────────────
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "results" not in st.session_state:
    st.session_state.results = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "video_info" not in st.session_state:
    st.session_state.video_info = None

# ──────────────────────────────────────────────
# SIDEBAR — CONTROL PANEL
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="
            font-family:'IBM Plex Mono',monospace;
            font-size:0.68rem;
            letter-spacing:0.2em;
            text-transform:uppercase;
            color:#2a7fa0;
            padding-bottom:0.6rem;
            border-bottom:1px solid #1a3349;
            margin-bottom:1.2rem;
        ">
        ◈ Control Panel
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Video upload ──
    uploaded_file = st.file_uploader(
        "Upload Walking Video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supports MP4, AVI, MOV, MKV",
    )

    st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

    # ── Display toggles ──
    st.markdown(
        "<p style='font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;"
        "color:#2a7fa0;margin-bottom:0.4rem;font-family:IBM Plex Mono,monospace;'>"
        "Overlay Options</p>",
        unsafe_allow_html=True,
    )
    show_bboxes     = st.checkbox("Show Bounding Boxes",      value=True)
    show_trajectory = st.checkbox("Show Centroid Trajectory", value=True)
    show_lower      = st.checkbox("Show Lower-Body Analysis", value=True)

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    # ── Smoothing slider ──
    smoothing_level = st.slider(
        "Smoothing Level",
        min_value=2, max_value=20, value=7, step=1,
        help="Kernel size for motion signal smoothing",
    )

    st.markdown("<div style='margin-top:1.4rem'></div>", unsafe_allow_html=True)

    # ── Run button ──
    run_analysis = st.button("▶  Run Analysis", disabled=(uploaded_file is None))

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.68rem;color:#2a5a72;text-align:center;"
        "font-family:IBM Plex Mono,monospace;line-height:1.8;'>"
        "Gait Analysis System v1.0<br/>CV-Based Motion Analysis</p>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# MAIN — PAGE HEADER
# ──────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom:0.2rem;">
        <h1 style="
            font-family:'IBM Plex Mono',monospace;
            font-size:1.9rem;
            font-weight:600;
            color:#38d9f5;
            letter-spacing:0.04em;
            margin-bottom:0.15rem;
        ">
        Gait Analysis System
        </h1>
        <p style="
            font-family:'IBM Plex Sans',sans-serif;
            font-size:0.9rem;
            color:#4a8fa8;
            margin:0;
            letter-spacing:0.06em;
        ">
        Computer Vision-Based Human Walking Pattern Analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-bottom:1.4rem'></div>", unsafe_allow_html=True)

# ── What is Gait Analysis? ──
st.markdown(
    """
    <div class="info-box">
        <strong>◈ WHAT IS GAIT ANALYSIS?</strong><br/>
        Gait analysis is the systematic study of human walking patterns using motion capture or
        computer vision techniques. It quantifies biomechanical parameters — cadence, stride length,
        and movement smoothness — to reveal insights into an individual's musculoskeletal health.
        Widely used in <em>clinical rehabilitation</em>, <em>sports performance optimisation</em>,
        and <em>neurological assessment</em>, gait analysis enables objective, data-driven evaluation
        of walking function without invasive procedures.
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# SECTION A — VIDEO INPUT
# ──────────────────────────────────────────────
st.markdown("<div class='section-header'>A — Video Input</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    video_info = load_video(uploaded_file)
    st.session_state.video_info = video_info

    col_prev, col_meta = st.columns([2, 1], gap="large")

    with col_prev:
        st.video(video_info["path"])

    with col_meta:
        st.markdown(
            "<p style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;"
            "letter-spacing:0.1em;text-transform:uppercase;color:#2a7fa0;"
            "margin-bottom:0.7rem;'>File Details</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<span class='file-chip'>📄 {video_info['name']}</span>"
            f"<span class='file-chip'>⚖ {video_info['size_mb']} MB</span>"
            f"<span class='file-chip'>🎞 {video_info['format']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
        st.info(
            "Upload complete. Configure overlay options in the sidebar, "
            "then press **Run Analysis** to begin processing.",
            icon="ℹ️",
        )
else:
    st.markdown(
        """
        <div style="
            border: 1px dashed #1a3a55;
            border-radius: 10px;
            padding: 2.5rem;
            text-align: center;
            color: #2a5a72;
            font-family: IBM Plex Mono, monospace;
            font-size: 0.82rem;
            letter-spacing: 0.06em;
        ">
            Upload a walking video via the sidebar to begin
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# ANALYSIS TRIGGER
# ──────────────────────────────────────────────
if run_analysis and uploaded_file is not None:
    options = {
        "show_bboxes":     show_bboxes,
        "show_trajectory": show_trajectory,
        "show_lower":      show_lower,
        "smoothing":       smoothing_level,
    }
    with st.spinner("Processing video — extracting motion signals…"):
        time.sleep(0.8)  # remove when real backend provides progress
        results = process_video(st.session_state.video_info, options)
        metrics = compute_metrics(results)

    st.session_state.results       = results
    st.session_state.metrics       = metrics
    st.session_state.analysis_done = True
    st.success("Analysis complete.", icon="✅")

# ──────────────────────────────────────────────
# SECTION B — LIVE PROCESSING DISPLAY
# ──────────────────────────────────────────────
st.markdown("<div class='section-header'>B — Motion Analysis Preview</div>", unsafe_allow_html=True)

if st.session_state.analysis_done and st.session_state.results is not None:
    results = st.session_state.results

    if results["frames"] is not None:
        # Real backend provided frames — display them
        st.image(results["frames"][0], caption="Processed frame with overlays", use_column_width=True)
    else:
        # Show a schematic overlay diagram using matplotlib
        fig_frame, ax_f = plt.subplots(figsize=(10, 4))
        ax_f.set_xlim(0, 640); ax_f.set_ylim(0, 480); ax_f.invert_yaxis()

        # Simulated person silhouette ─ upper + lower body bounding boxes
        if show_bboxes:
            upper_rect = mpatches.FancyBboxPatch(
                (270, 60), 100, 140, boxstyle="round,pad=4",
                linewidth=1.5, edgecolor="#00ccff", facecolor="#00ccff11",
            )
            lower_rect = mpatches.FancyBboxPatch(
                (280, 210), 80, 160, boxstyle="round,pad=4",
                linewidth=1.5, edgecolor="#ff9900", facecolor="#ff990011",
            )
            ax_f.add_patch(upper_rect)
            ax_f.add_patch(lower_rect)
            ax_f.text(274, 55,  "Upper Body", fontsize=7, color="#00ccff")
            ax_f.text(284, 205, "Lower Body", fontsize=7, color="#ff9900")

        # Trajectory path
        if show_trajectory:
            path = results["centroid_path"]
            ax_f.plot(path[:, 0], path[:, 1], color="#38d9f5", linewidth=1.2, alpha=0.6)
            ax_f.scatter(path[-1, 0], path[-1, 1], color="#38d9f5", s=60, zorder=5)
            ax_f.text(path[-1, 0] + 8, path[-1, 1], "Centroid", fontsize=7, color="#38d9f5")

        ax_f.set_title("Movement Tracking Overlay  ·  Frame Preview", pad=10)
        ax_f.set_xlabel("Horizontal (px)")
        ax_f.set_ylabel("Vertical (px)")
        ax_f.grid(True, alpha=0.2)
        fig_frame.tight_layout(pad=1.2)
        st.pyplot(fig_frame, use_container_width=True)
        plt.close(fig_frame)
else:
    st.markdown(
        "<p style='color:#2a5a72;font-size:0.85rem;font-family:IBM Plex Mono,monospace;"
        "padding:1rem 0;'>Processed frame preview will appear here after analysis.</p>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# SECTION C — GAIT METRICS DASHBOARD
# ──────────────────────────────────────────────
st.markdown("<div class='section-header'>C — Gait Metrics Dashboard</div>", unsafe_allow_html=True)

if st.session_state.analysis_done and st.session_state.metrics is not None:
    m = st.session_state.metrics
    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        st.metric(
            label="Walking Speed",
            value=f"{m['walking_speed']} px/fr",
            delta="within normal range",
        )
    with c2:
        st.metric(
            label="Gait Cycle Freq.",
            value=f"{m['gait_frequency']} Hz",
            delta=f"{m['step_count']} steps detected",
        )
    with c3:
        st.metric(
            label="Stride Length",
            value=f"{m['stride_length']} px",
            delta=f"over {m['duration_s']} s",
        )
    with c4:
        st.metric(
            label="Motion Smoothness",
            value=f"{m['motion_smoothness']}",
            delta="higher = smoother gait",
        )
else:
    st.markdown(
        "<p style='color:#2a5a72;font-size:0.85rem;font-family:IBM Plex Mono,monospace;"
        "padding:1rem 0;'>Gait metrics will populate after analysis completes.</p>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# SECTION D — STEP DETECTION GRAPH
# ──────────────────────────────────────────────
st.markdown("<div class='section-header'>D — Step Detection via Lower Body Motion</div>", unsafe_allow_html=True)

if st.session_state.analysis_done and st.session_state.results is not None:
    fig_gait = plot_gait_graph(
        st.session_state.results,
        {"show_lower": show_lower},
    )
    st.pyplot(fig_gait, use_container_width=True)
    plt.close(fig_gait)
else:
    st.markdown(
        "<p style='color:#2a5a72;font-size:0.85rem;font-family:IBM Plex Mono,monospace;"
        "padding:1rem 0;'>Step detection graph will render after analysis.</p>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# SECTION E — TRAJECTORY VISUALISATION
# ──────────────────────────────────────────────
st.markdown("<div class='section-header'>E — Walking Trajectory</div>", unsafe_allow_html=True)

if st.session_state.analysis_done and st.session_state.results is not None:
    fig_traj = plot_trajectory(st.session_state.results)
    st.pyplot(fig_traj, use_container_width=True)
    plt.close(fig_traj)

    # Summary row beneath trajectory
    m = st.session_state.metrics
    st.markdown(
        f"""
        <div style="
            display:flex; gap:1.5rem; margin-top:0.8rem;
            font-family:IBM Plex Mono,monospace; font-size:0.78rem; color:#4a8fa8;
        ">
            <span>◈ Path length derived from centroid displacement</span>
            <span>·</span>
            <span>Analysis duration: {m['duration_s']} s</span>
            <span>·</span>
            <span>Steps captured: {m['step_count']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<p style='color:#2a5a72;font-size:0.85rem;font-family:IBM Plex Mono,monospace;"
        "padding:1rem 0;'>Walking trajectory will render after analysis.</p>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("<div style='margin-top:3rem'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <hr/>
    <p style="
        text-align:center;
        font-family:IBM Plex Mono,monospace;
        font-size:0.68rem;
        color:#1e4a62;
        letter-spacing:0.1em;
    ">
    GAIT ANALYSIS SYSTEM  ·  COMPUTER VISION BIOMECHANICS MODULE  ·  FOR RESEARCH USE
    </p>
    """,
    unsafe_allow_html=True,
)
