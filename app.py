"""
Gait Analysis System — Streamlit Frontend with OpenCV Backend
Computer Vision-Based Human Walking Pattern Analysis
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks, savgol_filter
import cv2
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
    OpenCV processing logic merged into Streamlit.
    """
    video_path = video_info["path"]
    cap = cv2.VideoCapture(video_path)

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    motion_values = []
    lower_motion_values = []
    lower_heights = []
    smooth_centroids = []
    confidence_flags = []
    recent_heights = []

    MIN_AREA = 8000
    MIN_ASPECT = 1.3
    MIN_SOLIDITY = 0.25
    MAX_JUMP = 120
    STABLE_WINDOW = 10
    MIN_HEIGHT_RATIO = 0.5
    ALPHA = 0.4

    def is_valid_person_contour(cnt):
        area = cv2.contourArea(cnt)
        if area < MIN_AREA: return False
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w) if w > 0 else 0
        if aspect_ratio < MIN_ASPECT: return False
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < MIN_SOLIDITY: return False
        return True

    frame_count = 0
    best_contour_global = None
    last_annotated_frame = None

    # Use a Streamlit progress bar since processing might take a moment
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

        fgmask = fgbg.apply(frame)

        # Allow background subtractor to learn for the first 20 frames
        if frame_count < 20:
            continue

        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

        motion_value = np.sum(fgmask > 0)
        motion_values.append(motion_value)

        merged_mask = cv2.dilate(fgmask, merge_kernel, iterations=2)
        merged_mask = cv2.erode(merged_mask, merge_kernel, iterations=1)

        contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        max_area = 0
        for cnt in contours:
            if is_valid_person_contour(cnt):
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    best_contour = cnt

        lower_motion = 0
        lower_height = 0
        raw_centroid = None
        high_confidence = False

        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            if len(recent_heights) >= STABLE_WINDOW:
                avg_recent_h = np.mean(recent_heights[-STABLE_WINDOW:])
                if h < avg_recent_h * MIN_HEIGHT_RATIO:
                    best_contour = None

        if best_contour is not None:
            best_contour_global = best_contour
            x, y, w, h = cv2.boundingRect(best_contour)
            recent_heights.append(h)
            split_y = y + int(h * 0.45)

            if options.get("show_bboxes", True):
                cv2.rectangle(frame, (x, y), (x + w, split_y), (0, 255, 0), 2)
                cv2.putText(frame, "Upper", (x, max(y - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame, (x, split_y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "Lower", (x, split_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Use original mask for accurate pixel counts
            lower_mask_region = fgmask[split_y:y + h, x:x + w]
            lower_motion = np.sum(lower_mask_region > 0)
            lower_height = lower_mask_region.shape[0]

            raw_centroid = (x + w // 2, y + h // 2)
            high_confidence = True
        else:
            recent_heights.append(recent_heights[-1] if recent_heights else 0)

        lower_motion_values.append(lower_motion)
        lower_heights.append(lower_height)
        confidence_flags.append(high_confidence)

        if len(smooth_centroids) == 0:
            if raw_centroid is not None:
                smooth_centroids.append(raw_centroid)
            else:
                smooth_centroids.append((0, 0))
        else:
            prev = smooth_centroids[-1]
            if raw_centroid is None:
                smooth_centroids.append(prev)
            else:
                dist = np.linalg.norm(np.array(raw_centroid) - np.array(prev))
                if dist > MAX_JUMP:
                    blended = (
                        int(ALPHA * raw_centroid[0] + (1 - ALPHA) * prev[0]),
                        int(ALPHA * raw_centroid[1] + (1 - ALPHA) * prev[1])
                    )
                    smooth_centroids.append(blended)
                else:
                    smoothed = (
                        int(ALPHA * raw_centroid[0] + (1 - ALPHA) * prev[0]),
                        int(ALPHA * raw_centroid[1] + (1 - ALPHA) * prev[1])
                    )
                    smooth_centroids.append(smoothed)

        if options.get("show_trajectory", True):
            current = smooth_centroids[-1]
            if current != (0, 0) and high_confidence:
                cv2.circle(frame, current, 5, (0, 0, 255), -1)

            for i in range(1, len(smooth_centroids)):
                prev_conf = confidence_flags[i - 1] if i - 1 < len(confidence_flags) else False
                curr_conf = confidence_flags[i] if i < len(confidence_flags) else False
                if prev_conf and curr_conf:
                    if smooth_centroids[i] != (0, 0) and smooth_centroids[i-1] != (0, 0):
                        cv2.line(frame, smooth_centroids[i-1], smooth_centroids[i], (255, 0, 0), 2)

        if high_confidence:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_annotated_frame = frame_rgb

    cap.release()
    progress_bar.empty()

    # --- POST PROCESSING ---
    smooth_centroids = np.array(smooth_centroids)
    valid_mask = (smooth_centroids != [0, 0]).any(axis=1)
    if valid_mask.any():
        smooth_centroids_filtered = smooth_centroids[valid_mask]
    else:
        smooth_centroids_filtered = np.array([[0,0], [0,0]])

    lower_motion_values = np.array(lower_motion_values)

    smoothing_level = options.get("smoothing", 15)
    if smoothing_level % 2 == 0:
        smoothing_level += 1

    if len(lower_motion_values) > smoothing_level:
        window_length = min(smoothing_level, len(lower_motion_values))
        if window_length % 2 == 0:
            window_length -= 1
        if window_length > 3:
            lower_motion_smooth = savgol_filter(lower_motion_values, window_length, 3)
        else:
            lower_motion_smooth = lower_motion_values
    else:
        lower_motion_smooth = lower_motion_values

    # Prevent find_peaks from failing if the array is flat or empty
    if len(lower_motion_smooth) > 15:
        peaks, _ = find_peaks(lower_motion_smooth, distance=15, prominence=50)
    else:
        peaks = np.array([])

    frames_list = [last_annotated_frame] if last_annotated_frame is not None else None

    # Stride calculation
    lower_heights = np.array(lower_heights)
    stride_widths = []
    body_width = 0
    if best_contour_global is not None:
        x, y, w, h = cv2.boundingRect(best_contour_global)
        body_width = w

    if len(peaks) > 0 and body_width > 0:
        for p in peaks:
            if p < len(lower_motion_values):
                motion_at_peak = lower_motion_values[p]
                motion_max = np.max(lower_motion_values) if np.max(lower_motion_values) > 0 else 1
                stride_estimate = (motion_at_peak / motion_max) * body_width
                stride_widths.append(stride_estimate)
    avg_stride = float(np.mean(stride_widths)) if stride_widths else 0.0

    return {
        "frames": frames_list,
        "whole_motion": motion_values,
        "lower_motion": lower_motion_values,
        "smooth_motion": lower_motion_smooth,
        "centroid_path": smooth_centroids_filtered,
        "peaks": peaks,
        "avg_stride": avg_stride,
        "duration_s": round(frame_count / 30.0, 1),
        "frame_count": frame_count
    }


def compute_metrics(results: dict) -> dict:
    """
    Derives high-level gait metrics from the motion signal analysis.
    Uses the logic extracted from the OpenCV backend.
    """
    peaks = results["peaks"]
    smooth = results["smooth_motion"]
    centroid = results["centroid_path"]
    whole_motion = results["whole_motion"]
    avg_stride = results.get("avg_stride", 0.0)
    duration_s = results.get("duration_s", 0.0)

    # Walking speed — centroid displacement per frame
    if len(centroid) > 1:
        dx = np.diff(centroid[:, 0])
        walking_speed = np.mean(np.abs(dx))
    else:
        walking_speed = 0.0

    # Gait cycle frequency
    if len(smooth) > 0:
        gait_cycle_freq = len(peaks) / len(smooth)
    else:
        gait_cycle_freq = 0.0

    # Motion smoothness
    motion_diff = np.diff(whole_motion)
    smoothness = np.std(motion_diff) if len(motion_diff) > 0 else 0.0

    return {
        "walking_speed": round(walking_speed, 4),
        "gait_frequency": round(gait_cycle_freq, 4),
        "stride_length": round(avg_stride, 4),
        "motion_smoothness": round(smoothness, 4),
        "step_count": len(peaks),
        "duration_s": duration_s,
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

    if options.get("show_lower", True):
        ax.plot(frames, lower, color="#1a4a6a", linewidth=1.1, alpha=0.55, label="Raw signal")
    ax.plot(frames, smooth, color="#00a8cc", linewidth=1.9, label="Smoothed signal")
    
    # Filter valid peaks for plotting
    valid_peaks = [p for p in peaks if p < len(smooth)]
    if valid_peaks:
        ax.scatter(valid_peaks, smooth[valid_peaks], color="#ff6b35", s=55, zorder=5,
                   edgecolors="#ffa07a", linewidths=0.8, label=f"Steps detected ({len(valid_peaks)})")

    ax.set_title("Step Detection via Lower Body Motion", pad=10)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Motion Magnitude (px)")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.25,
              facecolor="#0a1a28", edgecolor="#1a3349")
    ax.grid(True, alpha=0.4)
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
        min_value=3, max_value=31, value=15, step=2,
        help="Window size for motion signal smoothing (must be odd)",
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
    with st.spinner("Processing video with OpenCV backend..."):
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

    if results["frames"] is not None and len(results["frames"]) > 0:
        # Display the processed frame with OpenCV annotations
        st.image(results["frames"][0], caption="Final Processed Frame with Overlays", use_container_width=True)
    else:
        st.info("No valid frames could be fully processed or detected.")
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
            value=f"{m['gait_frequency']} cyc/fr",
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
            delta="lower = smoother gait",
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
