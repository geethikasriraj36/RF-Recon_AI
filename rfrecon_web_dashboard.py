from __future__ import annotations

from pathlib import Path
import io
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rfrecon_consistent_features import extract_features, FEATURE_NAMES

# ---------------------------------------------------------------------
# RFRecon Web Dashboard
# New Streamlit interface; leaves the existing Tkinter dashboard intact.
# ---------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "rfrecon_five_class_synthetic_rf.pkl"
FEATURE_PATH = APP_DIR / "rfrecon_five_class_synthetic_features.pkl"
WINDOW_SIZE = 4096
MAX_ANALYSIS_WINDOWS = 1500
CLASS_LABELS = {
    "noise": "Noise",
    "wifi_like": "Wi-Fi-like",
    "bluetooth_like": "Bluetooth-like",
    "cellular_like": "Cellular-like",
    "zigbee_like": "Zigbee-like*",
}

st.set_page_config(
    page_title="RFRecon | Signal Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
      :root { --ink:#eaf1ff; --muted:#91a2bf; --panel:#101b2d; --line:#25344d; --cyan:#55d6d0; }
      .stApp { background: radial-gradient(ellipse at 12% 0%, #172d4a 0%, #0a1220 42%, #080e18 100%); color:var(--ink); }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
      html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"] { background-color:#080e18 !important; }
      [data-testid="stHeader"], header[data-testid="stHeader"] { background:#080e18 !important; height:2.2rem !important; }
      [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none !important; }
      [data-testid="stToolbar"] { right:1rem; }
      [data-testid="stMainBlockContainer"] { padding-top:.35rem !important; padding-bottom:.5rem !important; max-width:1800px; }
      [data-testid="stAppViewContainer"] { min-height:100vh; }
      [data-testid="stMain"] { padding-top:0 !important; }
      [data-testid="stBottom"], footer { display:none !important; }
      [data-testid="stAppViewContainer"] > .main { background:transparent !important; }
      [data-testid="stSidebar"] { padding-top:0 !important; }
      h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.025em; }
      h1 { font-size:2.35rem !important; }
      [data-testid="stSidebar"] { background:#0b1422; border-right:1px solid #24334b; }
      [data-testid="stMetric"] { background:linear-gradient(145deg,rgba(25,42,65,.92),rgba(14,25,41,.92)); border:1px solid #293c58; padding:16px 18px; border-radius:16px; }
      [data-testid="stMetricLabel"] { color:#9fb0c9 !important; }
      [data-testid="stMetricValue"] { color:#f4f8ff !important; font-family:'Space Grotesk',sans-serif; }
      [data-testid="stFileUploader"] { background:rgba(18,31,50,.65); border:1px dashed #3c5577; border-radius:14px; padding:8px; }
      .hero { padding:26px 28px; border:1px solid #2c4361; border-radius:22px; background:linear-gradient(115deg,rgba(21,43,68,.95),rgba(13,25,42,.75)); margin-bottom:18px; }
      .eyebrow { text-transform:uppercase; color:#55d6d0; font-size:.74rem; font-weight:700; letter-spacing:.17em; }
      .hero-title { font-family:'Space Grotesk',sans-serif; font-size:2.05rem; line-height:1.15; font-weight:700; color:#f2f7ff; margin:8px 0; }
      .hero-copy { color:#a9bad2; font-size:1rem; max-width:780px; }
      .pill { display:inline-block; border:1px solid #31516b; background:#132a3d; color:#8ce6df; padding:5px 10px; border-radius:100px; font-size:.76rem; margin-right:6px; }
      .section-note { color:#a2b3cb; font-size:.91rem; }
      div.stButton > button[kind="primary"] { background:#42c9c2; color:#061522; border:0; border-radius:10px; font-weight:700; }
      div.stButton > button { border-radius:10px; }
      hr { border-color:#25344d; }
      .small-muted { color:#91a2bf; font-size:.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

def pretty_class(value: object) -> str:
    return CLASS_LABELS.get(str(value), str(value).replace("_", " ").title())

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
        raise FileNotFoundError(
            "Five-class model files are missing. Put rfrecon_five_class_synthetic_rf.pkl and "
            "rfrecon_five_class_synthetic_features.pkl beside this app."
        )
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)
    if list(feature_names) != list(FEATURE_NAMES):
        raise ValueError("Saved model feature order does not match the feature extractor.")
    if not hasattr(model, "predict_proba"):
        raise ValueError("The saved classifier does not support probability estimates.")
    return model, list(feature_names)

def decode_iq(raw: bytes, fmt: str) -> np.ndarray:
    if fmt == "Complex64 (float32 I/Q pairs)":
        if len(raw) % 8:
            raise ValueError("Complex64 input must contain complete 8-byte I/Q samples.")
        iq = np.frombuffer(raw, dtype=np.complex64)
    elif fmt == "Int16 (interleaved I,Q pairs)":
        if len(raw) % 4:
            raise ValueError("Int16 I/Q input must contain complete 4-byte I/Q pairs.")
        values = np.frombuffer(raw, dtype="<i2")
        iq = (values[0::2].astype(np.float32) + 1j * values[1::2].astype(np.float32)) / 32768.0
    else:
        raise ValueError("Choose a supported IQ format.")
    if iq.size < WINDOW_SIZE:
        raise ValueError(f"Capture has {iq.size:,} complex samples; at least {WINDOW_SIZE:,} are needed.")
    if not np.all(np.isfinite(iq[: min(iq.size, WINDOW_SIZE * 2)])):
        raise ValueError("Capture contains non-finite IQ samples.")
    return iq

def analyze_iq(iq: np.ndarray, sample_rate_hz: float, model, max_windows: int):
    available = len(iq) // WINDOW_SIZE
    count = min(available, max_windows)
    if count < 1:
        raise ValueError(f"Not enough samples for a {WINDOW_SIZE}-sample window.")
    features = []
    for index in range(count):
        start = index * WINDOW_SIZE
        window = iq[start:start + WINDOW_SIZE]
        features.append(extract_features(window, sample_rate_hz))
    frame = pd.DataFrame(features, columns=FEATURE_NAMES)
    probabilities = model.predict_proba(frame[FEATURE_NAMES])
    classes = list(model.classes_)
    predicted_indices = np.argmax(probabilities, axis=1)
    predictions = [classes[i] for i in predicted_indices]
    result = frame.copy()
    result.insert(0, "window", np.arange(count))
    result["predicted_class"] = predictions
    result["max_probability"] = np.max(probabilities, axis=1)
    for i, label in enumerate(classes):
        result[f"prob_{label}"] = probabilities[:, i]
    return result, probabilities, classes, count, available

def capture_from_pluto(
    uri: str,
    sample_rate_hz: int,
    center_frequency_hz: int,
    rf_bandwidth_hz: int,
    gain_mode: str,
    manual_gain_db: float,
    requested_samples: int,
) -> np.ndarray:
    """Capture RX IQ from an ADALM-Pluto through libiio/pylibiio."""
    try:
        if os.name == "nt":
            import ctypes.util
            scopy_dir = Path(r"C:\Program Files\Analog Devices\Scopy")
            if scopy_dir.is_dir():
                # Keep the DLL directory handle alive for the duration of the process.
                if not hasattr(capture_from_pluto, "_dll_dir_handle"):
                    capture_from_pluto._dll_dir_handle = os.add_dll_directory(str(scopy_dir))
                original_find_library = ctypes.util.find_library
                if not hasattr(capture_from_pluto, "_find_library_patched"):
                    def _find_iio_library(name):
                        if name in ("iio", "libiio.dll"):
                            candidate = scopy_dir / "libiio.dll"
                            if candidate.is_file():
                                return str(candidate)
                        return original_find_library(name)
                    ctypes.util.find_library = _find_iio_library
                    capture_from_pluto._find_library_patched = True
        import iio
    except Exception as exc:
        raise RuntimeError(
            "Could not import pylibiio/libiio. Check that pylibiio is installed and, on Windows, "
            "that C:\\Program Files\\Analog Devices\\Scopy\\libiio.dll is available."
        ) from exc

    context = None
    rx_buffer = None
    rx_device = None
    try:
        context = iio.Context(uri.strip() or "ip:192.168.2.1")
        phy = context.find_device("ad9361-phy")
        rx_device = context.find_device("cf-ad9361-lpc")
        if phy is None or rx_device is None:
            raise RuntimeError(
                "Could not find AD9361 RX devices. Check the Pluto USB/network connection and URI."
            )

        rx_lo = phy.find_channel("altvoltage0", True)
        rx_i = rx_device.find_channel("voltage0", False)
        rx_q = rx_device.find_channel("voltage1", False)
        if rx_lo is None or rx_i is None or rx_q is None:
            raise RuntimeError("Pluto RX LO/I/Q channels were not found.")

        rx_lo.attrs["frequency"].value = str(int(center_frequency_hz))
        rx_i.attrs["sampling_frequency"].value = str(int(sample_rate_hz))
        rx_i.attrs["rf_bandwidth"].value = str(int(rf_bandwidth_hz))
        rx_i.attrs["gain_control_mode"].value = gain_mode
        if gain_mode == "manual":
            rx_i.attrs["hardwaregain"].value = str(float(manual_gain_db))

        rx_i.enabled = True
        rx_q.enabled = True
        chunk_samples = min(262144, max(4096, int(requested_samples)))
        rx_buffer = iio.Buffer(rx_device, chunk_samples, False)
        chunks = []
        remaining = int(requested_samples)
        while remaining > 0:
            rx_buffer.refill()
            raw = rx_buffer.read()
            values = np.frombuffer(raw, dtype="<i2")
            if values.size < 2:
                raise RuntimeError("Pluto returned an empty or incomplete IQ buffer.")
            pairs = values[: values.size - (values.size % 2)].reshape(-1, 2)
            complex_chunk = (
                pairs[:, 0].astype(np.float32) + 1j * pairs[:, 1].astype(np.float32)
            ) / 2048.0
            take = min(remaining, len(complex_chunk))
            chunks.append(complex_chunk[:take])
            remaining -= take
        iq = np.concatenate(chunks).astype(np.complex64, copy=False)
        if iq.size < WINDOW_SIZE:
            raise RuntimeError(f"Pluto returned only {iq.size:,} samples.")
        return iq
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(
            f"Pluto capture failed: {exc}. Verify the URI, Pluto connection, and libiio setup."
        ) from exc
    finally:
        if rx_buffer is not None:
            try:
                rx_buffer.cancel()
            except Exception:
                pass
        if rx_device is not None:
            for name in ("voltage0", "voltage1"):
                try:
                    channel = rx_device.find_channel(name, False)
                    if channel is not None:
                        channel.enabled = False
                except Exception:
                    pass


def plot_theme(fig, height=340):
    is_light = bool(st.session_state.get("rf_light_theme", False))
    font_color = "#26364b" if is_light else "#c5d3e8"
    plot_bg = "rgba(255,255,255,.55)" if is_light else "rgba(8,16,28,.25)"
    grid_color = "rgba(60,80,110,.15)" if is_light else "rgba(110,140,175,.13)"
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=plot_bg,
        font=dict(family="DM Sans, sans-serif", color=font_color, size=12),
        margin=dict(l=18, r=18, t=48, b=18),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=font_color)),
        title=dict(font=dict(color=font_color)),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, tickfont=dict(color=font_color), title_font=dict(color=font_color)),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, tickfont=dict(color=font_color), title_font=dict(color=font_color)),
    )
    return fig

def spectrum_figure(iq: np.ndarray, sample_rate_hz: float):
    n = min(len(iq), 65536)
    x = iq[:n]
    if n < 2:
        raise ValueError("Not enough samples to draw a spectrum.")
    window = np.hanning(n)
    spec = np.fft.fftshift(np.fft.fft(x * window))
    power_db = 20 * np.log10(np.abs(spec) / max(np.max(np.abs(spec)), 1e-12) + 1e-12)
    freq_mhz = np.fft.fftshift(np.fft.fftfreq(n, d=1 / sample_rate_hz)) / 1e6
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq_mhz, y=power_db, mode="lines", name="Relative spectrum",
                             line=dict(color="#55d6d0", width=1.5), fill="tozeroy",
                             fillcolor="rgba(85,214,208,.08)"))
    fig.update_layout(title="Relative power spectrum", xaxis_title="Frequency offset (MHz)",
                      yaxis_title="Relative magnitude (dB)", yaxis=dict(range=[-90, 3]))
    return plot_theme(fig)

def empty_chart(title: str, message: str):
    fig = go.Figure()
    fig.add_annotation(text=message, x=.5, y=.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(color="#91a2bf", size=14))
    fig.update_layout(title=title, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return plot_theme(fig)

# Sidebar controls
with st.sidebar:
    st.markdown("## 📡 RFRecon")
    st.caption("RF signal reconnaissance")
    theme = st.toggle("Bright / light theme", value=False, help="Switch between dark and bright dashboard themes.")
    st.session_state["rf_light_theme"] = theme
    st.markdown("---")
    page = st.radio("Workspace", ["IQ Signal Analysis", "CSV Data Explorer", "About & limitations"], index=0)
    st.markdown("---")
    st.markdown("**Classifier**")
    st.markdown('<span class="pill">5 classes</span><span class="pill">4096 samples/window</span>', unsafe_allow_html=True)
    st.caption("Predictions are model outputs, not ground-truth verification.")


if theme:
    st.markdown(
        """
        <style>
        .stApp, html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"] { background: linear-gradient(145deg,#f4f7fb,#e8eef7) !important; color:#15243a !important; }
        [data-testid="stHeader"], header[data-testid="stHeader"] { background:#f4f7fb !important; }
        [data-testid="stSidebar"] { background:#e8eef7 !important; border-right:1px solid #d0dbea !important; }
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color:#15243a !important; }
        [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
        h1, h2, h3, h4, h5, h6, label, p, span, li, summary, small,
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
        [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
        [data-testid="stText"], [data-testid="stAlert"] *,
        [data-testid="stExpander"] *, [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label, [data-testid="stSelectbox"] label,
        [data-testid="stNumberInput"] label, [data-testid="stSlider"] label,
        [data-testid="stFileUploader"] section, [data-testid="stFileUploader"] small,
        [data-testid="stDataFrame"] *, [data-testid="stTable"] { color:#15243a !important; }
        input, textarea, [data-baseweb="select"] *, [data-baseweb="input"] *,
        [data-baseweb="textarea"] * { color:#15243a !important; }
        [data-testid="stMetric"] { background:#ffffff !important; border-color:#d7e0ed !important; }
        [data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] *,
        [data-testid="stMetricDelta"] * { color:#15243a !important; }
        button, [data-testid="stDownloadButton"] button { color:#15243a !important; }
        [data-testid="stMetricValue"] { color:#15243a !important; }
        .hero { background:linear-gradient(115deg,#ffffff,#e8f1fb) !important; border-color:#d0dbea !important; }
        .hero-title { color:#15243a !important; }
        .hero-copy, .section-note, .small-muted { color:#52647e !important; }
        .pill { background:#ffffff !important; color:#17465a !important; border-color:#b8cddd !important; }
        div.stButton > button, [data-testid="stDownloadButton"] button { background:#ffffff !important; color:#15243a !important; border:1px solid #b8c8dc !important; }
        div.stButton > button[kind="primary"] { background:#42c9c2 !important; color:#08202b !important; border:0 !important; }
        input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] > div { background:#ffffff !important; color:#15243a !important; border-color:#c5d2e2 !important; }
        [data-testid="stFileUploader"], [data-testid="stFileUploader"] section { background:#ffffff !important; color:#15243a !important; border-color:#b8c8dc !important; }
        [data-testid="stExpander"], [data-testid="stExpander"] details, [data-testid="stExpander"] summary { background:#ffffff !important; color:#15243a !important; border-color:#d0dbea !important; }
        [data-testid="stDataFrame"], [data-testid="stTable"] { color:#15243a !important; }
        .js-plotly-plot .plotly text, .js-plotly-plot .plotly .gtitle, .js-plotly-plot .plotly .legend text { fill:#26364b !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color:#eaf1ff !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">RF SIGNAL INTELLIGENCE · LOCAL WORKSPACE</div>
      <div class="hero-title">Understand the signal.<br/>See the spectrum.</div>
      <div class="hero-copy">Capture live from Pluto or load an IQ file / feature CSV to explore signal characteristics, model predictions, probabilities and interactive graphs.</div>
      <div style="margin-top:16px"><span class="pill">Interactive charts</span><span class="pill">Local processing</span><span class="pill">Exportable results</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "IQ Signal Analysis":
    st.subheader("Live capture from ADALM-Pluto")
    st.markdown(
        '<div class="section-note">Capture receive-only IQ directly from a connected PlutoSDR, then automatically run the same five-class analysis and graph pipeline. This requires a working local libiio connection.</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        with st.form("pluto_live_capture_form"):
            p1, p2, p3 = st.columns(3)
            with p1:
                pluto_uri = st.text_input("Pluto URI", value="ip:192.168.2.1", help="Use ip:192.168.2.1 for the usual USB network address, or the URI configured for your device.")
                live_rate_msps = st.number_input("RX sample rate (MS/s)", min_value=1.0, max_value=61.44, value=20.0, step=1.0)
            with p2:
                live_center_mhz = st.number_input("RX center frequency (MHz)", min_value=70.0, max_value=6000.0, value=2437.0, step=1.0)
                live_bandwidth_mhz = st.number_input("RX RF bandwidth (MHz)", min_value=0.2, max_value=56.0, value=20.0, step=1.0)
            with p3:
                live_duration = st.slider("Capture duration (seconds)", min_value=0.1, max_value=2.0, value=0.25, step=0.1)
                live_gain_mode = st.selectbox("RX gain mode", ["slow_attack", "fast_attack", "manual"])
                live_gain_db = st.number_input("Manual RX gain (dB)", min_value=0.0, max_value=73.0, value=30.0, step=1.0, disabled=live_gain_mode != "manual")
            st.caption("The capture is capped at 2,000,000 complex samples to keep the browser session responsive. Receive settings depend on hardware/firmware support.")
            live_capture_btn = st.form_submit_button("Capture from Pluto and analyze", type="primary", use_container_width=True)

    if live_capture_btn:
        try:
            if live_bandwidth_mhz > live_rate_msps:
                st.warning("RF bandwidth exceeds sample rate; Pluto hardware may reject this setting.")
            requested_samples = min(2_000_000, max(WINDOW_SIZE, int(live_rate_msps * 1e6 * live_duration)))
            with st.spinner("Connecting to Pluto, capturing IQ, and running the five-class model…"):
                live_iq = capture_from_pluto(
                    pluto_uri,
                    int(live_rate_msps * 1e6),
                    int(live_center_mhz * 1e6),
                    int(live_bandwidth_mhz * 1e6),
                    live_gain_mode,
                    float(live_gain_db),
                    requested_samples,
                )
                live_model, _ = load_model()
                live_rate_hz = live_rate_msps * 1e6
                live_results, live_probabilities, live_classes, live_used, live_available = analyze_iq(
                    live_iq, live_rate_hz, live_model, MAX_ANALYSIS_WINDOWS
                )
                st.session_state["rf_results"] = live_results
                st.session_state["rf_probs"] = live_probabilities
                st.session_state["rf_classes"] = live_classes
                st.session_state["rf_source"] = f"Live Pluto RX · {pluto_uri} · {live_center_mhz:g} MHz"
                st.session_state["rf_sample_rate"] = live_rate_hz
                st.session_state["rf_iq_preview"] = np.asarray(live_iq[:min(len(live_iq), 65536)], dtype=np.complex64).copy()
                st.session_state["rf_used_windows"] = live_used
                st.session_state["rf_available_windows"] = live_available
                st.session_state["rf_live_capture_metadata"] = {
                    "URI": pluto_uri,
                    "Center frequency (MHz)": live_center_mhz,
                    "Sample rate (MS/s)": live_rate_msps,
                    "RF bandwidth (MHz)": live_bandwidth_mhz,
                    "Gain mode": live_gain_mode,
                    "Requested samples": requested_samples,
                    "Captured samples": len(live_iq),
                }
            st.success(f"Captured {len(live_iq):,} complex samples and analyzed {live_used:,} windows.")
        except Exception as exc:
            st.error(f"Live Pluto capture failed: {exc}")

    if "rf_live_capture_metadata" in st.session_state:
        with st.expander("Last live-capture settings"):
            st.json(st.session_state["rf_live_capture_metadata"])

    st.divider()
    st.subheader("Analyze a saved IQ capture")
    st.markdown('<div class="section-note">Supported raw formats: complex64 float32 I/Q pairs, or interleaved int16 I/Q pairs. Select the format that matches your capture.</div>', unsafe_allow_html=True)
    with st.container(border=True):
        left, right, third = st.columns([1.2, 1, 1])
        with left:
            sample_rate_msps = st.number_input("Sample rate (MS/s)", min_value=0.001, max_value=2000.0, value=20.0, step=1.0, format="%.3f")
        with right:
            iq_format = st.selectbox("IQ data format", ["Complex64 (float32 I/Q pairs)", "Int16 (interleaved I,Q pairs)"])
        with third:
            max_windows = st.slider("Maximum windows", min_value=1, max_value=1500, value=256, step=1)
        uploaded = st.file_uploader("Upload an IQ capture", type=["dat", "bin", "cfile", "iq", "raw"], help="Browser uploads are limited by .streamlit/config.toml. For multi-GB recordings, enter the local path below for reliable analysis without uploading the bytes.")
        local_path = st.text_input("Or enter a local capture path (for large files)", placeholder=r"real_dataset\real_noise_01.dat")
        analyze_btn = st.button("Analyze IQ capture", type="primary", use_container_width=True)

    if analyze_btn:
        try:
            if sample_rate_msps <= 0:
                raise ValueError("Sample rate must be a positive number.")
            use_local_path = bool(local_path.strip())
            if not use_local_path and uploaded is None:
                raise ValueError("Choose a file or enter its local file path. For very large captures, the local path is the most reliable option.")
            with st.spinner("Loading five-class model and extracting RF features…"):
                model, feature_names = load_model()
                # A non-empty local path takes priority. This also bypasses browser upload limits.
                if use_local_path:
                    path = Path(local_path.strip().strip('"')).expanduser()
                    if not path.is_absolute():
                        path = APP_DIR / path
                    path = path.resolve()
                    if not path.is_file():
                        raise FileNotFoundError(f"Local capture not found: {path}")
                    source_name = str(path)
                    if iq_format.startswith("Complex64"):
                        if path.stat().st_size % 8:
                            raise ValueError("File size is not divisible by 8 bytes; check the selected IQ format.")
                        iq = np.memmap(path, dtype=np.complex64, mode="r")
                    else:
                        if path.stat().st_size % 4:
                            raise ValueError("File size is not divisible by 4 bytes; check the selected IQ format.")
                        values = np.memmap(path, dtype="<i2", mode="r")
                        iq = (values[0::2].astype(np.float32) + 1j * values[1::2].astype(np.float32)) / 32768.0
                else:
                    raw = uploaded.getvalue()
                    source_name = uploaded.name
                    iq = decode_iq(raw, iq_format)
                if len(iq) < WINDOW_SIZE:
                    raise ValueError(f"Capture must contain at least {WINDOW_SIZE:,} complex samples.")
                if not np.all(np.isfinite(iq[:WINDOW_SIZE])):
                    raise ValueError("The first IQ window contains non-finite values.")
                sample_rate_hz = float(sample_rate_msps) * 1e6
                results, probabilities, classes, used, available = analyze_iq(iq, sample_rate_hz, model, max_windows)
                st.session_state["rf_results"] = results
                st.session_state["rf_probs"] = probabilities
                st.session_state["rf_classes"] = classes
                st.session_state["rf_source"] = source_name
                st.session_state["rf_sample_rate"] = sample_rate_hz
                st.session_state["rf_iq_preview"] = np.asarray(iq[:min(len(iq), 65536)], dtype=np.complex64).copy()
                st.session_state["rf_used_windows"] = used
                st.session_state["rf_available_windows"] = available
        except Exception as exc:
            st.error(f"Analysis could not be completed: {exc}")

    if "rf_results" in st.session_state:
        results = st.session_state["rf_results"]
        probabilities = st.session_state["rf_probs"]
        classes = st.session_state["rf_classes"]
        source_name = st.session_state["rf_source"]
        sample_rate_hz = st.session_state["rf_sample_rate"]
        used = st.session_state["rf_used_windows"]
        available = st.session_state["rf_available_windows"]
        counts = results["predicted_class"].value_counts()
        dominant = counts.index[0]
        mean_conf = float(results["max_probability"].mean())
        st.markdown(f"#### Results · `{source_name}`")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dominant predicted class", pretty_class(dominant))
        c2.metric("Mean max probability", f"{mean_conf:.1%}")
        c3.metric("Windows analyzed", f"{used:,}")
        c4.metric("Sample rate entered", f"{sample_rate_hz/1e6:g} MS/s")
        st.caption(f"{available:,} complete windows are available; {used:,} were analyzed. Sample rate is user-supplied and is not independently verified.")
        chart1, chart2 = st.columns([1.3, 1])
        with chart1:
            st.plotly_chart(spectrum_figure(st.session_state["rf_iq_preview"], sample_rate_hz), use_container_width=True)
        with chart2:
            class_counts = counts.rename(index=pretty_class).rename_axis("Predicted class").reset_index(name="Windows")
            fig = px.bar(class_counts, x="Predicted class", y="Windows", color="Predicted class",
                         title="Predictions across windows", color_discrete_sequence=["#55d6d0", "#7c8cff", "#f2b66d", "#d98be8"])
            st.plotly_chart(plot_theme(fig), use_container_width=True)
        st.subheader("Extracted RF features — all 10")
        st.caption("Each point is one analyzed 4096-sample window. Feature scales differ; use these plots to inspect signal characteristics, not as protocol proof.")
        feature_cols = list(FEATURE_NAMES)
        feature_labels = {
            "mean_power": "Mean power",
            "rms": "RMS amplitude",
            "peak_amplitude": "Peak amplitude",
            "variance": "Variance",
            "peak_frequency_normalized": "Normalized peak frequency",
            "spectral_centroid_normalized": "Normalized spectral centroid",
            "spectral_bandwidth_normalized": "Normalized spectral bandwidth",
            "occupied_bandwidth_normalized": "Normalized occupied bandwidth",
            "spectral_flatness": "Spectral flatness",
            "peak_to_average_ratio": "Peak-to-average ratio",
        }
        feature_choice = st.selectbox(
            "Feature to inspect",
            feature_cols,
            format_func=lambda name: feature_labels.get(name, name.replace("_", " ").title()),
            key="rf_feature_choice",
        )
        fleft, fright = st.columns(2)
        with fleft:
            feature_line = px.line(
                results.head(1500), x="window", y=feature_choice,
                title=f"{feature_labels.get(feature_choice, feature_choice)} by window",
            )
            st.plotly_chart(plot_theme(feature_line), use_container_width=True)
        with fright:
            feature_hist = px.histogram(
                results, x=feature_choice, nbins=40,
                title=f"Distribution: {feature_labels.get(feature_choice, feature_choice)}",
            )
            st.plotly_chart(plot_theme(feature_hist), use_container_width=True)

        with st.expander("View all 10 feature trends", expanded=True):
            feature_rows = st.columns(2)
            for i, feat in enumerate(feature_cols):
                fig_feature = px.line(
                    results.head(1500), x="window", y=feat,
                    title=feature_labels.get(feat, feat.replace("_", " ").title()),
                )
                with feature_rows[i % 2]:
                    st.plotly_chart(plot_theme(fig_feature, 260), use_container_width=True)
        st.download_button(
            "Download extracted 10-feature table (CSV)",
            results[["window"] + feature_cols].to_csv(index=False).encode("utf-8"),
            file_name="rfrecon_extracted_features.csv",
            mime="text/csv",
        )

        st.subheader("Average model probabilities")
        mean_probs = probabilities.mean(axis=0)
        prob_df = pd.DataFrame({"Class": [pretty_class(c) for c in classes], "Mean probability": mean_probs})
        fig = px.bar(prob_df, x="Class", y="Mean probability", range_y=[0, 1],
                     color="Mean probability", color_continuous_scale=["#243b55", "#55d6d0"])
        st.plotly_chart(plot_theme(fig, 300), use_container_width=True)
        st.subheader("Window-level results")
        display = results[["window", "predicted_class", "max_probability"]].copy()
        display["predicted_class"] = display["predicted_class"].map(pretty_class)
        display["max_probability"] = (display["max_probability"] * 100).round(2)
        display = display.rename(columns={"window": "Window", "predicted_class": "Predicted class", "max_probability": "Max probability (%)"})
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button("Download window predictions (CSV)", results.to_csv(index=False).encode("utf-8"),
                           file_name="rfrecon_window_predictions.csv", mime="text/csv")
        st.info("Interpretation: probability is the model's confidence-like output, not a calibrated guarantee. The spectrum is relative and uses frequency offset, not absolute RF frequency.")

elif page == "CSV Data Explorer":
    st.subheader("Explore a feature or results CSV")
    st.markdown('<div class="section-note">Upload a CSV to view summary statistics, choose columns, and generate interactive graphs. This page explores the supplied table; it does not infer RF labels unless you run the IQ classifier above.</div>', unsafe_allow_html=True)
    csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_explorer")
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                st.warning("The CSV has no data rows.")
            else:
                st.caption(f"{len(df):,} rows × {len(df.columns):,} columns")
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows", f"{len(df):,}")
                m2.metric("Columns", f"{len(df.columns):,}")
                m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
                st.dataframe(df.head(500), use_container_width=True, hide_index=True)
                numeric_cols = list(df.select_dtypes(include="number").columns)
                if numeric_cols:
                    xcol = st.selectbox("X-axis", list(df.columns), index=0, key="csv_x")
                    ycol = st.selectbox("Y-axis", numeric_cols, index=min(0, len(numeric_cols)-1), key="csv_y")
                    chart_type = st.selectbox("Chart type", ["Line", "Scatter", "Histogram", "Box"], key="csv_type")
                    if chart_type == "Line":
                        fig = px.line(df.head(10000), x=xcol, y=ycol, title=f"{ycol} vs {xcol}")
                    elif chart_type == "Scatter":
                        fig = px.scatter(df.head(10000), x=xcol, y=ycol, title=f"{ycol} vs {xcol}", opacity=.75)
                    elif chart_type == "Histogram":
                        fig = px.histogram(df, x=ycol, title=f"Distribution of {ycol}")
                    else:
                        fig = px.box(df, y=ycol, title=f"Spread of {ycol}")
                    st.plotly_chart(plot_theme(fig), use_container_width=True)
                    if len(numeric_cols) >= 2:
                        st.subheader("Correlation heatmap")
                        corr = df[numeric_cols].corr(numeric_only=True)
                        heat = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Tealgrn",
                                         title="Numeric feature correlations")
                        st.plotly_chart(plot_theme(heat, 420), use_container_width=True)
                else:
                    st.info("No numeric columns were found to graph.")
                st.download_button("Download loaded CSV", df.to_csv(index=False).encode("utf-8"),
                                   file_name="rfrecon_csv_export.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    else:
        st.plotly_chart(empty_chart("CSV visualizer", "Upload a CSV to display its table and interactive graphs."), use_container_width=True)

else:
    st.subheader("About this dashboard")
    st.markdown(
        """
        **RFRecon Web Dashboard** is a local browser interface for the existing five-class Random Forest model.

        - **IQ Signal Analysis:** accepts raw complex IQ captures, extracts the same ten features, predicts per-window classes, plots a relative spectrum and exports results.
        - **CSV Data Explorer:** previews CSV data and creates line, scatter, histogram, box and correlation charts.
        - **Model artifacts:** uses `rfrecon_five_class_synthetic_rf.pkl`, `rfrecon_five_class_synthetic_features.pkl`, and `rfrecon_consistent_features.py` from this project folder.
        - **Feature views:** plots trends and distributions for all ten extracted RF features and exports the feature table.

        ### Important limitations
        - The sample rate and IQ storage format must be supplied correctly; the app cannot infer them reliably from raw bytes.
        - Model predictions are not ground truth.
        - The current web classifier is **five-class**: Bluetooth-like, cellular-like, noise, Wi-Fi-like, and Zigbee-like.
        - The five-class model was trained/evaluated on synthetic data. Zigbee-like predictions are experimental, not verified real-world Zigbee identification.
        - The spectrum is relative frequency offset, not absolute RF tuning.
        - Live Pluto capture is integrated through pylibiio in this interface, but hardware operation must be tested on the connected device before claiming it is verified. Saved IQ files can also be analyzed locally.
        """
    )
    st.markdown("---")
    st.caption("RFRecon · Local-first analysis · No data is intentionally sent to an external service by this app.")
