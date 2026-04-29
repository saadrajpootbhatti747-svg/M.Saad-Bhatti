import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as plotly_go
from plotly.subplots import make_subplots
import neurokit2 as nk
import scipy.signal as signal
import base64
import os
import glob

# --- Page Configuration ---
st.set_page_config(
    page_title="ECG & HRV Analysis Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
        border-left: 4px solid #f38ba8;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #f38ba8;
    }
    .metric-label {
        font-size: 14px;
        color: #bac2de;
        margin-top: 5px;
    }
    .section-header {
        font-size: 22px;
        font-weight: bold;
        color: #89b4fa;
        margin-top: 30px;
        margin-bottom: 10px;
        border-bottom: 1px solid #313244;
        padding-bottom: 5px;
    }
    .stApp {
        background-color: #11111b;
        color: #cdd6f4;
    }
    header {visibility: hidden;}
    [data-testid="stSidebar"] {
        background-color: #181825;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def load_synthetic_data(duration=60, sampling_rate=250, noise=0.1, heart_rate=70):
    """Generate synthetic ECG data using neurokit2."""
    ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, noise=noise, heart_rate=heart_rate)
    df = pd.DataFrame({"Time": np.arange(len(ecg)) / sampling_rate, "ECG": ecg})
    return df

@st.cache_data
def process_ecg(ecg_signal, sampling_rate, filter_lowcut=0.5, filter_highcut=40.0, filter_order=3):
    """Process ECG: Bandpass Filter + Pan-Tompkins Peak Detection."""
    # 1. Custom Bandpass filter for visual display
    nyq = 0.5 * sampling_rate
    low = filter_lowcut / nyq
    high = filter_highcut / nyq
    b, a = signal.butter(filter_order, [low, high], btype='band')
    visual_filtered_ecg = signal.filtfilt(b, a, ecg_signal)
    
    # 2. Robust R-peak detection (Pan-Tompkins via neurokit2)
    # ecg_clean handles its own optimal bandpass for peak detection
    nk_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method="pantompkins1985")
    _, info = nk.ecg_peaks(nk_cleaned, sampling_rate=sampling_rate, method="pantompkins1985")
    rpeaks = info["ECG_R_Peaks"]
    
    return visual_filtered_ecg, rpeaks

def calculate_hrv(rpeaks, sampling_rate):
    """Calculate Time, Frequency, and Non-Linear HRV metrics."""
    # Convert peaks to RR intervals in milliseconds
    rr_intervals = np.diff(rpeaks) / sampling_rate * 1000
    
    if len(rr_intervals) < 2:
        return None, None, None, rr_intervals, None
    
    # Time Domain
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals, ddof=1)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    
    time_metrics = {
        "Mean RR (ms)": mean_rr,
        "SDNN (ms)": sdnn,
        "RMSSD (ms)": rmssd
    }
    
    # Frequency Domain (Welch's Method)
    time_rr = np.cumsum(rr_intervals) / 1000.0  # Time in seconds
    freq_resample = 4.0 # Hz
    time_resample = np.arange(time_rr[0], time_rr[-1], 1/freq_resample)
    
    # Linear interpolation
    rr_resampled = np.interp(time_resample, time_rr, rr_intervals)
    
    # Welch's method
    f, pxx = signal.welch(rr_resampled, fs=freq_resample, window='hann', nperseg=min(256, len(rr_resampled)), noverlap=min(128, len(rr_resampled)//2))
    
    # Calculate Power in bands
    vlf_band = (0.0033, 0.04)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    
    lf_idx = np.logical_and(f >= lf_band[0], f < lf_band[1])
    hf_idx = np.logical_and(f >= hf_band[0], f < hf_band[1])
    
    try:
        trapz_func = np.trapezoid
    except AttributeError:
        trapz_func = np.trapz
        
    lf_power = trapz_func(pxx[lf_idx], f[lf_idx]) if np.any(lf_idx) else 0
    hf_power = trapz_func(pxx[hf_idx], f[hf_idx]) if np.any(hf_idx) else 0
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
    
    freq_metrics = {
        "LF Power (ms²)": lf_power,
        "HF Power (ms²)": hf_power,
        "LF/HF Ratio": lf_hf_ratio
    }
    
    # Non-linear Domain
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]
    
    sd1 = np.sqrt(np.std(rr_n1 - rr_n) ** 2 / 2) if len(rr_n) > 1 else 0
    sd2 = np.sqrt(np.std(rr_n1 + rr_n) ** 2 / 2) if len(rr_n) > 1 else 0
    
    # Sample Entropy
    try:
        sampen, _ = nk.entropy_sample(rr_intervals)
    except:
        sampen = np.nan
        
    nl_metrics = {
        "SD1 (ms)": sd1,
        "SD2 (ms)": sd2,
        "Sample Entropy": sampen
    }
    
    return time_metrics, freq_metrics, nl_metrics, rr_intervals, (f, pxx)

def create_kpi_card(title, value, unit=""):
    """HTML for a KPI card."""
    if pd.isna(value):
        val_str = "N/A"
    else:
        val_str = f"{value:.2f} {unit}"
        
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{val_str}</div>
        <div class="metric-label">{title}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.markdown("Upload ECG data or simulate a signal.")

data_source = st.sidebar.radio("Data Source", ["Select Local File", "Upload File", "Synthetic Data"], index=0)

sampling_rate = st.sidebar.number_input("Sampling Rate (Hz)", min_value=50, max_value=2000, value=250, step=50)

def load_data_file(file_path_or_buffer, filename):
    try:
        if filename.endswith('.dat') or filename.endswith('.txt'):
            df_temp = pd.read_csv(file_path_or_buffer, sep=r'\s+|,', engine='python', header=None)
        else:
            df_temp = pd.read_csv(file_path_or_buffer)
            
        # Drop fully NaN columns
        df_temp = df_temp.dropna(axis=1, how='all')
        return df_temp
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        return None

df_raw = None
if data_source == "Select Local File":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_files = []
    for ext in ('*.csv', '*.txt', '*.dat'):
        local_files.extend(glob.glob(os.path.join(script_dir, ext)))
    
    if not local_files:
        st.sidebar.warning(f"No ECG files (*.csv, *.txt, *.dat) found in {script_dir}")
    else:
        file_options = {os.path.basename(f): f for f in local_files}
        selected_filename = st.sidebar.selectbox("Choose an ECG File:", list(file_options.keys()))
        df_raw = load_data_file(file_options[selected_filename], selected_filename)

elif data_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload Data (.csv, .txt, .dat)", type=["csv", "txt", "dat"])
    if uploaded_file is not None:
        df_raw = load_data_file(uploaded_file, uploaded_file.name)

df = None
if data_source in ["Select Local File", "Upload File"] and df_raw is not None:
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.sidebar.error("No numerical columns found in the data.")
    else:
        # Default to the last numeric column or a column named 'ECG'
        default_idx = len(numeric_cols) - 1
        for i, col in enumerate(numeric_cols):
            if str(col).upper() == 'ECG':
                default_idx = i
                break
                
        ecg_col = st.sidebar.selectbox("Select ECG Signal Column:", numeric_cols, index=default_idx)
        
        df = pd.DataFrame()
        # Drop NaNs to prevent filter errors
        clean_signal = df_raw[ecg_col].dropna().values
        df["ECG"] = clean_signal
        df["Time"] = np.arange(len(df)) / sampling_rate

elif data_source == "Synthetic Data":
    st.sidebar.subheader("Simulation Parameters")
    duration = st.sidebar.slider("Duration (s)", 30, 300, 60)
    heart_rate = st.sidebar.slider("Heart Rate (BPM)", 40, 150, 70)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.1)
    df = load_synthetic_data(duration, sampling_rate, noise_level, heart_rate)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Settings (Bandpass)")
filter_lowcut = st.sidebar.number_input("Lowcut Freq (Hz)", value=0.5, step=0.1)
filter_highcut = st.sidebar.number_input("Highcut Freq (Hz)", value=40.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Display Options")
show_raw = st.sidebar.checkbox("Show Raw ECG Signal", value=True)

# --- Main Dashboard ---
st.title("❤️ Biomedical ECG & HRV Analysis Dashboard")
st.markdown("A clinically meaningful interface visualizing sympathetic and parasympathetic activity through advanced ECG signal processing and HRV analysis.")

if df is not None:
    ecg_signal = df["ECG"].values
    time_array = df["Time"].values
    
    with st.spinner("Processing Signal & Extracting Features..."):
        filtered_ecg, rpeaks = process_ecg(ecg_signal, sampling_rate, filter_lowcut, filter_highcut)
        time_m, freq_m, nl_m, rr_ints, psd_data = calculate_hrv(rpeaks, sampling_rate)
        
        if time_m is not None:
            all_metrics = {**time_m, **freq_m, **nl_m}
            metrics_df = pd.DataFrame([all_metrics])
        else:
            metrics_df = pd.DataFrame()

    # =========================================================================
    # SECTION 1: ECG SIGNAL VISUALIZATION (TOP)
    # =========================================================================
    st.markdown('<div class="section-header">📈 1. ECG Signal Processing & R-Peak Detection</div>', unsafe_allow_html=True)
    
    fig_ecg = make_subplots(rows=2 if show_raw else 1, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    row_filtered = 2 if show_raw else 1
    
    if show_raw:
        fig_ecg.add_trace(plotly_go.Scatter(x=time_array, y=ecg_signal, name="Raw ECG", line=dict(color='gray', width=1.5)), row=1, col=1)
        fig_ecg.update_yaxes(title_text="Raw Amp (mV)", row=1, col=1)
        
    fig_ecg.add_trace(plotly_go.Scatter(x=time_array, y=filtered_ecg, name="Filtered ECG", line=dict(color='#89b4fa', width=1.5)), row=row_filtered, col=1)
    fig_ecg.add_trace(plotly_go.Scatter(x=time_array[rpeaks], y=filtered_ecg[rpeaks], mode='markers', name="Detected R-Peaks", marker=dict(color='#f38ba8', size=8, symbol='cross')), row=row_filtered, col=1)
    
    fig_ecg.update_layout(
        height=500 if show_raw else 350, 
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cdd6f4'),
        hovermode="x unified",
        xaxis_title="Time (s)" if not show_raw else "",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    if show_raw:
        fig_ecg.update_xaxes(title_text="Time (s)", row=2, col=1)
    
    fig_ecg.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#313244')
    fig_ecg.update_yaxes(title_text="Filtered Amp (mV)", showgrid=True, gridwidth=1, gridcolor='#313244', row=row_filtered, col=1)
    st.plotly_chart(fig_ecg, use_container_width=True)

    if time_m is not None:
        # =========================================================================
        # SECTION 2: HRV METRICS (MIDDLE)
        # =========================================================================
        st.markdown('<div class="section-header">⏱️ 2. Heart Rate Variability (Time & Frequency Domain)</div>', unsafe_allow_html=True)
        
        # KPI Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: create_kpi_card("Mean RR", time_m["Mean RR (ms)"], "ms")
        with col2: create_kpi_card("SDNN", time_m["SDNN (ms)"], "ms")
        with col3: create_kpi_card("RMSSD", time_m["RMSSD (ms)"], "ms")
        with col4: create_kpi_card("LF Power", freq_m["LF Power (ms²)"], "ms²")
        with col5: create_kpi_card("LF/HF Ratio", freq_m["LF/HF Ratio"], "")
        
        # Tachogram & PSD
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.markdown("**RR Interval Tachogram**")
            fig_rr = plotly_go.Figure()
            fig_rr.add_trace(plotly_go.Scatter(x=np.arange(len(rr_ints)), y=rr_ints, mode='lines+markers', name='RR Intervals', line=dict(color='#a6e3a1', width=1.5), marker=dict(size=4, color='#a6e3a1')))
            fig_rr.update_layout(
                height=350,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#cdd6f4'),
                xaxis_title="Beat Number", yaxis_title="RR Interval (ms)",
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig_rr.update_xaxes(showgrid=True, gridcolor='#313244')
            fig_rr.update_yaxes(showgrid=True, gridcolor='#313244')
            st.plotly_chart(fig_rr, use_container_width=True)
            
        with col_plot2:
            st.markdown("**Power Spectral Density (Welch)**")
            f, pxx = psd_data
            fig_psd = plotly_go.Figure()
            
            # Full Spectrum
            fig_psd.add_trace(plotly_go.Scatter(x=f, y=pxx, mode='lines', name='PSD', line=dict(color='white', width=1)))
            
            # LF Band Shading (0.04 - 0.15 Hz)
            lf_idx = np.logical_and(f >= 0.04, f < 0.15)
            fig_psd.add_trace(plotly_go.Scatter(x=f[lf_idx], y=pxx[lf_idx], fill='tozeroy', mode='none', name='LF Band (Sympathetic)', fillcolor='rgba(137, 180, 250, 0.4)'))
            
            # HF Band Shading (0.15 - 0.4 Hz)
            hf_idx = np.logical_and(f >= 0.15, f < 0.4)
            fig_psd.add_trace(plotly_go.Scatter(x=f[hf_idx], y=pxx[hf_idx], fill='tozeroy', mode='none', name='HF Band (Parasympathetic)', fillcolor='rgba(166, 227, 161, 0.4)'))
            
            fig_psd.update_layout(
                height=350,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#cdd6f4'),
                xaxis_title="Frequency (Hz)", yaxis_title="Power (ms²/Hz)",
                xaxis=dict(range=[0, 0.5]),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(0,0,0,0.5)')
            )
            fig_psd.update_xaxes(showgrid=True, gridcolor='#313244')
            fig_psd.update_yaxes(showgrid=True, gridcolor='#313244')
            st.plotly_chart(fig_psd, use_container_width=True)

        # =========================================================================
        # SECTION 3: NON-LINEAR HRV (BOTTOM)
        # =========================================================================
        st.markdown('<div class="section-header">🌀 3. Non-Linear HRV Analysis</div>', unsafe_allow_html=True)
        
        col_nl1, col_nl2, col_nl3 = st.columns([1, 1, 2])
        
        with col_nl1:
            create_kpi_card("SD1 (ms)", nl_m["SD1 (ms)"], "")
            st.caption("Reflects short-term HRV (Parasympathetic activity).")
        with col_nl2:
            create_kpi_card("SD2 (ms)", nl_m["SD2 (ms)"], "")
            st.caption("Reflects long-term HRV (Sympathetic & Parasympathetic).")
            create_kpi_card("Sample Entropy", nl_m["Sample Entropy"], "")
            st.caption("Measures signal complexity and unpredictability.")
            
        with col_nl3:
            st.markdown("**Poincaré Plot (RRn vs RRn+1)**")
            rr_n = rr_ints[:-1]
            rr_n1 = rr_ints[1:]
            
            fig_poin = plotly_go.Figure()
            fig_poin.add_trace(plotly_go.Scatter(x=rr_n, y=rr_n1, mode='markers', name='RR Pairs', marker=dict(color='#cba6f7', size=6, opacity=0.7)))
            
            # Calculate Identity line
            if len(rr_n) > 0:
                min_rr = min(min(rr_n), min(rr_n1)) - 50
                max_rr = max(max(rr_n), max(rr_n1)) + 50
                fig_poin.add_trace(plotly_go.Scatter(x=[min_rr, max_rr], y=[min_rr, max_rr], mode='lines', name='Identity Line', line=dict(color='gray', dash='dash', width=1)))
            
            fig_poin.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#cdd6f4'),
                xaxis_title="RRn (ms)", yaxis_title="RRn+1 (ms)",
                xaxis=dict(scaleanchor="y", scaleratio=1), # Make it square
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig_poin.update_xaxes(showgrid=True, gridcolor='#313244')
            fig_poin.update_yaxes(showgrid=True, gridcolor='#313244')
            st.plotly_chart(fig_poin, use_container_width=True)

        # --- Export Options ---
        st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)
        csv = metrics_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="hrv_metrics.csv" style="padding: 10px 20px; background-color: #89b4fa; color: #11111b; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">Download HRV Metrics (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.caption("To export plots as images, hover over any chart and click the camera icon 📷 in the top right corner.")
    else:
        st.warning("Not enough peaks detected to calculate HRV metrics. Try adjusting the filter settings or simulation duration.")
else:
    st.info("Please upload an ECG CSV file or use the synthetic data generator in the sidebar.")
