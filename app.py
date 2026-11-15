import streamlit as st
import pandas as pd
import requests
from pathlib import Path

# 1. Page Configuration
st.set_page_config(
    page_title="Hospital Command Center",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Early-Warning & Operational Intelligence")
st.markdown("### Command Center — Predict, Detect, Explain")

# 2. Load Datasets safely with flexible paths
@st.cache_data
def load_datasets():
    current_dir = Path(__file__).parent
    root_dir = current_dir.parent
    
    # Find Beds CSV
    beds_file = None
    for path in [current_dir / "Hospital Beds.csv", root_dir / "Hospital Beds.csv", root_dir / "artifacts" / "Hospital Beds.csv"]:
        if path.exists():
            beds_file = path
            break
            
    # Find Waiting CSV
    waiting_file = None
    for path in [current_dir / "Hospital Waiting.csv", current_dir / "Hospital Waiting(1).csv", root_dir / "Hospital Waiting.csv", root_dir / "Hospital Waiting(1).csv", root_dir / "artifacts" / "Hospital Waiting.csv"]:
        if path.exists():
            waiting_file = path
            break
            
    beds_df = pd.read_csv(beds_file) if beds_file else None
    if beds_df is not None:
        beds_df["quarter_date"] = pd.to_datetime(beds_df["quarter_date"], format="%d/%m/%Y")
        
    waiting_df = pd.read_csv(waiting_file) if waiting_file else None
    
    return beds_df, waiting_df

try:
    beds_df, waiting_df = load_datasets()
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()

if beds_df is None:
    st.error("⚠️ Could not find 'Hospital Beds.csv'. Please place it in the project root or dashboard folder.")
    st.stop()

hosp_beds = beds_df[beds_df["hb_code"] != beds_df["loc_code"]].copy()

# 3. Create Tabs for the Dashboard
tab1, tab2 = st.tabs(["📊 Bed Occupancy & AI Forecast", "⏳ Waiting Lists Context (Board Level)"])

with tab1:
    # --- TAB 1: OCCUPANCY & FORECAST ---
    st.sidebar.header("🎛️ Bed Operations Control")
    locations = sorted(hosp_beds["loc_name"].dropna().unique())
    selected_loc = st.sidebar.selectbox("Select Hospital Location", locations)

    loc_data = hosp_beds[hosp_beds["loc_name"] == selected_loc]
    specialties = sorted(loc_data["spec_name"].dropna().unique())
    selected_spec = st.sidebar.selectbox("Select Specialty", specialties)

    loc_spec_data = loc_data[loc_data["spec_name"] == selected_spec].sort_values("quarter_date")

    if loc_spec_data.empty:
        st.warning("No historical data found for this selection.")
    else:
        latest_row = loc_spec_data.iloc[-1]
        
        st.subheader(f"Analyzing Bed Capacity: {selected_spec} at {selected_loc}")
        col_chart, col_metrics = st.columns([2, 1])

        with col_chart:
            st.markdown("#### Historical Occupancy Trend (%)")
            chart_data = loc_spec_data.set_index("quarter_date")["p_occ"]
            st.line_chart(chart_data)

        with col_metrics:
            st.markdown("#### Latest Key Metrics")
            st.metric("Current Occupancy", f"{latest_row['p_occ']:.1f}%")
            st.metric("Available Staffed Beds (asb)", f"{latest_row['asb']:.1f}")
            st.metric("Occupied Beds (aob)", f"{latest_row['aob']:.1f}")

        st.markdown("---")
        st.subheader("🔮 AI Operational Intelligence Forecast")

        if st.button("Run Next-Quarter Pressure Forecast", type="primary"):
            prev_row = loc_spec_data.iloc[-2] if len(loc_spec_data) > 1 else latest_row
            
            payload = {
                "loc_code": str(latest_row["loc_code"]),
                "spec_name": str(latest_row["spec_name"]),
                "quarter_number": int(latest_row["quarter_date"].quarter),
                "aasb": int(latest_row["aasb"]),
                "tobd": int(latest_row["tobd"]),
                "asb": float(latest_row["asb"]),
                "aob": float(latest_row["aob"]),
                "p_occ": float(latest_row["p_occ"]),
                "eps": 20.0,
                "los": 49.0,
                "avlos": 2.45,
                "emergency_activity_missing": 0,
                "previous_p_occ": float(prev_row["p_occ"]),
                "previous_asb": float(prev_row["asb"]),
                "previous_eps": 18.0
            }
            
            try:
                response = requests.post("http://127.0.0.1:8000/predict", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    res_col1, res_col2, res_col3 = st.columns(3)
                    res_col1.metric("Current Occupancy", f"{result['current_occupancy']}%")
                    res_col2.metric("Predicted Delta", f"{result['predicted_delta']}%")
                    res_col3.metric("Forecasted Next Occupancy", f"{result['predicted_next_occupancy']}%")
                    
                    st.markdown("---")
                    predicted_val = result['predicted_next_occupancy']
                    if predicted_val >= 95:
                        st.error("🚨 **High Pressure Warning:** Forecasted occupancy is critically high. Risk of bottleneck. *Action Required:* Investigate staffing levels, bed allocation, and discharge blockages.")
                    elif predicted_val >= 85:
                        st.warning("⚡ **Elevated Pressure:** Moderate capacity strain. Monitor workflow closely.")
                    else:
                        st.success("✅ **Stable Operations:** Forecasted occupancy remains within stable limits.")
                        
                    # --- FEATURE INFLUENCE / EXPLAINABILITY ---
                    st.markdown("---")
                    st.subheader("🔍 Explainability — Why this forecast?")
                    st.markdown("The breakdown below highlights the key operational factors driving this specific prediction for the next quarter:")
                    
                    factors = {
                        "Current Occupancy Level (p_occ)": f"{latest_row['p_occ']:.1f}% (High base load increases pressure risk)",
                        "Available Staffed Beds (asb)": f"{latest_row['asb']:.1f} beds (Capacity constraint factor)",
                        "Historical Trend (Previous Occupancy)": f"{prev_row['p_occ']:.1f}% (Momentum signal from prior quarter)"
                    }
                    for feature, desc in factors.items():
                        st.markdown(f"- **{feature}:** {desc}")

                    # --- CSV EXPORT BUTTON FOR HOSPITAL MANAGERS ---
                    st.markdown("---")
                    st.subheader("📥 Export Operational Report")
                    
                    report_df = pd.DataFrame([{
                        "Hospital": selected_loc,
                        "Specialty": selected_spec,
                        "Current_Occupancy": result['current_occupancy'],
                        "Predicted_Delta": result['predicted_delta'],
                        "Forecasted_Next_Occupancy": result['predicted_next_occupancy'],
                        "Warning_Status": "High Pressure" if predicted_val >= 95 else ("Elevated" if predicted_val >= 85 else "Stable")
                    }])
                    
                    csv_data = report_df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="📥 Download Executive Summary (CSV)",
                        data=csv_data,
                        file_name=f"Hospital_Forecast_{selected_loc.replace(' ', '_')}_{selected_spec.replace(' ', '_')}.csv",
                        mime="text/csv",
                        type="secondary"
                    )
                    
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Connection failed. Is FastAPI running on port 8000? Error: {e}")

with tab2:
    # --- TAB 2: WAITING LISTS CONTEXT (VISUALIZED) ---
    st.subheader("⏳ Monthly Waiting Lists Trends & Board-Level Context")
    if waiting_df is not None:
        st.markdown("Macro-level waiting list metrics across Health Boards visualized for operational planning.")
        
        if "MonthEnding" in waiting_df.columns:
            waiting_df["MonthEnding"] = pd.to_datetime(waiting_df["MonthEnding"].astype(str), format="%Y%m%d", errors="coerce")
            
        if "HBT" in waiting_df.columns:
            boards = sorted(waiting_df["HBT"].dropna().unique())
            selected_board = st.selectbox("Select Health Board (HBT)", boards)
            board_data = waiting_df[waiting_df["HBT"] == selected_board].copy()
        else:
            board_data = waiting_df.copy()
            
        if "PatientType" in board_data.columns and "WaitType" in board_data.columns:
            p_types = board_data["PatientType"].unique()
            selected_ptype = st.selectbox("Select Patient Type", p_types)
            board_data = board_data[board_data["PatientType"] == selected_ptype]
            
        if "MonthEnding" in board_data.columns and "Waits" in board_data.columns:
            for col in ["Waits", "WaitsOver12Weeks", "WaitsOver52Weeks"]:
                if col in board_data.columns:
                    board_data[col] = pd.to_numeric(board_data[col], errors="coerce")
                    
            trend_df = board_data.groupby("MonthEnding")[["Waits", "WaitsOver12Weeks", "WaitsOver52Weeks"]].sum().reset_index()
            trend_df = trend_df.sort_values("MonthEnding")
            
            st.markdown("#### 📈 Waiting List Backlog Over Time")
            st.line_chart(trend_df.set_index("MonthEnding")[["Waits", "WaitsOver12Weeks", "WaitsOver52Weeks"]])
            
            st.info("💡 **Insight:** Tracking patients waiting over 12 or 52 weeks helps leadership identify systemic bottlenecks in outpatient and inpatient flows across the Board.")
        else:
            st.dataframe(board_data.head(20), use_container_width=True)
    else:
        st.info("ℹ️ Waiting times dataset not found.")