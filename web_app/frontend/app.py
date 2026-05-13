import streamlit as st
import requests
import pandas as pd
import io
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="HDD RiskGuard | Enterprise Monitor", layout="wide", initial_sidebar_state="expanded")

# --- DARK MODE CSS (Transparent Cards + White Text) ---
st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
        
        /* 1. REMOVE WHITE BOXES from Metrics */
        div[data-testid="stMetric"] {
            background-color: transparent !important;
            border: 1px solid #374151; /* Subtle dark border */
            padding: 10px;
            border-radius: 5px;
            box-shadow: none !important;
        }

        /* 2. FORCE TEXT TO WHITE */
        div[data-testid="stMetricLabel"] {
            color: #9ca3af !important; /* Light Grey for Label */
            font-size: 0.9rem !important;
        }
        
        div[data-testid="stMetricValue"] {
            color: #ffffff !important; /* Pure White for Value */
        }

        /* 3. Fix Expanders (Transparent Background, White Text) */
        div[data-testid="stExpander"] {
            background-color: transparent !important;
            border: 1px solid #4b5563;
            color: #ffffff !important;
        }
        
        div[data-testid="stExpander"] p, div[data-testid="stExpander"] span {
            color: #ffffff !important;
        }

        /* 4. Headers */
        h1, h2, h3, h4, h5, p, span {
            color: #ffffff !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "token" not in st.session_state: st.session_state.token = None
if "page" not in st.session_state: st.session_state.page = "login"
if "email" not in st.session_state: st.session_state.email = ""
if "last_report" not in st.session_state: st.session_state.last_report = None

# --- API HELPERS ---
def get_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}

def api_post(endpoint, data=None, files=None):
    try:
        if files:
            response = requests.post(f"{API_URL}{endpoint}", headers=get_headers(), files=files)
        else:
            response = requests.post(f"{API_URL}{endpoint}", headers=get_headers(), json=data)
        return response
    except requests.exceptions.ConnectionError:
        return None

def api_get(endpoint):
    try:
        response = requests.get(f"{API_URL}{endpoint}", headers=get_headers())
        return response
    except requests.exceptions.ConnectionError:
        return None

# --- VISUALIZATION: STATUS BAR ---
def create_status_bar(buckets):
    """Creates a horizontal stacked bar showing fleet composition."""
    labels = list(buckets.keys())
    values = list(buckets.values())
    total = sum(values) if sum(values) > 0 else 1
    
    color_map = {
        "Very Good Health": "#10b981", 
        "Medium Health": "#f59e0b",    
        "Lesser Health": "#f97316",    
        "Bad Health": "#ef4444",       
        "Critical Health": "#b91c1c"   
    }
    colors = [color_map.get(l, "#cbd5e1") for l in labels]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[''],
        x=values,
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v}" if v > 0 else "" for v in values],
        textposition='auto',
        hoverinfo='text+x',
        hovertext=[f"{l}: {v}" for l, v in zip(labels, values)],
        textfont=dict(color='white') 
    ))
    
    fig.update_layout(
        barmode='stack',
        height=40,
        margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- AUTH PAGES ---
def login_page():
    st.markdown("<h2 style='text-align: center; color: white;'>RiskGuard Enterprise Login</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.form("login_form"):
            email = st.text_input("Username / Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign In", use_container_width=True):
                res = api_post("/auth/login", data={"email": email, "password": password})
                if res and res.status_code == 200:
                    st.session_state.token = res.json()["access_token"]
                    st.session_state.email = email
                    st.session_state.page = "dashboard"
                    st.rerun()
                elif res: st.error("Authentication Failed")

        if st.button("Register New ID", use_container_width=True):
            st.session_state.page = "register"
            st.rerun()

def register_page():
    st.title("System Registration")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")
    
    if st.button("Create ID"):
        if password != confirm: st.error("Passwords mismatch"); return
        res = api_post("/auth/register", data={"email": email, "password": password})
        if res and res.status_code == 200:
            st.success("ID Created. Redirecting..."); time.sleep(1.0); st.session_state.page = "login"; st.rerun()
        elif res: 
            detail = res.json().get("detail")
            st.error(str(detail))
    
    if st.button("Cancel"): st.session_state.page = "login"; st.rerun()

# --- MAIN VIEWS ---

def view_live_ops():
    """Industrial Control Panel View"""
    st_autorefresh(interval=5000, limit=None, key="fcounter")

    # Header Row
    c1, c2 = st.columns([6, 1])
    c1.markdown("###  Fleet Operations Center")
    c2.caption(f"LIVE | {datetime.now().strftime('%H:%M:%S UTC')}")

    state_res = api_get("/live/state")
    if not state_res or state_res.status_code != 200:
        st.error(" Connection Lost: Telemetry stream unreachable.")
        return

    state = state_res.json()
    if not state:
        st.info("Waiting for telemetry stream...")
        return

    # --- AGGREGATION ---
    total_drives = 0
    total_crit = 0
    active_locs = len(state)
    
    for wh in state.values():
        for v in wh.values():
            total_drives += v['total_drives']
            total_crit += v['critical_count']

    # --- TOP LEVEL METRICS ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Assets", f"{total_drives:,}", help="Total monitored HDDs")
    k2.metric("Active Warehouses", active_locs)
    
    crit_label = "Nominal"
    crit_color = "normal"
    if total_crit > 0:
        crit_label = "Risk Detected"
        crit_color = "inverse"
    k3.metric("Critical Alerts", total_crit, delta=crit_label if total_crit > 0 else None, delta_color=crit_color)
    
    risk_pct = (total_crit / total_drives * 100) if total_drives > 0 else 0.0
    k4.metric("Fleet Risk %", f"{risk_pct:.2f}%", delta="-High" if risk_pct > 5 else "Stable", delta_color="inverse")

    st.divider()

    # --- WAREHOUSE STATUS GRID ---
    for wh_name, vaults in state.items():
        with st.container():
            st.markdown(f"**📍 {wh_name} Data Center**")
            
            # Header
            h1, h2, h3, h4 = st.columns([1, 1, 1, 3])
            h1.caption("Vault ID")
            h2.caption("Drives")
            h3.caption("Status")
            h4.caption("Health Distribution")
            
            for vault_name, stats in vaults.items():
                r1, r2, r3, r4 = st.columns([1, 1, 1, 3])
                
                r1.write(f"**{vault_name}**")
                r2.write(f"{stats['total_drives']}")
                
                if stats['critical_count'] > 0:
                    r3.markdown(f":red[**⚠️ {stats['critical_count']} Crit**]")
                else:
                    r3.markdown(":green[**✓ OK**]")
                
                # Visual Bar
                with r4:
                    if stats.get("buckets"):
                        fig = create_status_bar(stats["buckets"])
                        st.plotly_chart(fig, use_container_width=True, key=f"bar_{wh_name}_{vault_name}", config={'displayModeBar': False})
            
            st.markdown("---")

    # --- EVENT LOG ---
    st.subheader("Event Log")
    warn_res = api_get("/live/warnings")
    
    if warn_res and warn_res.status_code == 200:
        warnings = warn_res.json()
        if warnings:
            df = pd.DataFrame(warnings)
            if not df.empty and "time" in df.columns:
                df = df[["time", "warehouse", "vault", "device", "risk", "severity"]]
                df.columns = ["Time", "Warehouse", "Vault", "Device ID", "Risk %", "Severity"]
                
                st.dataframe(
                    df, 
                    use_container_width=True, 
                    hide_index=True,
                    height=250,
                    column_config={
                        "Risk %": st.column_config.ProgressColumn(
                            "Failure Prob.", 
                            format="%.1f%%", 
                            min_value=0, 
                            max_value=100
                        ),
                        "Time": st.column_config.DatetimeColumn("Timestamp", format="HH:mm:ss"),
                    }
                )
            else:
                st.caption("No events recorded yet.")
        else:
            st.caption("No active alerts in the last 24 hours.")

def view_batch_analysis():
    st.header("Manual Fleet Analysis")
    st.markdown("Upload telemetry CSV dump for ad-hoc diagnostics.")
    
    uploaded_file = st.file_uploader("Select CSV File", type=["csv"])
    
    if uploaded_file:
        if st.button("Process File", type="primary"):
            with st.status("Analyzing...", expanded=True) as status:
                files = {"file": uploaded_file.getvalue()}
                res = api_post("/upload-csv", files=files)
                
                if res and res.status_code == 200:
                    st.session_state.last_report = res.json()
                    status.update(label="Complete", state="complete", expanded=False)
                elif res:
                    status.update(label="Failed", state="error")
                    st.error(f"Error: {res.text}")

    if st.session_state.last_report:
        render_report_details(st.session_state.last_report)

def view_history():
    st.header("Evaluation History")
    if st.button("Refresh"): st.rerun()
        
    res = api_get("/reports/history")
    if res and res.status_code == 200:
        history = res.json()
        if history:
            data = []
            for item in history:
                data.append({
                    "Date": str(item['date']).replace("T", " ")[:16],
                    "Filename": item['filename'],
                    "Drives": item['total_drives'],
                    "Critical": item['critical_count'],
                    "ID": item['report_id']
                })
            
            df = pd.DataFrame(data)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(df.drop(columns=["ID"]), use_container_width=True, hide_index=True, height=400)
            with col2:
                st.info("To download a report, select the ID below:")
                selected_id = st.selectbox("Select Report", df["ID"].tolist(), format_func=lambda x: f"Report ending in ...{x[-6:]}")
                if st.button("Download PDF"):
                     pdf_url = f"{API_URL}/generate-pdf/{selected_id}"
                     st.markdown(f'<a href="{pdf_url}" target="_blank">Click here to open PDF</a>', unsafe_allow_html=True)

def render_report_details(report):
    summ = report["summary"]
    st.divider()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Drives", summ['total_drives'])
    c2.metric("Critical", len(summ['top_critical']))
    c3.metric("Healthy", summ.get('buckets', {}).get('Very Good Health', 0))

    st.subheader("Critical Assets")
    crit_df = pd.DataFrame(summ["top_critical"])
    
    if not crit_df.empty:
        crit_df = crit_df[["device_id", "health_score", "risk_score", "dominant_mechanism"]]
        crit_df.columns = ["Device ID", "Health", "Risk %", "Mechanism"]
        st.dataframe(crit_df, use_container_width=True, hide_index=True)
    
    pdf_url = f"{API_URL}/generate-pdf/{report['report_id']}"
    st.markdown(f'<a href="{pdf_url}" target="_blank" class="stButton"><button style="padding:10px; width:100%; cursor:pointer;">Download Full PDF Report</button></a>', unsafe_allow_html=True)

# --- MAIN LAYOUT ---
def dashboard():
    with st.sidebar:
        st.markdown("### RiskGuard")
        st.caption("Enterprise Edition")
        st.markdown("---")
        menu = st.radio("Menu", ["Live Operations", "Manual Analysis", "History"], label_visibility="collapsed")
        st.markdown("---")
        st.caption(f"User: {st.session_state.email}")
        if st.button("Logout"):
            st.session_state.token = None; st.session_state.page = "login"; st.rerun()

    if menu == "Live Operations": view_live_ops()
    elif menu == "Manual Analysis": view_batch_analysis()
    elif menu == "History": view_history()

# --- ROUTER ---
if st.session_state.page == "login": login_page()
elif st.session_state.page == "register": register_page()
elif st.session_state.page == "dashboard": dashboard()