# IMPORTS 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import google.generativeai as genai
from statsmodels.tsa.arima.model import ARIMA

#  PAGE CONFIG 
st.set_page_config(
    page_title="Healthcare Analytics with Forecast",
    page_icon="🏥",
    layout="wide"
)

#  THEME COLORS 
PRIMARY_COLOR = "#10b981" # Emerald Green
SECONDARY_COLOR = "#0d2e33" # Deep Teal
BG_COLOR = "#f8fafc"
CHART_COLORS = ['#10b981', '#0d2e33', '#3b82f6', '#f59e0b', '#ef4444']

#  SESSION STATE
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "hospital" not in st.session_state:
    st.session_state["hospital"] = None
if "messages" not in st.session_state:
    st.session_state.messages = []

#  GEMINI API CONFIG 
# Requires a .streamlit/secrets.toml file with: GEMINI_API_KEY = "your_key"
try:
    genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", ""))
    model = genai.GenerativeModel('gemini-1.5-flash')
    api_configured = True
except Exception:
    api_configured = False
    
# GLOBAL CSS THEME 
def apply_global_theme():
    st.markdown(f"""
        <style>
        .stApp {{ background-color: {BG_COLOR}; }}
        [data-testid="stSidebar"] {{ background-color: {SECONDARY_COLOR} !important; }}
        [data-testid="stSidebar"] * {{ color: white !important; }}
        div.stMetric {{ 
            background-color: white; 
            padding: 15px; 
            border-radius: 10px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.05); 
            border-left: 5px solid {PRIMARY_COLOR}; 
        }}
        .stButton>button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border-radius: 8px;
            border: none;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #059669;
            box-shadow: 0 5px 15px rgba(16, 185, 129, 0.3);
        }}
        </style>
    """, unsafe_allow_html=True)

# DYNAMIC LOGIN PAGE 
def login_page():
    st.markdown(f"""
        <style>
        .stApp {{ background: linear-gradient(135deg, {SECONDARY_COLOR} 0%, {PRIMARY_COLOR} 100%); }}
        .login-card {{
            background: rgba(255, 255, 255, 0.95);
            padding: 3rem;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }}
        .main-title {{ color: {SECONDARY_COLOR}; font-size: 32px; font-weight: 800; text-align: center; margin-bottom: 5px; }}
        .sub-title {{ color: #64748b; font-size: 16px; text-align: center; margin-bottom: 30px; }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 1.5, 1])

    with col2:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        col_img1, col_img2, col_img3 = st.columns([1,2,1])
        with col_img2:
            st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", use_container_width=True)
        
        st.markdown("<h2 class='main-title'>Healthcare Analytics with forecast</h2>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>Predictive insights for modern hospital management.</p>", unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            hospital = st.selectbox("Assigning Hospital", ["Hospital1", "Hospital2"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            login_btn = st.form_submit_button("Access Dashboard", use_container_width=True)

        if login_btn:
            if username == "admin" and password == "admin123":
                st.session_state["logged_in"] = True
                st.session_state["hospital"] = hospital
                st.rerun()
            else:
                st.error("Invalid Credentials. Please check and try again.")
        st.markdown("</div>", unsafe_allow_html=True)

# Show login if not logged in
if not st.session_state["logged_in"]:
    login_page()
    st.stop()

# Apply the theme to the rest of the app
apply_global_theme()

#  LOAD DATA 
@st.cache_data
def load_data(hospital_name):
    try:
        if hospital_name == "Hospital1":
            df = pd.read_csv("dataset/appointments_cleaned.csv")
            table_name = "appointments"
        else:
            df = pd.read_csv("dataset/patients_cleaned.csv")
            table_name = "patients"
        return df, table_name
    except FileNotFoundError:
        st.error(f"⚠️ Dataset not found! Please ensure your CSV files are securely placed inside the 'dataset/' folder.")
        st.stop()

df, table_name = load_data(st.session_state.hospital)

# ===================== SQLITE =====================
conn = sqlite3.connect("hospital_major_project.db", check_same_thread=False)
df.to_sql(table_name, conn, if_exists="replace", index=False)

def load_from_db(table):
    return pd.read_sql(f"SELECT * FROM {table}", conn)

# ===================== SIDEBAR =====================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=50)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Visualizations", "Correlation", "Forecasting", "AI Chatbot", "Database"])

st.sidebar.markdown("---")
st.sidebar.write("Hospital Node:", f"**{st.session_state['hospital']}**")

if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.session_state["hospital"] = None
    st.rerun()

# DASHBOARD 
if page == "Overview":
    st.title("🏥 Hospital Overview")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df))
    c2.metric("Departments", df["department"].nunique() if "department" in df.columns else "N/A")
    c3.metric("Average Age", round(df["age"].mean(),1) if "age" in df.columns else "N/A")
    c4.metric("Beds Available", df["bed_availability"].sum() if "bed_availability" in df.columns else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Dataset Structure")
    clean_df = df.dropna()
    col1, col2 = st.columns(2)
    with col1: st.write("**Dataset Shape:**", df.shape)
    with col2: st.write("**After Cleaning:**", clean_df.shape)

    with st.expander("View Cleaned Data"):
        st.dataframe(clean_df.head(), use_container_width=True)

#  VISUALIZATIONS 
elif page == "Visualizations":
    st.title("📈 Analytics Visualizations")

    col1, col2 = st.columns(2)
    
    if "department" in df.columns:
        with col1:
            fig = px.pie(df, names="department", title="Department Distribution", color_discrete_sequence=CHART_COLORS)
            st.plotly_chart(fig, use_container_width=True)
            
    if "gender" in df.columns:
        with col2:
            fig = px.bar(df, x="gender", title="Gender Distribution", color="gender", color_discrete_sequence=CHART_COLORS)
            st.plotly_chart(fig, use_container_width=True)
    elif "status" in df.columns:
        with col2:
            fig = px.pie(df, names="status", title="Appointment Status", color_discrete_sequence=CHART_COLORS)
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    if "age" in df.columns:
        with col3:
            fig = px.histogram(df, x="age", nbins=30, title="Age Distribution", color_discrete_sequence=[PRIMARY_COLOR])
            st.plotly_chart(fig, use_container_width=True)

    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols:
        date_col = date_cols[0]
        daily = df.groupby(date_col).size().reset_index(name="count")
        with col4:
            fig = px.line(daily, x=date_col, y="count", title="Daily Trend", color_discrete_sequence=[SECONDARY_COLOR])
            st.plotly_chart(fig, use_container_width=True)

# ===================== CORRELATION =====================
elif page == "Correlation":
    st.title("🔗 Correlation Matrix")
    corr = df.select_dtypes(include=np.number).corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Numerical Feature Correlation", color_continuous_scale="Tealgrn")
    st.plotly_chart(fig, use_container_width=True)

# ===================== FORECASTING =====================
elif page == "Forecasting":
    st.title("🔮 Daily Admissions Forecast")
    date_cols = [c for c in df.columns if "date" in c.lower()]

    if not date_cols:
        st.warning("No date column found for forecasting.")
    else:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col])
        daily = df.groupby(date_col).size().reset_index(name="admissions")

        if len(daily) < 20:
            st.warning("Not enough data points for ARIMA forecasting.")
        else:
            model = ARIMA(daily["admissions"], order=(1,1,1))
            model_fit = model.fit()

            forecast_steps = st.slider("Select Forecast Horizon (Days)", 7, 30, 10)
            forecast = model_fit.forecast(steps=forecast_steps)
            future_dates = pd.date_range(start=daily[date_col].max(), periods=forecast_steps+1, freq="D")[1:]

























