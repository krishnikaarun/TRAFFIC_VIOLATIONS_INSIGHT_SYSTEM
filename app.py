import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Traffic Violations Insight Dashboard",
    page_icon="🚓",
    layout="wide"
)

st.title("🚓 Traffic Violations Insight System")
st.markdown("Interactive EDA and Analytics Dashboard for Traffic Stop Violations")
st.markdown("---")

# -----------------------------------------------------------------------------
# DATA LOADING & CLEANING CACHE
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    # Replace with your actual dataset file path
    try:
        df = pd.read_csv('traffic_violations.csv')
    except FileNotFoundError:
        # Fallback for demonstration if file isn't present yet
        return pd.DataFrame()

    # 1. Standardize String Columns
    string_cols = ['Agency', 'SubAgency', 'Description', 'Location', 'State', 
                   'VehicleType', 'Make', 'Model', 'Color', 'Violation Type', 
                   'Charge', 'Article', 'Race', 'Driver City', 'Driver State', 
                   'DL State', 'Arrest Type']

    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df[col] = df[col].replace(['NAN', 'NONE', 'UNKNOWN', ''], np.nan)

    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].astype(str).str.strip().str.upper().fillna('UNKNOWN')

    # 2. Date & Time Parsing + Safer Drop-Down Extractions
    if 'Date Of Stop' in df.columns:
        df['Date Of Stop'] = pd.to_datetime(df['Date Of Stop'], errors='coerce')
        df.loc[(df['Date Of Stop'].dt.year > 2026) | (df['Date Of Stop'].dt.year < 2000), 'Date Of Stop'] = pd.NaT
        df['Weekday'] = df['Date Of Stop'].dt.day_name()
        df['Month'] = df['Date Of Stop'].dt.month_name()

    if 'Time Of Stop' in df.columns:
        df['Time Of Stop'] = df['Time Of Stop'].astype(str).str.replace('.', ':', regex=False)
        time_delta = pd.to_timedelta(df['Time Of Stop'], errors='coerce')
        df['Hour'] = time_delta.dt.components['hours']

    # 3. Geographic Clean Up
    for col in ['Latitude', 'Longitude']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] == 0.0, col] = np.nan

    if 'Longitude' in df.columns:
        df.loc[(df['Longitude'] > -65) | (df['Longitude'] < -125), 'Longitude'] = np.nan

    # 4. Standardize Boolean Columns
    boolean_cols = ['Accident', 'Belts', 'Personal Injury', 'Property Damage', 'Fatal', 
                    'Commercial License', 'HAZMAT', 'Commercial Vehicle', 'Alcohol', 
                    'Work Zone', 'Search Conducted', 'Contributed To Accident']

    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df[col] = df[col].map({'YES': True, 'Y': True, 'TRUE': True, 'NO': False, 'N': False, 'FALSE': False})
            df[col] = df[col].fillna(False)

    # 5. Vehicle Year Bounds Check
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df.loc[(df['Year'] < 1960) | (df['Year'] > 2027), 'Year'] = np.nan

    return df

df_raw = load_and_clean_data()

if df_raw.empty:
    st.error("⚠️ 'traffic_violations.csv' not found or empty. Please ensure the file is in the working directory.")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR MULTI-FILTERS
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Filter Configurations")

# Date Range Filter
if 'Date Of Stop' in df_raw.columns and not df_raw['Date Of Stop'].isna().all():
    min_date = df_raw['Date Of Stop'].min().date()
    max_date = df_raw['Date Of Stop'].max().date()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
else:
    date_range = None

# Categorical Multi-selects
def create_multiselect(label, column):
    if column in df_raw.columns:
        options = sorted(df_raw[column].dropna().unique())
        return st.sidebar.multiselect(label, options=options)
    return []

selected_locations = create_multiselect("Locations", 'Location')
selected_v_types = create_multiselect("Vehicle Types", 'VehicleType')
selected_genders = create_multiselect("Gender", 'Gender')
selected_races = create_multiselect("Race", 'Race')
selected_categories = create_multiselect("Violation Categories", 'Violation Type')

# Apply Filters to Dataset
df_filtered = df_raw.copy()

if date_range and len(date_range) == 2:
    df_filtered = df_filtered[(df_filtered['Date Of Stop'].dt.date >= date_range[0]) & 
                              (df_filtered['Date Of Stop'].dt.date <= date_range[1])]

if selected_locations:
    df_filtered = df_filtered[df_filtered['Location'].isin(selected_locations)]
if selected_v_types:
    df_filtered = df_filtered[df_filtered['VehicleType'].isin(selected_v_types)]
if selected_genders:
    df_filtered = df_filtered[df_filtered['Gender'].isin(selected_genders)]
if selected_races:
    df_filtered = df_filtered[df_filtered['Race'].isin(selected_races)]
if selected_categories:
    df_filtered = df_filtered[df_filtered['Violation Type'].isin(selected_categories)]

# -----------------------------------------------------------------------------
# SUMMARY STATISTICS (KPI CARDS)
# -----------------------------------------------------------------------------
st.subheader("📊 High-Level Metrics Summary")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(label="Total Violations Count", value=f"{len(df_filtered):,}")

with kpi2:
    accident_count = df_filtered['Accident'].sum() if 'Accident' in df_filtered.columns else 0
    st.metric(label="Accident-Involved Cases", value=f"{accident_count:,}")

with kpi3:
    high_risk_zone = df_filtered['Location'].mode()[0] if 'Location' in df_filtered.columns and not df_filtered['Location'].empty else "N/A"
    st.metric(label="Top High-Risk Zone", value=str(high_risk_zone)[:22])

with kpi4:
    top_make = df_filtered['Make'].mode()[0] if 'Make' in df_filtered.columns and not df_filtered['Make'].empty else "N/A"
    st.metric(label="Most Cited Vehicle Make", value=str(top_make))

st.markdown("---")

# -----------------------------------------------------------------------------
# CHARTS & VISUALIZATIONS
# -----------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🔀 Demographic and Violation Types")
    if 'Race' in df_filtered.columns and 'Violation Type' in df_filtered.columns:
        # Cross tab metrics for Plotly Express
        df_cross = df_filtered.groupby(['Race', 'Violation Type']).size().reset_index(name='Counts')
        fig_demo = px.bar(df_cross, x='Race', y='Counts', color='Violation Type', 
                          title="Violation Type Breakdown across Races", barmode='stack')
        st.plotly_chart(fig_demo, use_container_width=True)
    else:
        st.info("Demographic data columns not available.")

with col_right:
    st.subheader("🏎️ Most Frequently Involved Makes")
    if 'Make' in df_filtered.columns:
        top_makes_df = df_filtered['Make'].value_counts().head(10).reset_index(name='Count')
        
        # FIXED: Changed color_continuous_scale from 'rocket' to 'magma'
        fig_make = px.bar(top_makes_df, x='Count', y='Make', orientation='h',
                          title="Top 10 Vehicle Makes Involved", color='Count', 
                          color_continuous_scale='magma')
                          
        fig_make.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_make, use_container_width=True)

# --- Temporal Variations Section ---
st.subheader("⏰ Temporal Trends Analysis")
t_col1, t_col2, t_col3 = st.columns(3)

with t_col1:
    if 'Hour' in df_filtered.columns:
        hour_df = df_filtered['Hour'].value_counts().sort_index().reset_index(name='Count')
        fig_hour = px.line(hour_df, x='Hour', y='Count', title="Hourly Peak Violation Activity", markers=True)
        st.plotly_chart(fig_hour, use_container_width=True)

with t_col2:
    if 'Weekday' in df_filtered.columns:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_df = df_filtered['Weekday'].value_counts().reindex(day_order).reset_index(name='Count')
        fig_day = px.bar(day_df, x='Weekday', y='Count', title="Violations by Day of the Week", color='Count')
        st.plotly_chart(fig_day, use_container_width=True)

with t_col3:
    if 'Month' in df_filtered.columns:
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        existing_m = [m for m in month_order if m in df_filtered['Month'].unique()]
        month_df = df_filtered['Month'].value_counts().reindex(existing_m).reset_index(name='Count')
        fig_month = px.bar(month_df, x='Month', y='Count', title="Violations by Month Tracked", color='Count', color_continuous_scale='Purples')
        st.plotly_chart(fig_month, use_container_width=True)

# -----------------------------------------------------------------------------
# GEOGRAPHICAL INCIDENT MAP
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📍 Geographical Map Bounding Hotspots")

geo_df = df_filtered.dropna(subset=['Latitude', 'Longitude'])

if not geo_df.empty:
    # FIXED: Explicitly tell Streamlit which columns map to latitude and longitude
    st.map(geo_df, latitude='Latitude', longitude='Longitude')
else:
    st.info("No valid Latitude/Longitude spatial coordinates found within current filtered range to render map data.")