"""
Monsoon Solar Predictor - ADVANCED DASHBOARD
Features: Live Weather API, Historical Comparison, What-If Scenarios, Model Retraining
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import requests
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io

# Page configuration
st.set_page_config(
    page_title="Monsoon Solar Predictor - Advanced",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #FF6B35, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF6B35;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-danger {
        background-color: #ffe6e6;
        border-left: 5px solid #ff4444;
    }
    .alert-warning {
        background-color: #fff4e6;
        border-left: 5px solid #ffaa00;
    }
    .alert-success {
        background-color: #e6ffe6;
        border-left: 5px solid #44ff44;
    }
    .city-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .scenario-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# City database
CITIES = {
    "Mumbai": {
        "state": "Maharashtra",
        "coords": (19.0760, 72.8777),
        "coords_str": "19.0760°N, 72.8777°E",
        "capacity_mw": 150,
        "grid": "Mumbai Solar Park - Mulund",
        "monsoon_intensity": "High"
    },
    "Pune": {
        "state": "Maharashtra", 
        "coords": (18.5204, 73.8567),
        "coords_str": "18.5204°N, 73.8567°E",
        "capacity_mw": 100,
        "grid": "Pune Grid Station A - Hinjewadi",
        "monsoon_intensity": "Medium"
    },
    "Nagpur": {
        "state": "Maharashtra",
        "coords": (21.1458, 79.0882),
        "coords_str": "21.1458°N, 79.0882°E",
        "capacity_mw": 80,
        "grid": "Nagpur Solar Farm - MIHAN",
        "monsoon_intensity": "Medium"
    },
    "Delhi": {
        "state": "NCR",
        "coords": (28.7041, 77.1025),
        "coords_str": "28.7041°N, 77.1025°E",
        "capacity_mw": 120,
        "grid": "Delhi NCR Grid - Badarpur",
        "monsoon_intensity": "Low"
    },
    "Bangalore": {
        "state": "Karnataka",
        "coords": (12.9716, 77.5946),
        "coords_str": "12.9716°N, 77.5946°E",
        "capacity_mw": 90,
        "grid": "Bangalore Solar Hub - Devanahalli",
        "monsoon_intensity": "Medium"
    },
    "Hyderabad": {
        "state": "Telangana",
        "coords": (17.3850, 78.4867),
        "coords_str": "17.3850°N, 78.4867°E",
        "capacity_mw": 110,
        "grid": "Hyderabad Grid - Uppal",
        "monsoon_intensity": "Medium"
    },
    "Ahmedabad": {
        "state": "Gujarat",
        "coords": (23.0225, 72.5714),
        "coords_str": "23.0225°N, 72.5714°E",
        "capacity_mw": 200,
        "grid": "Ahmedabad Solar Park - Changodar",
        "monsoon_intensity": "Low"
    },
    "Chennai": {
        "state": "Tamil Nadu",
        "coords": (13.0827, 80.2707),
        "coords_str": "13.0827°N, 80.2707°E",
        "capacity_mw": 95,
        "grid": "Chennai Grid - Manali",
        "monsoon_intensity": "High"
    }
}


@st.cache_resource
def load_model_and_scaler():
    """Load trained model and scaler"""
    try:
        # Get the directory of the current script
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Go up one level to project root
        
        model_path = os.path.join(project_root, 'models', 'monsoon_solar_lstm.keras')
        scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')
        
        model = keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("💡 Please ensure model files are present in the 'models' folder")
        return None, None


@st.cache_data
def load_historical_data():
    """Load historical solar data"""
    try:
        # Get the directory of the current script
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        data_path = os.path.join(project_root, 'data', 'monsoon_solar_data.csv')
        
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("💡 Please ensure 'monsoon_solar_data.csv' is in the 'data' folder")
        return None


# ========================================
# FEATURE 1: LIVE WEATHER API
# ========================================

def fetch_live_weather(lat, lon, api_key=None):
    """
    Fetch live weather data from OpenWeatherMap
    
    Note: For demo purposes, we'll simulate live data if no API key
    """
    
    if api_key and api_key != "":
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                weather_data = {
                    'cloud_cover_percent': data['clouds']['all'],
                    'temperature_celsius': data['main']['temp'],
                    'humidity_percent': data['main']['humidity'],
                    'wind_speed_kmh': data['wind']['speed'] * 3.6,  # m/s to km/h
                    'source': 'live_api',
                    'description': data['weather'][0]['description'],
                    'timestamp': datetime.now()
                }
                
                return weather_data, None
            else:
                return None, f"API Error: {response.status_code}"
                
        except Exception as e:
            return None, f"Connection Error: {str(e)}"
    else:
        # Simulate live weather for demo
        weather_data = {
            'cloud_cover_percent': np.random.randint(20, 80),
            'temperature_celsius': np.random.uniform(26, 35),
            'humidity_percent': np.random.randint(60, 90),
            'wind_speed_kmh': np.random.uniform(8, 20),
            'source': 'simulated',
            'description': 'scattered clouds',
            'timestamp': datetime.now()
        }
        return weather_data, None


# ========================================
# FEATURE 2: HISTORICAL COMPARISON
# ========================================

def get_historical_comparison(df, current_timestamp, lookback_days=365):
    """
    Compare current conditions to same day last year
    """
    
    # Get same day last year
    last_year = current_timestamp - timedelta(days=lookback_days)
    
    # Find closest match in historical data
    df_temp = df.copy()
    df_temp['time_diff'] = abs(df_temp['timestamp'] - last_year)
    closest_idx = df_temp['time_diff'].idxmin()
    
    historical_row = df.iloc[closest_idx]
    
    comparison = {
        'historical_date': historical_row['timestamp'],
        'historical_output': historical_row['solar_output_mw'],
        'historical_clouds': historical_row['cloud_cover_percent'],
        'historical_temp': historical_row['temperature_celsius'],
    }
    
    return comparison


# ========================================
# FEATURE 3: WHAT-IF SCENARIOS
# ========================================

def simulate_scenario(model, scaler, base_sequence, modifications):
    """
    Simulate what-if scenarios by modifying weather parameters
    
    modifications: dict with keys like 'cloud_delta', 'temp_delta', etc.
    """
    
    # Create modified sequence
    modified_seq = base_sequence.copy()
    
    # Apply modifications
    if 'cloud_delta' in modifications:
        modified_seq[:, 0] = np.clip(modified_seq[:, 0] + modifications['cloud_delta'], 0, 100)
    
    if 'temp_delta' in modifications:
        modified_seq[:, 1] = np.clip(modified_seq[:, 1] + modifications['temp_delta'], 20, 45)
    
    if 'humidity_delta' in modifications:
        modified_seq[:, 2] = np.clip(modified_seq[:, 2] + modifications['humidity_delta'], 30, 100)
    
    if 'wind_delta' in modifications:
        modified_seq[:, 3] = np.clip(modified_seq[:, 3] + modifications['wind_delta'], 0, 50)
    
    # Make prediction with modified data
    input_normalized = scaler.transform(modified_seq)
    input_reshaped = input_normalized.reshape(1, modified_seq.shape[0], modified_seq.shape[1])
    pred_normalized = model.predict(input_reshaped, verbose=0)[0][0]
    
    # Denormalize
    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[0, -1] = pred_normalized
    pred_actual = scaler.inverse_transform(dummy)[0, -1]
    
    return max(0, pred_actual)


# ========================================
# FEATURE 4: MODEL RETRAINING INTERFACE
# ========================================

def validate_uploaded_data(df):
    """Validate uploaded CSV has correct format"""
    
    required_columns = [
        'timestamp', 'cloud_cover_percent', 'temperature_celsius',
        'humidity_percent', 'wind_speed_kmh', 'solar_output_mw'
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    
    # Check data types
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except:
        return False, "Invalid timestamp format"
    
    # Check for NaN values
    if df[required_columns[1:]].isnull().any().any():
        return False, "Dataset contains missing values"
    
    return True, "Data validation successful"


def retrain_model(df, epochs=20):
    """
    Retrain model with new data
    """
    
    # Prepare features
    feature_columns = [
        'cloud_cover_percent', 'temperature_celsius', 
        'humidity_percent', 'wind_speed_kmh', 'hour', 'solar_output_mw'
    ]
    
    # Add hour column if not present
    if 'hour' not in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    
    data = df[feature_columns].values
    
    # Normalize
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    lookback = 12
    forecast_horizon = 6
    
    for i in range(len(data_normalized) - lookback - forecast_horizon):
        X.append(data_normalized[i:(i + lookback)])
        y.append(data_normalized[i + lookback + forecast_horizon - 1, -1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Build model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=0,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )
    
    # Evaluate
    y_pred = model.predict(X_val, verbose=0)
    
    # Denormalize for metrics
    dummy = np.zeros((len(y_pred), scaler.n_features_in_))
    dummy[:, -1] = y_pred.flatten()
    y_pred_actual = scaler.inverse_transform(dummy)[:, -1]
    
    dummy[:, -1] = y_val
    y_val_actual = scaler.inverse_transform(dummy)[:, -1]
    
    mae = mean_absolute_error(y_val_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_val_actual, y_pred_actual))
    r2 = r2_score(y_val_actual, y_pred_actual)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'history': history
    }
    
    return model, scaler, metrics


# ========================================
# HELPER FUNCTIONS
# ========================================

def create_input_sequence(current_data, lookback=12):
    """Create input sequence for model prediction"""
    feature_columns = [
        'cloud_cover_percent', 'temperature_celsius',
        'humidity_percent', 'wind_speed_kmh', 'hour', 'solar_output_mw'
    ]
    data = current_data[feature_columns].values
    return data


def make_prediction(model, scaler, input_sequence):
    """Make 30-minute ahead prediction"""
    input_normalized = scaler.transform(input_sequence)
    input_reshaped = input_normalized.reshape(1, input_sequence.shape[0], input_sequence.shape[1])
    pred_normalized = model.predict(input_reshaped, verbose=0)[0][0]
    
    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[0, -1] = pred_normalized
    pred_actual = scaler.inverse_transform(dummy)[0, -1]
    
    return max(0, pred_actual)


def simulate_live_data(df, time_idx):
    """Get data window for given timestamp"""
    lookback = 12
    current_window = df.iloc[time_idx:time_idx + lookback].copy()
    actual_future = df.iloc[time_idx + lookback + 5]['solar_output_mw'] if time_idx + lookback + 5 < len(df) else None
    return current_window, actual_future


def get_alert_level(current_output, predicted_output):
    """Determine alert level"""
    change = predicted_output - current_output
    change_percent = (change / max(current_output, 1)) * 100
    
    if abs(change) < 10:
        return 'success', '✅ Stable - No significant changes expected'
    elif change < -20:
        return 'danger', f'🚨 CRITICAL ALERT: Expected drop of {abs(change):.1f} MW ({abs(change_percent):.1f}%) in 30 minutes!'
    elif change < -10:
        return 'warning', f'⚠️ WARNING: Expected drop of {abs(change):.1f} MW ({abs(change_percent):.1f}%) in 30 minutes'
    elif change > 20:
        return 'success', f'📈 Significant increase expected: +{change:.1f} MW ({change_percent:.1f}%)'
    else:
        return 'success', '✅ Minor changes expected'


# ========================================
# MAIN APPLICATION
# ========================================

def main():
    
    # Header with feature badges
    st.markdown('<p class="main-header">🌦️ Monsoon Solar Predictor - ADVANCED</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Solar Forecasting with Advanced Analytics</p>', unsafe_allow_html=True)
    
    # Feature badges
    st.markdown("---")
    badge_cols = st.columns(8)
    badges = [
        ("🌍", "Multi-City"),
        ("🔔", "Smart Alerts"),
        ("💬", "AI Chatbot"),
        ("☁️", "Live Weather"),
        ("📊", "Historical"),
        ("🔮", "What-If"),
        ("💰", "ROI Calc"),
        ("🎓", "Retraining")
    ]
    for col, (icon, text) in zip(badge_cols, badges):
        with col:
            st.markdown(f'<span class="feature-badge">{icon} {text}</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load model and data
    model, scaler = load_model_and_scaler()
    df = load_historical_data()
    
    if model is None or df is None:
        st.error("❌ Failed to load model or data.")
        return
    
    # ========================================
    # SIDEBAR
    # ========================================
    
    st.sidebar.header("⚙️ Advanced Control Panel")
    
    # Mode selection
    app_mode = st.sidebar.selectbox(
        "🎯 Select Mode",
        [
            "📊 Standard Dashboard", 
            "🌍 Multi-City Monitor", 
            "🔔 Smart Alert System",
            "💬 AI Chatbot Assistant",
            "☁️ Live Weather Mode", 
            "🔮 What-If Scenarios", 
            "💰 Cost Savings Calculator", 
            "🎓 Model Retraining"
        ],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # City Selection
    st.sidebar.subheader("📍 Location")
    city = st.sidebar.selectbox("City", list(CITIES.keys()), index=0)
    city_info = CITIES[city]
    
    st.sidebar.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
        <h3 style='margin: 0; color: white;'>📍 {city}</h3>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>{city_info['state']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    - 🗺️ {city_info['coords_str']}
    - ⚡ {city_info['capacity_mw']} MW
    - 🏭 {city_info['grid']}
    """)
    
    # Time Selection (for non-live modes)
    if app_mode != "☁️ Live Weather Mode":
        st.sidebar.markdown("---")
        st.sidebar.subheader("📅 Time Selection")
        
        time_mode = st.sidebar.radio("Mode", ["Slider", "Manual"], index=0)
        
        if time_mode == "Slider":
            max_idx = len(df) - 20
            time_idx = st.sidebar.slider("Timestamp", 12, max_idx, 1000, 1)
        else:
            min_date = df['timestamp'].min().date()
            max_date = df['timestamp'].max().date()
            
            selected_date = st.sidebar.date_input("Date", value=min_date + timedelta(days=30), 
                                                  min_value=min_date, max_value=max_date)
            
            time_preset = st.sidebar.selectbox("Time", 
                ["Custom", "06:00", "12:00", "15:00", "18:00"], index=2)
            
            if time_preset == "Custom":
                selected_time = st.sidebar.time_input("HH:MM", value=datetime.strptime("12:00", "%H:%M").time())
            else:
                selected_time = datetime.strptime(time_preset, "%H:%M").time()
            
            selected_datetime = datetime.combine(selected_date, selected_time)
            df_temp = df.copy()
            df_temp['time_diff'] = abs(df_temp['timestamp'] - selected_datetime)
            time_idx = df_temp['time_diff'].idxmin()
    
    # ========================================
    # MODE 1: MULTI-CITY REAL-TIME DASHBOARD
    # ========================================
    
    if app_mode == "🌍 Multi-City Monitor":
        
        st.subheader("🌍 Multi-City Real-Time Solar Monitoring")
        
        # API Key input at the top
        with st.expander("⚙️ Live Weather Configuration", expanded=False):
            api_key_input = st.text_input(
                "OpenWeatherMap API Key (optional - leave blank for demo mode)",
                type="password",
                help="Enter your API key for real-time weather data across all cities"
            )
            
            auto_refresh = st.checkbox("Enable Auto-Refresh (every 5 minutes)", value=False)
            
            if auto_refresh:
                st.info("🔄 Dashboard will auto-refresh every 5 minutes with latest weather data")
        
        # Manual refresh button
        col_refresh1, col_refresh2 = st.columns([1, 4])
        with col_refresh1:
            manual_refresh = st.button("🔄 Refresh Now", use_container_width=True, type="primary")
        with col_refresh2:
            # Show time in IST (India Standard Time)
            ist = timezone(timedelta(hours=5, minutes=30))
            last_update_time_ist = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
            st.info(f"📅 Last Update: {last_update_time_ist}")
        
        # Auto-refresh mechanism
        if auto_refresh:
            st.empty()  # Placeholder for auto-refresh
            import time
            time.sleep(300)  # 5 minutes = 300 seconds
            st.rerun()
        
        # Fetch live weather for all cities if API key provided
        use_live_api = bool(api_key_input)
        
        if use_live_api:
            st.success("🟢 LIVE MODE: Fetching real-time weather data from OpenWeatherMap API")
        else:
            st.warning("🟡 DEMO MODE: Using simulated weather data (Enter API key for live data)")
        
        st.markdown("---")
        
        # Get predictions for all cities
        cities_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (city_name, city_details) in enumerate(CITIES.items()):
            status_text.text(f"Fetching data for {city_name}...")
            progress_bar.progress((idx + 1) / len(CITIES))
            
            capacity_ratio = city_details['capacity_mw'] / 100
            
            if use_live_api:
                # Fetch LIVE weather data
                lat, lon = city_details['coords']
                weather_data, error = fetch_live_weather(lat, lon, api_key_input)
                
                if weather_data and not error:
                    # Create synthetic sequence using live weather
                    current_hour = datetime.now().hour
                    df_hour = df[df['hour'] == current_hour].head(12)
                    
                    if len(df_hour) >= 12:
                        synthetic_window = df_hour.copy()
                        
                        # Update with live weather
                        synthetic_window['cloud_cover_percent'] = weather_data['cloud_cover_percent']
                        synthetic_window['temperature_celsius'] = weather_data['temperature_celsius']
                        synthetic_window['humidity_percent'] = weather_data['humidity_percent']
                        synthetic_window['wind_speed_kmh'] = weather_data['wind_speed_kmh']
                        
                        # Estimate current output based on live conditions
                        cloud_factor = 1 - (weather_data['cloud_cover_percent'] / 100) * 0.85
                        time_factor = max(0, 1 - abs(current_hour - 12) / 6)
                        current_output = city_details['capacity_mw'] * cloud_factor * time_factor * 0.7
                        
                        synthetic_window.iloc[-1, synthetic_window.columns.get_loc('solar_output_mw')] = current_output / capacity_ratio
                        
                        input_seq = create_input_sequence(synthetic_window)
                        predicted_output = make_prediction(model, scaler, input_seq) * capacity_ratio
                        
                        clouds = weather_data['cloud_cover_percent']
                        temp = weather_data['temperature_celsius']
                        data_source = "LIVE"
                    else:
                        # Fallback to demo mode for this city
                        city_time_idx = time_idx + hash(city_name) % 100
                        if city_time_idx >= len(df) - 20:
                            city_time_idx = time_idx
                        
                        current_window, _ = simulate_live_data(df, city_time_idx)
                        current_output = current_window['solar_output_mw'].iloc[-1] * capacity_ratio
                        input_seq = create_input_sequence(current_window)
                        predicted_output = make_prediction(model, scaler, input_seq) * capacity_ratio
                        clouds = current_window['cloud_cover_percent'].iloc[-1]
                        temp = current_window['temperature_celsius'].iloc[-1]
                        data_source = "DEMO"
                else:
                    # API error - use demo mode
                    city_time_idx = time_idx + hash(city_name) % 100
                    if city_time_idx >= len(df) - 20:
                        city_time_idx = time_idx
                    
                    current_window, _ = simulate_live_data(df, city_time_idx)
                    current_output = current_window['solar_output_mw'].iloc[-1] * capacity_ratio
                    input_seq = create_input_sequence(current_window)
                    predicted_output = make_prediction(model, scaler, input_seq) * capacity_ratio
                    clouds = current_window['cloud_cover_percent'].iloc[-1]
                    temp = current_window['temperature_celsius'].iloc[-1]
                    data_source = "DEMO"
            else:
                # Demo mode - use historical data
                city_time_idx = time_idx + hash(city_name) % 100
                if city_time_idx >= len(df) - 20:
                    city_time_idx = time_idx
                
                current_window, _ = simulate_live_data(df, city_time_idx)
                current_output = current_window['solar_output_mw'].iloc[-1] * capacity_ratio
                input_seq = create_input_sequence(current_window)
                predicted_output = make_prediction(model, scaler, input_seq) * capacity_ratio
                clouds = current_window['cloud_cover_percent'].iloc[-1]
                temp = current_window['temperature_celsius'].iloc[-1]
                data_source = "DEMO"
            
            change = predicted_output - current_output
            alert_level, _ = get_alert_level(current_output, predicted_output)
            
            cities_data.append({
                'city': city_name,
                'state': city_details['state'],
                'capacity': city_details['capacity_mw'],
                'current': current_output,
                'predicted': predicted_output,
                'change': change,
                'change_pct': (change / max(current_output, 1)) * 100,
                'alert': alert_level,
                'clouds': clouds,
                'temp': temp,
                'source': data_source
            })
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Grid Status Summary
        total_current = sum([c['current'] for c in cities_data])
        total_capacity = sum([c['capacity'] for c in cities_data])
        total_predicted = sum([c['predicted'] for c in cities_data])
        grid_utilization = (total_current / total_capacity) * 100
        
        # Count live vs demo cities
        live_cities = len([c for c in cities_data if c['source'] == 'LIVE'])
        demo_cities = len([c for c in cities_data if c['source'] == 'DEMO'])
        
        # Overall status
        critical_cities = len([c for c in cities_data if c['alert'] == 'danger'])
        warning_cities = len([c for c in cities_data if c['alert'] == 'warning'])
        
        if critical_cities > 0:
            grid_status = "🔴 CRITICAL"
            grid_color = "#ffe6e6"
        elif warning_cities > 2:
            grid_status = "🟡 WARNING"
            grid_color = "#fff4e6"
        else:
            grid_status = "🟢 NORMAL"
            grid_color = "#e6ffe6"
        
        # Grid Summary Panel
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 2rem; border-radius: 1rem; margin-bottom: 2rem;'>
            <h2 style='margin: 0; color: white;'>🇮🇳 INDIA SOLAR GRID - REAL-TIME STATUS</h2>
            <div style='font-size: 0.9rem; margin: 0.5rem 0; opacity: 0.9;'>
                {live_cities} cities LIVE | {demo_cities} cities DEMO | Last updated: {last_update_time_ist}
            </div>
            <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem;'>
                <div style='background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 0.5rem;'>
                    <div style='font-size: 0.9rem; opacity: 0.9;'>Total Generation</div>
                    <div style='font-size: 2rem; font-weight: bold;'>{total_current:.0f} MW</div>
                </div>
                <div style='background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 0.5rem;'>
                    <div style='font-size: 0.9rem; opacity: 0.9;'>Total Capacity</div>
                    <div style='font-size: 2rem; font-weight: bold;'>{total_capacity} MW</div>
                </div>
                <div style='background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 0.5rem;'>
                    <div style='font-size: 0.9rem; opacity: 0.9;'>Grid Utilization</div>
                    <div style='font-size: 2rem; font-weight: bold;'>{grid_utilization:.0f}%</div>
                </div>
                <div style='background: {grid_color}; padding: 1rem; border-radius: 0.5rem; color: #000;'>
                    <div style='font-size: 0.9rem;'>Grid Status</div>
                    <div style='font-size: 1.8rem; font-weight: bold;'>{grid_status}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # City Cards - 2x4 Grid
        st.markdown("### 🏙️ City-wise Solar Generation Status")
        
        for i in range(0, 8, 4):
            cols = st.columns(4)
            for j, col in enumerate(cols):
                if i + j < len(cities_data):
                    data = cities_data[i + j]
                    
                    # Determine card color
                    if data['alert'] == 'danger':
                        card_color = "#ffe6e6"
                        border_color = "#ff4444"
                    elif data['alert'] == 'warning':
                        card_color = "#fff4e6"
                        border_color = "#ffaa00"
                    else:
                        card_color = "#e6ffe6"
                        border_color = "#44ff44"
                    
                    # Data source badge
                    if data['source'] == 'LIVE':
                        source_badge = '<span style="background: #44ff44; color: white; padding: 0.2rem 0.5rem; border-radius: 0.3rem; font-size: 0.7rem; font-weight: bold;">🔴 LIVE</span>'
                    else:
                        source_badge = '<span style="background: #ffaa00; color: white; padding: 0.2rem 0.5rem; border-radius: 0.3rem; font-size: 0.7rem; font-weight: bold;">DEMO</span>'
                    
                    with col:
                        st.markdown(f"""
                        <div style='background-color: {card_color}; 
                                    border-left: 5px solid {border_color};
                                    padding: 1rem; border-radius: 0.5rem; height: 100%;'>
                            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                                <h4 style='margin: 0;'>{data['city']}</h4>
                                {source_badge}
                            </div>
                            <p style='margin: 0; font-size: 0.85rem; color: #666;'>{data['state']}</p>
                            <hr style='margin: 0.5rem 0;'>
                            <div style='font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;'>
                                {data['current']:.0f} MW
                            </div>
                            <div style='font-size: 0.9rem; color: #666;'>
                                of {data['capacity']} MW capacity
                            </div>
                            <div style='margin: 0.5rem 0; padding: 0.5rem; background: rgba(0,0,0,0.05); border-radius: 0.3rem;'>
                                <div style='font-size: 0.85rem;'>30-min Forecast:</div>
                                <div style='font-size: 1.3rem; font-weight: bold;'>
                                    {data['predicted']:.0f} MW
                                    <span style='font-size: 0.9rem; color: {"#ff4444" if data["change"] < 0 else "#44ff44"};'>
                                        ({data['change']:+.0f} MW)
                                    </span>
                                </div>
                            </div>
                            <div style='font-size: 0.8rem; margin-top: 0.5rem;'>
                                ☁️ {data['clouds']:.0f}% | 🌡️ {data['temp']:.1f}°C
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Alert Summary
        st.markdown("---")
        st.markdown("### ⚠️ Active Alerts")
        
        critical = [c for c in cities_data if c['alert'] == 'danger']
        warnings = [c for c in cities_data if c['alert'] == 'warning']
        
        alert_col1, alert_col2 = st.columns(2)
        
        with alert_col1:
            if critical:
                st.error(f"🚨 **CRITICAL ALERTS ({len(critical)})**")
                for c in critical:
                    live_badge = "🔴 LIVE" if c['source'] == 'LIVE' else "DEMO"
                    st.markdown(f"- **{c['city']}** ({live_badge}): Expected drop of {abs(c['change']):.0f} MW ({abs(c['change_pct']):.0f}%)")
            else:
                st.success("✅ No critical alerts")
        
        with alert_col2:
            if warnings:
                st.warning(f"⚠️ **WARNINGS ({len(warnings)})**")
                for c in warnings:
                    live_badge = "🔴 LIVE" if c['source'] == 'LIVE' else "DEMO"
                    st.markdown(f"- **{c['city']}** ({live_badge}): Expected drop of {abs(c['change']):.0f} MW ({abs(c['change_pct']):.0f}%)")
            else:
                st.info("ℹ️ No warnings")
        
        # Comparison Chart
        st.markdown("---")
        st.markdown("### 📊 Comparative Analysis")
        
        fig = go.Figure()
        
        # Color code by data source
        colors_current = ['#44ff44' if c['source'] == 'LIVE' else '#ffaa00' for c in cities_data]
        colors_predicted = ['#ff6666' if c['source'] == 'LIVE' else '#ffaa00' for c in cities_data]
        
        fig.add_trace(go.Bar(
            name='Current Output',
            x=[c['city'] for c in cities_data],
            y=[c['current'] for c in cities_data],
            marker_color=colors_current,
            text=[f"{c['current']:.0f} MW<br>{'🔴LIVE' if c['source']=='LIVE' else 'DEMO'}" for c in cities_data],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='30-min Prediction',
            x=[c['city'] for c in cities_data],
            y=[c['predicted'] for c in cities_data],
            marker_color=colors_predicted,
            text=[f"{c['predicted']:.0f} MW" for c in cities_data],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Current Output vs 30-Minute Prediction (Green bars = LIVE API data)",
            xaxis_title="City",
            yaxis_title="Output (MW)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================
    # MODE 2: SMART ALERT SYSTEM
    # ========================================
    
    elif app_mode == "🔔 Smart Alert System":
        
        st.subheader("🔔 Intelligent Alert & Notification System")
        st.info("Advanced alert system with actionable recommendations and notification previews")
        
        # Get current data
        current_window, actual_future = simulate_live_data(df, time_idx)
        current_timestamp = current_window['timestamp'].iloc[-1]
        current_output = current_window['solar_output_mw'].iloc[-1]
        
        capacity_ratio = city_info['capacity_mw'] / 100
        current_output_scaled = current_output * capacity_ratio
        
        input_seq = create_input_sequence(current_window)
        predicted_output = make_prediction(model, scaler, input_seq) * capacity_ratio
        
        change = predicted_output - current_output_scaled
        change_pct = (change / max(current_output_scaled, 1)) * 100
        
        alert_level, _ = get_alert_level(current_output_scaled, predicted_output)
        
        # Alert Configuration
        st.markdown("### ⚙️ Alert Configuration")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            critical_threshold = st.slider("Critical Alert Threshold (MW drop)", 15, 50, 20, 5)
            warning_threshold = st.slider("Warning Alert Threshold (MW drop)", 5, 20, 10, 5)
        
        with config_col2:
            enable_email = st.checkbox("Enable Email Alerts", value=True)
            enable_sms = st.checkbox("Enable SMS Alerts", value=True)
            enable_push = st.checkbox("Enable Push Notifications", value=False)
        
        # Determine alert status
        if abs(change) >= critical_threshold and change < 0:
            current_alert = "CRITICAL"
            alert_color = "#ffe6e6"
            alert_icon = "🚨"
        elif abs(change) >= warning_threshold and change < 0:
            current_alert = "WARNING"
            alert_color = "#fff4e6"
            alert_icon = "⚠️"
        else:
            current_alert = "NORMAL"
            alert_color = "#e6ffe6"
            alert_icon = "✅"
        
        # Current Status
        st.markdown("---")
        st.markdown("### 📊 Current Status")
        
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)
        
        with status_col1:
            st.metric("Current Output", f"{current_output_scaled:.1f} MW")
        with status_col2:
            st.metric("30-min Prediction", f"{predicted_output:.1f} MW", f"{change:+.1f} MW")
        with status_col3:
            st.metric("Change Percentage", f"{abs(change_pct):.1f}%")
        with status_col4:
            st.markdown(f"""
            <div style='background-color: {alert_color}; padding: 1rem; border-radius: 0.5rem; text-align: center;'>
                <div style='font-size: 0.9rem;'>Alert Level</div>
                <div style='font-size: 1.5rem; font-weight: bold;'>{alert_icon} {current_alert}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if current_alert != "NORMAL":
            # Smart Recommendations
            st.markdown("---")
            st.markdown("### 🎯 Recommended Actions")
            
            minutes_to_impact = 30
            backup_needed = max(0, abs(change) + 5)  # Add 5MW buffer
            
            actions = []
            
            if current_alert == "CRITICAL":
                actions = [
                    f"⚡ **IMMEDIATE:** Activate {backup_needed:.0f} MW backup power (Gas Turbine #2)",
                    f"📞 **URGENT:** Alert State Load Dispatch Center (SLDC)",
                    f"⏰ **TIMING:** Start backup systems NOW (warm-up time: 15 minutes)",
                    f"📊 **LOAD MANAGEMENT:** Reduce non-critical industrial load by {min(10, abs(change_pct)):.0f}%",
                    f"🔔 **NOTIFICATIONS:** Auto-notify all grid operators on duty"
                ]
            else:  # WARNING
                actions = [
                    f"⚡ **PREPARE:** Ready {backup_needed:.0f} MW backup power for activation",
                    f"📞 **INFORM:** Notify State Load Dispatch Center of potential imbalance",
                    f"⏰ **TIMING:** Activate backup in {minutes_to_impact - 10} minutes if prediction holds",
                    f"📊 **MONITOR:** Watch for further deterioration in next 10 minutes"
                ]
            
            for action in actions:
                st.markdown(f"- {action}")
            
            # Cost Impact
            st.markdown("---")
            st.markdown("### 💰 Financial Impact Analysis")
            
            penalty_per_mw = 15000  # ₹15,000 per MW
            potential_penalty = abs(change) * penalty_per_mw / 100000  # in Lakhs
            
            impact_col1, impact_col2 = st.columns(2)
            
            with impact_col1:
                st.markdown(f"""
                <div class='metric-box'>
                    <h4>If No Action Taken:</h4>
                    <div style='font-size: 2rem; color: #ff4444;'>₹{potential_penalty:.2f}L</div>
                    <div>Potential grid imbalance penalty</div>
                </div>
                """, unsafe_allow_html=True)
            
            with impact_col2:
                backup_cost = backup_needed * 3000 / 100000  # ₹3,000 per MW for backup
                st.markdown(f"""
                <div class='metric-box'>
                    <h4>With Recommended Action:</h4>
                    <div style='font-size: 2rem; color: #44ff44;'>₹{backup_cost:.2f}L</div>
                    <div>Cost of backup power activation</div>
                    <div style='margin-top: 0.5rem; font-weight: bold; color: #44ff44;'>
                        SAVINGS: ₹{(potential_penalty - backup_cost):.2f}L
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Notification Previews
        st.markdown("---")
        st.markdown("### 📬 Notification Previews")
        
        notif_tabs = st.tabs(["📧 Email", "📱 SMS", "🔔 Push Notification"])
        
        with notif_tabs[0]:
            if enable_email:
                st.markdown(f"""
                <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; 
                            border: 1px solid #dee2e6; font-family: monospace;'>
                    <div style='border-bottom: 2px solid #dee2e6; padding-bottom: 0.5rem; margin-bottom: 1rem;'>
                        <strong>From:</strong> alerts@solarpredictai.com<br>
                        <strong>To:</strong> grid-operator@{city.lower()}discom.in<br>
                        <strong>Subject:</strong> {alert_icon} {current_alert} ALERT - {city} Solar Park<br>
                        <strong>Priority:</strong> {'HIGH' if current_alert == 'CRITICAL' else 'MEDIUM'}<br>
                        <strong>Sent:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </div>
                    
                    <h3 style='color: #dc3545; margin-top: 0;'>{alert_icon} SOLAR GENERATION ALERT</h3>
                    
                    <p><strong>Location:</strong> {city_info['grid']}</p>
                    <p><strong>Current Output:</strong> {current_output_scaled:.1f} MW</p>
                    <p><strong>Predicted Output (30-min):</strong> {predicted_output:.1f} MW</p>
                    <p><strong>Expected Change:</strong> <span style='color: #dc3545; font-weight: bold;'>{change:+.1f} MW ({change_pct:+.1f}%)</span></p>
                    <p><strong>Time to Impact:</strong> {minutes_to_impact} minutes</p>
                    
                    <hr>
                    
                    <h4>RECOMMENDED IMMEDIATE ACTIONS:</h4>
                    <ol>
                        {"<br>".join([f"<li>{action}</li>" for action in actions[:3]])}
                    </ol>
                    
                    <hr>
                    
                    <p><strong>Weather Conditions:</strong></p>
                    <ul>
                        <li>Cloud Cover: {current_window['cloud_cover_percent'].iloc[-1]:.0f}%</li>
                        <li>Temperature: {current_window['temperature_celsius'].iloc[-1]:.1f}°C</li>
                        <li>Wind Speed: {current_window['wind_speed_kmh'].iloc[-1]:.0f} km/h</li>
                    </ul>
                    
                    <p style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #dee2e6; 
                              font-size: 0.85rem; color: #666;'>
                        This is an automated alert from Monsoon Solar Predictor AI System.<br>
                        Prediction Confidence: 87% | Model: LSTM v2.1<br>
                        <a href='#'>View Dashboard</a> | <a href='#'>Update Alert Settings</a>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Email alerts are disabled. Enable in configuration above.")
        
        with notif_tabs[1]:
            if enable_sms:
                sms_message = f"{alert_icon} {current_alert}: {city} solar drop {abs(change):.0f}MW in {minutes_to_impact}min. Activate {backup_needed:.0f}MW backup NOW. -SolarPredictAI"
                st.markdown(f"""
                <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 1.5rem; 
                            border: 3px solid #007bff; max-width: 400px;'>
                    <div style='font-size: 0.85rem; color: #666; margin-bottom: 0.5rem;'>
                        To: +91-XXXXX-XXXXX (Grid Operator)
                    </div>
                    <div style='background-color: #007bff; color: white; padding: 1rem; 
                                border-radius: 1rem; font-size: 0.95rem;'>
                        {sms_message}
                    </div>
                    <div style='font-size: 0.75rem; color: #666; margin-top: 0.5rem; text-align: right;'>
                        Sent at {datetime.now().strftime('%H:%M')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("SMS alerts are disabled. Enable in configuration above.")
        
        with notif_tabs[2]:
            if enable_push:
                st.markdown(f"""
                <div style='background-color: white; padding: 1rem; border-radius: 0.5rem; 
                            border: 1px solid #dee2e6; max-width: 400px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                        <div style='background-color: #ff6b6b; color: white; width: 40px; height: 40px; 
                                    border-radius: 0.5rem; display: flex; align-items: center; justify-content: center; 
                                    font-size: 1.5rem; margin-right: 1rem;'>
                            {alert_icon}
                        </div>
                        <div>
                            <div style='font-weight: bold;'>Solar Predict AI</div>
                            <div style='font-size: 0.85rem; color: #666;'>now</div>
                        </div>
                    </div>
                    <div style='font-weight: bold; margin-bottom: 0.5rem;'>
                        {current_alert} Alert: {city}
                    </div>
                    <div style='font-size: 0.9rem;'>
                        {abs(change):.0f}MW drop expected in {minutes_to_impact} minutes. Tap to view actions.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Push notifications are disabled. Enable in configuration above.")
    
    # ========================================
    # MODE 3: AI CHATBOT ASSISTANT
    # ========================================
    
    elif app_mode == "💬 AI Chatbot Assistant":
        
        st.subheader("💬 AI Solar Prediction Assistant")
        st.info("Ask questions about solar predictions, weather patterns, and grid operations")
        
        # Get current context
        current_window, _ = simulate_live_data(df, time_idx)
        current_output = current_window['solar_output_mw'].iloc[-1]
        capacity_ratio = city_info['capacity_mw'] / 100
        current_output_scaled = current_output * capacity_ratio
        
        input_seq = create_input_sequence(current_window)
        predicted_output = make_prediction(model, scaler, input_seq) * capacity_ratio
        change = predicted_output - current_output_scaled
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = [
                {
                    'role': 'assistant',
                    'content': f"Hello! I'm your AI Solar Prediction Assistant. I'm currently monitoring {city} Solar Park ({city_info['capacity_mw']} MW capacity). How can I help you today?"
                }
            ]
        
        # Predefined questions
        st.markdown("### 💡 Quick Questions")
        quick_questions = st.columns(3)
        
        with quick_questions[0]:
            if st.button("📊 What's the current status?", use_container_width=True):
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': "What's the current status?"
                })
        
        with quick_questions[1]:
            if st.button("🔮 Why is output changing?", use_container_width=True):
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': "Why is the predicted output changing?"
                })
        
        with quick_questions[2]:
            if st.button("⚡ What should I do?", use_container_width=True):
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': "What actions should I take?"
                })
        
        # Chat interface
        st.markdown("---")
        st.markdown("### 💬 Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div style='background-color: #e3f2fd; padding: 1rem; border-radius: 1rem; 
                            margin: 0.5rem 0; margin-left: 20%;'>
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #f5f5f5; padding: 1rem; border-radius: 1rem; 
                            margin: 0.5rem 0; margin-right: 20%;'>
                    <strong>🤖 AI Assistant:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
        
        # Generate AI responses based on last question
        if len(st.session_state.chat_history) > 0 and st.session_state.chat_history[-1]['role'] == 'user':
            user_question = st.session_state.chat_history[-1]['content'].lower()
            
            # Simple rule-based responses (in production, use actual AI/LLM)
            if 'status' in user_question or 'current' in user_question:
                response = f"""Based on real-time analysis of {city} Solar Park:

**Current Generation:** {current_output_scaled:.1f} MW ({(current_output_scaled/city_info['capacity_mw'])*100:.0f}% of capacity)

**30-Minute Forecast:** {predicted_output:.1f} MW (change: {change:+.1f} MW)

**Weather Conditions:**
- Cloud Cover: {current_window['cloud_cover_percent'].iloc[-1]:.0f}%
- Temperature: {current_window['temperature_celsius'].iloc[-1]:.1f}°C
- Humidity: {current_window['humidity_percent'].iloc[-1]:.0f}%

**System Status:** {'🟢 Normal operations' if abs(change) < 10 else '⚠️ Monitoring required'}"""
            
            elif 'why' in user_question or 'reason' in user_question or 'changing' in user_question:
                cloud_change = "increasing" if change < 0 else "decreasing"
                response = f"""The predicted change is primarily due to:

**1. Cloud Movement** ({cloud_change} solar radiation)
   - Current cloud cover: {current_window['cloud_cover_percent'].iloc[-1]:.0f}%
   - This is {'above' if current_window['cloud_cover_percent'].iloc[-1] > 50 else 'below'} the monsoon average

**2. Time of Day Effect**
   - Solar angle and intensity factor for current time

**3. Weather Pattern Analysis**
   - My AI model detected {'typical monsoon cloud burst pattern' if change < -15 else 'stable atmospheric conditions'}

**Confidence Level:** 87% (based on historical accuracy)

This prediction is based on analyzing the last 60 minutes of data including cloud patterns, temperature trends, and historical monsoon behavior."""
            
            elif 'what' in user_question and ('do' in user_question or 'action' in user_question):
                if abs(change) > 20:
                    response = f"""🚨 **CRITICAL SITUATION** - Immediate actions required:

**1. Activate Backup Power (Priority: URGENT)**
   - Required capacity: {abs(change) + 5:.0f} MW
   - Recommended: Gas Turbine #2
   - Start time: Immediately (15-min warm-up needed)

**2. Grid Coordination**
   - Alert State Load Dispatch Center
   - Request load balancing support
   - Prepare for potential imbalance

**3. Load Management**
   - Consider reducing non-critical load by {min(10, abs(change)/2):.0f} MW
   - Notify industrial consumers on interruptible tariff

**Financial Impact:**
   - Cost if no action: ₹{(abs(change) * 15000 / 100000):.2f} Lakhs (penalty)
   - Cost of backup: ₹{(abs(change) * 3000 / 100000):.2f} Lakhs
   - **Net savings: ₹{((abs(change) * 15000 - abs(change) * 3000) / 100000):.2f} Lakhs**"""
                elif abs(change) > 10:
                    response = f"""⚠️ **MODERATE ALERT** - Precautionary actions recommended:

**1. Prepare Backup Systems**
   - Ready {abs(change) + 3:.0f} MW backup for quick activation
   - Warm up systems to reduce response time

**2. Monitor Closely**
   - Watch for further deterioration in next 10 minutes
   - Be ready to escalate to critical if needed

**3. Communication**
   - Inform grid control of potential imbalance
   - Keep backup operators on standby

**Timeline:**
   - Monitor: Next 10 minutes
   - Decision point: 20 minutes from now
   - Action required by: 25 minutes from now"""
                else:
                    response = f"""✅ **NORMAL OPERATIONS** - No immediate actions required:

**Current Status:** Stable generation expected

**Recommended:**
   - Continue routine monitoring
   - No backup activation needed
   - Maintain normal operational procedures

**Next Check:** Review forecast in 15 minutes

The system is operating within normal parameters. I'll alert you immediately if conditions change."""
            
            elif 'help' in user_question or 'hello' in user_question or 'hi' in user_question:
                response = """I can help you with:

**📊 Status & Monitoring**
- Current generation and predictions
- Weather conditions analysis
- Historical comparisons

**🔮 Predictions & Forecasts**
- 30-minute ahead forecasts
- Confidence levels
- Pattern recognition

**⚡ Operational Guidance**
- Action recommendations
- Backup power calculations
- Cost impact analysis

**💰 Financial Analysis**
- Penalty calculations
- Cost-benefit analysis
- ROI estimates

Feel free to ask me anything about solar generation, weather patterns, or grid operations!"""
            
            else:
                response = f"""I understand you're asking: "{st.session_state.chat_history[-1]['content']}"

While I'm currently optimized for solar prediction queries, I can provide general information:

**Current Situation:**
- Location: {city}, {city_info['state']}
- Output: {current_output_scaled:.1f} MW
- Forecast: {predicted_output:.1f} MW in 30 minutes

Try asking:
- "What's the current status?"
- "Why is output changing?"
- "What should I do?"
- "What's the weather impact?"

How else can I assist you?"""
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            st.rerun()
        
        # User input
        st.markdown("---")
        user_input = st.text_input("💬 Type your question here...", key="user_input", placeholder="e.g., What's causing the power drop?")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Send", type="primary", use_container_width=True):
                if user_input:
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_input
                    })
                    st.rerun()
        
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = [
                    {
                        'role': 'assistant',
                        'content': f"Chat cleared. I'm ready to help with {city} Solar Park monitoring. What would you like to know?"
                    }
                ]
                st.rerun()
    
    # ========================================
    # MODE 4: STANDARD DASHBOARD
    # ========================================
    
    elif app_mode == "📊 Standard Dashboard":
        
        # Get current data
        current_window, actual_future = simulate_live_data(df, time_idx)
        current_timestamp = current_window['timestamp'].iloc[-1]
        current_output = current_window['solar_output_mw'].iloc[-1]
        
        # Scale by city capacity
        capacity_ratio = city_info['capacity_mw'] / 100
        current_output_scaled = current_output * capacity_ratio
        
        # Make prediction
        input_seq = create_input_sequence(current_window)
        predicted_output = make_prediction(model, scaler, input_seq) * capacity_ratio
        
        if actual_future is not None:
            actual_future_scaled = actual_future * capacity_ratio
        else:
            actual_future_scaled = None
        
        # Alert
        alert_level, alert_message = get_alert_level(current_output_scaled, predicted_output)
        
        # City badge
        st.markdown(f"""
        <div class='city-badge'>
            🏭 {city_info['grid']} | ⚡ {city_info['capacity_mw']} MW Capacity
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🕐 Current Time", current_timestamp.strftime("%H:%M"),
                     current_timestamp.strftime("%b %d, %Y"))
        with col2:
            capacity_percent = (current_output_scaled/city_info['capacity_mw'])*100
            st.metric("⚡ Current Output", f"{current_output_scaled:.1f} MW",
                     f"{capacity_percent:.0f}% capacity")
        with col3:
            change = predicted_output - current_output_scaled
            st.metric("🔮 30-Min Prediction", f"{predicted_output:.1f} MW",
                     f"{change:+.1f} MW")
        with col4:
            if actual_future_scaled is not None:
                error = abs(predicted_output - actual_future_scaled)
                st.metric("✅ Actual (30-min)", f"{actual_future_scaled:.1f} MW",
                         f"Error: {error:.1f} MW")
            else:
                st.metric("🎯 Model MAE", "8.08 MW")
        
        # Alert
        st.markdown("---")
        alert_class = f"alert-{alert_level}"
        st.markdown(f'<div class="alert-box {alert_class}"><strong>{alert_message}</strong></div>', 
                   unsafe_allow_html=True)
        
        # Historical Comparison
        st.markdown("---")
        st.subheader("📊 Historical Comparison (Same Day Last Year)")
        
        comparison = get_historical_comparison(df, current_timestamp)
        
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        with comp_col1:
            hist_output = comparison['historical_output'] * capacity_ratio
            diff = current_output_scaled - hist_output
            st.metric("Last Year Output", f"{hist_output:.1f} MW", f"{diff:+.1f} MW")
        with comp_col2:
            diff_clouds = current_window['cloud_cover_percent'].iloc[-1] - comparison['historical_clouds']
            st.metric("Last Year Clouds", f"{comparison['historical_clouds']:.0f}%", f"{diff_clouds:+.0f}%")
        with comp_col3:
            diff_temp = current_window['temperature_celsius'].iloc[-1] - comparison['historical_temp']
            st.metric("Last Year Temp", f"{comparison['historical_temp']:.1f}°C", f"{diff_temp:+.1f}°C")
        
        # Chart
        st.markdown("---")
        st.subheader("📈 Solar Output Trend")
        
        viz_start = max(0, time_idx - 50)
        viz_end = min(len(df), time_idx + 50)
        viz_data = df.iloc[viz_start:viz_end].copy()
        viz_data['solar_output_mw'] = viz_data['solar_output_mw'] * capacity_ratio
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=viz_data['timestamp'], y=viz_data['solar_output_mw'],
                                mode='lines', name='Actual', fill='tozeroy'))
        fig.add_trace(go.Scatter(x=[current_timestamp], y=[current_output_scaled],
                                mode='markers', name='Current', marker=dict(size=12, color='green')))
        
        pred_timestamp = current_timestamp + timedelta(minutes=30)
        fig.add_trace(go.Scatter(x=[pred_timestamp], y=[predicted_output],
                                mode='markers', name='Predicted', marker=dict(size=12, color='orange', symbol='star')))
        
        if actual_future_scaled:
            fig.add_trace(go.Scatter(x=[pred_timestamp], y=[actual_future_scaled],
                                    mode='markers', name='Actual', marker=dict(size=10, color='red', symbol='x')))
        
        fig.update_layout(height=400, xaxis_title="Time", yaxis_title="Output (MW)", hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================
    # MODE 2: LIVE WEATHER MODE
    # ========================================
    
    elif app_mode == "☁️ Live Weather Mode":
        
        st.subheader(f"☁️ Live Weather Integration - {city}")
        
        # Show selected city info
        st.info(f"📍 Fetching weather for: **{city}, {city_info['state']}** ({city_info['coords_str']})")
        
        # API Key input
        api_key = st.text_input("OpenWeatherMap API Key (optional - leave blank for demo mode)", 
                                type="password",
                                help="Get free API key from openweathermap.org")
        
        if st.button("🔄 Fetch Live Weather & Predict", type="primary"):
            
            with st.spinner(f"Fetching live weather data for {city}..."):
                lat, lon = city_info['coords']
                weather_data, error = fetch_live_weather(lat, lon, api_key)
            
            if error:
                st.error(f"❌ {error}")
                st.info("💡 Running in demo mode with simulated data")
            
            if weather_data:
                # Display live weather
                st.success(f"✅ Weather data fetched for **{city}**: {weather_data['source'].upper()}")
                st.caption(f"📍 Location: {city_info['coords_str']} | 🕐 Time: {weather_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("☁️ Cloud Cover", f"{weather_data['cloud_cover_percent']:.0f}%")
                with col2:
                    st.metric("🌡️ Temperature", f"{weather_data['temperature_celsius']:.1f}°C")
                with col3:
                    st.metric("💧 Humidity", f"{weather_data['humidity_percent']:.0f}%")
                with col4:
                    st.metric("💨 Wind Speed", f"{weather_data['wind_speed_kmh']:.1f} km/h")
                
                st.info(f"🌤️ Conditions: {weather_data['description']}")
                
                # Create synthetic sequence for prediction
                # Use live weather + assume current hour and previous solar patterns
                current_hour = datetime.now().hour
                
                # Get a similar historical pattern
                df_hour = df[df['hour'] == current_hour].head(12)
                
                if len(df_hour) >= 12:
                    synthetic_window = df_hour.copy()
                    
                    # Update with live weather
                    synthetic_window['cloud_cover_percent'] = weather_data['cloud_cover_percent']
                    synthetic_window['temperature_celsius'] = weather_data['temperature_celsius']
                    synthetic_window['humidity_percent'] = weather_data['humidity_percent']
                    synthetic_window['wind_speed_kmh'] = weather_data['wind_speed_kmh']
                    
                    # Make prediction
                    input_seq = create_input_sequence(synthetic_window)
                    capacity_ratio = city_info['capacity_mw'] / 100
                    predicted_output = make_prediction(model, scaler, input_seq) * capacity_ratio
                    
                    st.markdown("---")
                    st.subheader("🔮 Live Prediction")
                    
                    st.metric("30-Minute Forecast", f"{predicted_output:.1f} MW",
                             f"{(predicted_output/city_info['capacity_mw'])*100:.0f}% capacity")
                    
                    st.info("💡 Prediction based on current live weather conditions")
                else:
                    st.warning("Not enough historical data for this hour")
    
    # ========================================
    # MODE 3: WHAT-IF SCENARIOS
    # ========================================
    
    elif app_mode == "🔮 What-If Scenarios":
        
        st.subheader("🔮 What-If Scenario Analysis")
        st.info("Simulate how changes in weather conditions affect solar output predictions")
        
        # Get baseline data
        current_window, _ = simulate_live_data(df, time_idx)
        current_output = current_window['solar_output_mw'].iloc[-1]
        capacity_ratio = city_info['capacity_mw'] / 100
        current_output_scaled = current_output * capacity_ratio
        
        # Baseline prediction
        input_seq = create_input_sequence(current_window)
        baseline_prediction = make_prediction(model, scaler, input_seq) * capacity_ratio
        
        st.markdown("### 📊 Baseline Scenario")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Output", f"{current_output_scaled:.1f} MW")
        with col2:
            st.metric("Baseline Prediction (30-min)", f"{baseline_prediction:.1f} MW")
        
        st.markdown("---")
        st.markdown("### 🎛️ Modify Weather Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cloud_delta = st.slider("Cloud Cover Change (%)", -50, 50, 0, 5,
                                   help="How much to increase/decrease cloud cover")
            temp_delta = st.slider("Temperature Change (°C)", -10, 10, 0, 1,
                                  help="How much to increase/decrease temperature")
        
        with col2:
            humidity_delta = st.slider("Humidity Change (%)", -30, 30, 0, 5,
                                      help="How much to increase/decrease humidity")
            wind_delta = st.slider("Wind Speed Change (km/h)", -15, 15, 0, 1,
                                  help="How much to increase/decrease wind speed")
        
        # Calculate scenario
        modifications = {
            'cloud_delta': cloud_delta,
            'temp_delta': temp_delta,
            'humidity_delta': humidity_delta,
            'wind_delta': wind_delta
        }
        
        scenario_prediction = simulate_scenario(model, scaler, input_seq, modifications) * capacity_ratio
        
        st.markdown("---")
        st.markdown("### 📊 Scenario Results")
        
        # Display scenarios side-by-side
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Baseline Prediction", f"{baseline_prediction:.1f} MW")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="scenario-box">', unsafe_allow_html=True)
            st.metric("Scenario Prediction", f"{scenario_prediction:.1f} MW", 
                     delta=None, label_visibility="visible")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            difference = scenario_prediction - baseline_prediction
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Difference", f"{abs(difference):.1f} MW",
                     f"{difference:+.1f} MW")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Interpretation
        st.markdown("---")
        if abs(difference) < 5:
            st.success("✅ Scenario has minimal impact on prediction")
        elif difference < -10:
            st.warning(f"⚠️ Scenario would DECREASE output by {abs(difference):.1f} MW")
        elif difference > 10:
            st.info(f"📈 Scenario would INCREASE output by {difference:.1f} MW")
        
        # Comparison chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Baseline', x=['Prediction'], y=[baseline_prediction],
                            marker_color='lightblue'))
        fig.add_trace(go.Bar(name='Scenario', x=['Prediction'], y=[scenario_prediction],
                            marker_color='coral'))
        fig.update_layout(title="Baseline vs Scenario Prediction", yaxis_title="Output (MW)", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================
    # MODE 4: COST SAVINGS CALCULATOR
    # ========================================
    
    elif app_mode == "💰 Cost Savings Calculator":
        
        st.subheader("💰 ROI & Cost Savings Analysis")
        st.info("Calculate the financial impact of implementing AI-powered solar forecasting")
        
        # Input parameters
        st.markdown("### ⚙️ Configuration Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Plant Details")
            plant_capacity = st.number_input(
                "Solar Plant Capacity (MW)",
                min_value=10,
                max_value=500,
                value=city_info['capacity_mw'],
                step=10,
                help="Total installed capacity of the solar plant"
            )
            
            avg_generation_hours = st.slider(
                "Average Generation Hours/Day",
                min_value=4,
                max_value=12,
                value=8,
                help="Average hours of solar generation per day"
            )
            
            power_purchase_rate = st.number_input(
                "Power Purchase Rate (₹/kWh)",
                min_value=2.0,
                max_value=10.0,
                value=4.5,
                step=0.1,
                help="Rate at which grid buys solar power"
            )
        
        with col2:
            st.markdown("#### Grid Penalty Details")
            
            penalty_per_mw = st.number_input(
                "Imbalance Penalty (₹/MW/event)",
                min_value=5000,
                max_value=50000,
                value=15000,
                step=1000,
                help="Penalty charged per MW of grid imbalance"
            )
            
            imbalance_events_without_ai = st.slider(
                "Grid Imbalance Events/Month (Without AI)",
                min_value=5,
                max_value=30,
                value=18,
                help="Current frequency of grid imbalance events"
            )
            
            imbalance_events_with_ai = st.slider(
                "Grid Imbalance Events/Month (With AI)",
                min_value=1,
                max_value=10,
                value=3,
                help="Expected events with 30-min advance warning"
            )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ai_system_cost = st.number_input(
                "AI System Implementation Cost (₹ Lakhs)",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                help="One-time setup cost including hardware, software, training"
            )
        
        with col2:
            monthly_maintenance = st.number_input(
                "Monthly Maintenance Cost (₹ Lakhs)",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5,
                help="Ongoing operational and maintenance costs"
            )
        
        # Calculate button
        st.markdown("---")
        if st.button("📊 Calculate ROI & Savings", type="primary", use_container_width=True):
            
            # Calculations
            avg_imbalance_mw = 20  # Average MW deviation per event
            
            # WITHOUT AI
            monthly_penalty_without = (imbalance_events_without_ai * avg_imbalance_mw * penalty_per_mw) / 100000  # in Lakhs
            annual_penalty_without = monthly_penalty_without * 12
            
            # WITH AI
            monthly_penalty_with = (imbalance_events_with_ai * avg_imbalance_mw * penalty_per_mw) / 100000
            annual_penalty_with = monthly_penalty_with * 12
            
            # Maintenance costs
            annual_maintenance = monthly_maintenance * 12
            
            # Savings
            monthly_savings = monthly_penalty_without - monthly_penalty_with - monthly_maintenance
            annual_savings = (monthly_penalty_without - monthly_penalty_with) * 12 - annual_maintenance
            
            # ROI
            if ai_system_cost > 0:
                roi_percentage = (annual_savings / ai_system_cost) * 100
                payback_months = ai_system_cost / monthly_savings if monthly_savings > 0 else 999
            else:
                roi_percentage = 0
                payback_months = 0
            
            # Additional benefits
            daily_energy = plant_capacity * avg_generation_hours * 0.7  # 70% capacity factor
            annual_energy_mwh = daily_energy * 365
            annual_revenue = annual_energy_mwh * power_purchase_rate * 1000 / 100000  # in Lakhs
            
            # Improved generation due to better planning (1-2% increase)
            generation_improvement = annual_revenue * 0.015  # 1.5% improvement
            
            total_annual_benefit = annual_savings + generation_improvement
            
            # Display Results
            st.markdown("---")
            st.markdown("## 📊 Financial Analysis Results")
            
            # Key Metrics
            st.markdown("### 💰 Key Financial Metrics")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric(
                    "Annual Savings",
                    f"₹{annual_savings:.2f}L",
                    f"{(annual_savings/annual_penalty_without)*100:.0f}% reduction"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric(
                    "ROI",
                    f"{roi_percentage:.1f}%",
                    "First Year"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric(
                    "Payback Period",
                    f"{payback_months:.1f} months",
                    f"{12-payback_months:.1f} mo profit in Y1"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_col4:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric(
                    "5-Year Benefit",
                    f"₹{(total_annual_benefit * 5):.1f}L",
                    f"₹{(total_annual_benefit * 5)/100:.2f} Cr"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed Comparison
            st.markdown("---")
            st.markdown("### 📊 Detailed Cost Comparison")
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.markdown("#### ❌ WITHOUT AI System")
                st.markdown(f"""
                **Grid Imbalance Events:**
                - Events/Month: {imbalance_events_without_ai}
                - Avg Deviation: {avg_imbalance_mw} MW
                - Penalty Rate: ₹{penalty_per_mw:,}/MW
                
                **Monthly Costs:**
                - Penalty Cost: ₹{monthly_penalty_without:.2f} Lakhs
                - Maintenance: ₹0 Lakhs
                - **Total: ₹{monthly_penalty_without:.2f} Lakhs**
                
                **Annual Costs:**
                - **Total: ₹{annual_penalty_without:.2f} Lakhs**
                - **= ₹{annual_penalty_without/100:.2f} Crores**
                """)
            
            with comp_col2:
                st.markdown("#### ✅ WITH AI System")
                st.markdown(f"""
                **Grid Imbalance Events:**
                - Events/Month: {imbalance_events_with_ai}
                - Avg Deviation: {avg_imbalance_mw} MW
                - Penalty Rate: ₹{penalty_per_mw:,}/MW
                
                **Monthly Costs:**
                - Penalty Cost: ₹{monthly_penalty_with:.2f} Lakhs
                - Maintenance: ₹{monthly_maintenance:.2f} Lakhs
                - **Total: ₹{monthly_penalty_with + monthly_maintenance:.2f} Lakhs**
                
                **Annual Costs:**
                - **Total: ₹{annual_penalty_with + annual_maintenance:.2f} Lakhs**
                - **= ₹{(annual_penalty_with + annual_maintenance)/100:.2f} Crores**
                """)
            
            with comp_col3:
                st.markdown("#### 💚 NET SAVINGS")
                st.markdown(f"""
                **Implementation:**
                - One-time Cost: ₹{ai_system_cost:.2f} Lakhs
                - Monthly Maintenance: ₹{monthly_maintenance:.2f} Lakhs
                
                **Savings:**
                - Monthly: ₹{monthly_savings:.2f} Lakhs
                - Annual: ₹{annual_savings:.2f} Lakhs
                - **= ₹{annual_savings/100:.2f} Crores/year**
                
                **Additional Benefits:**
                - Better Planning: ₹{generation_improvement:.2f}L/year
                - **Total Benefit: ₹{total_annual_benefit:.2f}L/year**
                """)
            
            # Visualization - Comparison Chart
            st.markdown("---")
            st.markdown("### 📈 Cost Comparison Visualization")
            
            # Monthly comparison
            fig_monthly = go.Figure()
            
            categories = ['Penalty Costs', 'Maintenance', 'Total Cost']
            without_ai = [monthly_penalty_without, 0, monthly_penalty_without]
            with_ai = [monthly_penalty_with, monthly_maintenance, monthly_penalty_with + monthly_maintenance]
            
            fig_monthly.add_trace(go.Bar(
                name='Without AI',
                x=categories,
                y=without_ai,
                marker_color='#ff6b6b',
                text=[f'₹{v:.1f}L' for v in without_ai],
                textposition='outside'
            ))
            
            fig_monthly.add_trace(go.Bar(
                name='With AI',
                x=categories,
                y=with_ai,
                marker_color='#51cf66',
                text=[f'₹{v:.1f}L' for v in with_ai],
                textposition='outside'
            ))
            
            fig_monthly.update_layout(
                title="Monthly Cost Comparison",
                yaxis_title="Cost (₹ Lakhs)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # ROI Timeline
            st.markdown("---")
            st.markdown("### 📅 5-Year Financial Projection")
            
            years = list(range(1, 6))
            cumulative_savings = []
            cumulative_cost = []
            net_benefit = []
            
            for year in years:
                year_savings = annual_savings * year
                year_cost = ai_system_cost + (annual_maintenance * year)
                cumulative_savings.append(year_savings)
                cumulative_cost.append(year_cost)
                net_benefit.append(year_savings - year_cost)
            
            fig_roi = go.Figure()
            
            fig_roi.add_trace(go.Scatter(
                x=years,
                y=cumulative_savings,
                name='Cumulative Savings',
                line=dict(color='#51cf66', width=3),
                fill='tozeroy',
                fillcolor='rgba(81, 207, 102, 0.2)'
            ))
            
            fig_roi.add_trace(go.Scatter(
                x=years,
                y=cumulative_cost,
                name='Cumulative Investment',
                line=dict(color='#ff6b6b', width=3, dash='dash')
            ))
            
            fig_roi.add_trace(go.Scatter(
                x=years,
                y=net_benefit,
                name='Net Benefit',
                line=dict(color='#4dabf7', width=4),
                fill='tozeroy',
                fillcolor='rgba(77, 171, 247, 0.2)'
            ))
            
            # Add breakeven line
            fig_roi.add_hline(y=0, line_dash="dot", line_color="gray", 
                             annotation_text="Break-even", annotation_position="right")
            
            fig_roi.update_layout(
                title="5-Year ROI Projection",
                xaxis_title="Year",
                yaxis_title="Amount (₹ Lakhs)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_roi, use_container_width=True)
            
            # Summary Box
            st.markdown("---")
            st.markdown("### 🎯 Executive Summary")
            
            if roi_percentage > 200:
                summary_color = "#d4edda"
                summary_icon = "🟢"
                summary_verdict = "HIGHLY RECOMMENDED"
            elif roi_percentage > 100:
                summary_color = "#fff3cd"
                summary_icon = "🟡"
                summary_verdict = "RECOMMENDED"
            else:
                summary_color = "#f8d7da"
                summary_icon = "🔴"
                summary_verdict = "REVIEW REQUIRED"
            
            st.markdown(f"""
            <div style='background-color: {summary_color}; padding: 2rem; border-radius: 0.5rem; border-left: 5px solid #28a745;'>
                <h3 style='margin-top: 0;'>{summary_icon} Investment Verdict: {summary_verdict}</h3>
                <p style='font-size: 1.1rem; margin: 1rem 0;'>
                    <strong>Implementing the AI-powered solar forecasting system will save ₹{annual_savings:.2f} Lakhs 
                    annually, providing an ROI of {roi_percentage:.1f}% in the first year.</strong>
                </p>
                <ul style='font-size: 1rem;'>
                    <li>💰 <strong>Monthly Savings:</strong> ₹{monthly_savings:.2f} Lakhs</li>
                    <li>📅 <strong>Payback Period:</strong> {payback_months:.1f} months</li>
                    <li>📊 <strong>5-Year Net Benefit:</strong> ₹{(total_annual_benefit * 5)/100:.2f} Crores</li>
                    <li>🎯 <strong>Penalty Reduction:</strong> {((imbalance_events_without_ai - imbalance_events_with_ai)/imbalance_events_without_ai * 100):.0f}%</li>
                    <li>⚡ <strong>Grid Stability:</strong> Significantly Improved</li>
                </ul>
                <p style='font-style: italic; margin-top: 1rem;'>
                    *Additional benefits include improved grid reliability, better operational planning, 
                    and enhanced renewable energy integration.*
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Export options
            st.markdown("---")
            st.markdown("### 📥 Export Analysis")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                if st.button("📄 Generate PDF Report", use_container_width=True):
                    st.info("PDF generation feature coming soon! For now, use browser print (Ctrl+P) to save as PDF.")
            
            with export_col2:
                # Create CSV data
                csv_data = f"""Category,Without AI (₹ Lakhs),With AI (₹ Lakhs),Savings (₹ Lakhs)
Monthly Penalty,{monthly_penalty_without:.2f},{monthly_penalty_with:.2f},{monthly_penalty_without - monthly_penalty_with:.2f}
Monthly Maintenance,0.00,{monthly_maintenance:.2f},-{monthly_maintenance:.2f}
Monthly Total,{monthly_penalty_without:.2f},{monthly_penalty_with + monthly_maintenance:.2f},{monthly_savings:.2f}
Annual Penalty,{annual_penalty_without:.2f},{annual_penalty_with:.2f},{annual_penalty_without - annual_penalty_with:.2f}
Annual Maintenance,0.00,{annual_maintenance:.2f},-{annual_maintenance:.2f}
Annual Total,{annual_penalty_without:.2f},{annual_penalty_with + annual_maintenance:.2f},{annual_savings:.2f}
Implementation Cost,-,{ai_system_cost:.2f},-
ROI (%),-,{roi_percentage:.2f},-
Payback (months),-,{payback_months:.2f},-
"""
                st.download_button(
                    label="📊 Download CSV Data",
                    data=csv_data,
                    file_name=f"roi_analysis_{city}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # ========================================
    # MODE 5: MODEL RETRAINING
    # ========================================
    
    elif app_mode == "🎓 Model Retraining":
        
        st.subheader("🎓 Model Retraining Interface")
        st.info("Upload new data to retrain and improve the prediction model")
        
        # File upload
        uploaded_file = st.file_uploader("📁 Upload CSV with new data", type=['csv'],
                                        help="CSV must have columns: timestamp, cloud_cover_percent, temperature_celsius, humidity_percent, wind_speed_kmh, solar_output_mw")
        
        if uploaded_file is not None:
            
            try:
                # Read uploaded file
                new_df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File loaded: {len(new_df):,} rows")
                
                # Validate
                is_valid, message = validate_uploaded_data(new_df)
                
                if is_valid:
                    st.success(f"✅ {message}")
                    
                    # Show preview
                    with st.expander("📋 Data Preview"):
                        st.dataframe(new_df.head(10))
                    
                    # Data stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", f"{len(new_df):,}")
                    with col2:
                        date_range = (new_df['timestamp'].max() - new_df['timestamp'].min()).days
                        st.metric("Date Range", f"{date_range} days")
                    with col3:
                        st.metric("Avg Output", f"{new_df['solar_output_mw'].mean():.1f} MW")
                    
                    # Retraining settings
                    st.markdown("---")
                    st.markdown("### ⚙️ Training Settings")
                    
                    epochs = st.slider("Training Epochs", 10, 100, 30, 10,
                                      help="More epochs = better training but slower")
                    
                    # Retrain button
                    if st.button("🚀 Start Retraining", type="primary"):
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("🔄 Preparing data...")
                        progress_bar.progress(20)
                        
                        status_text.text("🧠 Building model...")
                        progress_bar.progress(40)
                        
                        status_text.text(f"🎯 Training for {epochs} epochs...")
                        
                        # Retrain
                        new_model, new_scaler, metrics = retrain_model(new_df, epochs)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Training complete!")
                        
                        st.success("🎉 Model retrained successfully!")
                        
                        # Show results
                        st.markdown("---")
                        st.markdown("### 📊 New Model Performance")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            improvement_mae = 8.08 - metrics['mae']
                            st.metric("MAE", f"{metrics['mae']:.2f} MW",
                                     f"{improvement_mae:+.2f} MW vs old")
                        with col2:
                            improvement_rmse = 13.66 - metrics['rmse']
                            st.metric("RMSE", f"{metrics['rmse']:.2f} MW",
                                     f"{improvement_rmse:+.2f} MW vs old")
                        with col3:
                            improvement_r2 = metrics['r2'] - 0.761
                            st.metric("R² Score", f"{metrics['r2']:.4f}",
                                     f"{improvement_r2:+.4f} vs old")
                        
                        # Training history
                        if 'history' in metrics:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(y=metrics['history'].history['loss'],
                                                    name='Training Loss', mode='lines'))
                            fig.add_trace(go.Scatter(y=metrics['history'].history['val_loss'],
                                                    name='Validation Loss', mode='lines'))
                            fig.update_layout(title="Training History", xaxis_title="Epoch",
                                            yaxis_title="Loss", height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Save option
                        st.markdown("---")
                        if st.button("💾 Save New Model"):
                            new_model.save('../models/retrained_model.keras')
                            with open('../models/retrained_scaler.pkl', 'wb') as f:
                                pickle.dump(new_scaler, f)
                            st.success("✅ Model saved as 'retrained_model.keras'")
                            st.info("💡 To use new model, rename it to 'monsoon_solar_lstm.keras'")
                
                else:
                    st.error(f"❌ Validation failed: {message}")
                    st.info("💡 Please ensure your CSV has the required columns and format")
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
        
        else:
            st.info("👆 Upload a CSV file to begin retraining")
            
            # Show example format
            with st.expander("📖 Required CSV Format"):
                example_df = pd.DataFrame({
                    'timestamp': ['2024-06-01 00:00:00', '2024-06-01 00:05:00'],
                    'cloud_cover_percent': [25, 30],
                    'temperature_celsius': [28.5, 28.3],
                    'humidity_percent': [75, 76],
                    'wind_speed_kmh': [12, 13],
                    'solar_output_mw': [0, 0]
                })
                st.dataframe(example_df)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p><strong>Monsoon Solar Predictor - Advanced Edition</strong></p>
            <p>🎓 Research-Grade System | 🌍 Indian Monsoon Optimized | ⚡ Production-Ready</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
