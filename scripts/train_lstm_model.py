"""
Monsoon Solar Predictor - LSTM Model Training
Predicts solar output 30 minutes ahead using historical patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)


def load_data(filepath):
    """Load the generated monsoon solar data"""
    print("\n" + "="*60)
    print("📂 LOADING DATA")
    print("="*60)
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ Loaded {len(df):,} records")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Features: {list(df.columns)}")
    
    return df


def prepare_features(df):
    """
    Select and normalize features for model training
    
    Features used:
    - cloud_cover_percent: Most important for solar prediction
    - temperature_celsius: Affects panel efficiency
    - humidity_percent: Atmospheric conditions
    - wind_speed_kmh: Can affect panel temperature
    - hour: Time of day pattern
    - solar_output_mw: Current output (to predict future)
    """
    
    print("\n" + "="*60)
    print("🔧 PREPARING FEATURES")
    print("="*60)
    
    # Select relevant features
    feature_columns = [
        'cloud_cover_percent',
        'temperature_celsius', 
        'humidity_percent',
        'wind_speed_kmh',
        'hour',
        'solar_output_mw'  # Target variable (also input feature)
    ]
    
    data = df[feature_columns].values
    
    print(f"✅ Selected {len(feature_columns)} features:")
    for i, col in enumerate(feature_columns):
        print(f"   {i+1}. {col}")
    
    # Normalize to 0-1 range (LSTM works better with normalized data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)
    
    # Save scaler for later use (when making predictions)
    os.makedirs('../models', exist_ok=True)
    with open('../models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✅ Data normalized to range [0, 1]")
    print(f"✅ Scaler saved to: ../models/scaler.pkl")
    
    return data_normalized, scaler, feature_columns


def create_sequences(data, lookback=12, forecast_horizon=6):
    """
    Create input sequences for LSTM
    
    Args:
        data: Normalized feature array
        lookback: How many past timesteps to use (12 = 1 hour)
        forecast_horizon: How far ahead to predict (6 = 30 minutes)
    
    Example:
        Input: Last 12 readings (1 hour of data)
        Output: Solar power 6 steps ahead (30 minutes)
    
    Returns:
        X: Input sequences (samples, timesteps, features)
        y: Target values (samples,)
    """
    
    print("\n" + "="*60)
    print("📊 CREATING SEQUENCES")
    print("="*60)
    
    print(f"   Lookback period: {lookback} timesteps ({lookback * 5} minutes)")
    print(f"   Forecast horizon: {forecast_horizon} timesteps ({forecast_horizon * 5} minutes)")
    
    X, y = [], []
    
    for i in range(len(data) - lookback - forecast_horizon):
        # Input: Past 'lookback' timesteps (all features)
        X.append(data[i:(i + lookback)])
        
        # Target: Solar output at 'forecast_horizon' ahead
        # Index -1 is solar_output_mw (last column)
        y.append(data[i + lookback + forecast_horizon - 1, -1])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Created {len(X):,} sequences")
    print(f"   Input shape: {X.shape} (samples, timesteps, features)")
    print(f"   Output shape: {y.shape} (samples,)")
    
    return X, y


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Split data into train, validation, and test sets
    
    Train: 70% - Model learns from this
    Validation: 15% - Tune model during training
    Test: 15% - Final evaluation (model never sees this)
    """
    
    print("\n" + "="*60)
    print("✂️  SPLITTING DATA")
    print("="*60)
    
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    print(f"   Total samples: {n:,}")
    print(f"   Training: {len(X_train):,} ({train_ratio*100:.0f}%)")
    print(f"   Validation: {len(X_val):,} ({val_ratio*100:.0f}%)")
    print(f"   Test: {len(X_test):,} ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_lstm_model(input_shape):
    """
    Build LSTM neural network
    
    Architecture:
    - LSTM Layer 1: 64 units (learns temporal patterns)
    - Dropout: 20% (prevents overfitting)
    - LSTM Layer 2: 32 units (refines patterns)
    - Dropout: 20%
    - Dense Layer: 16 units (combines features)
    - Output: 1 unit (predicted solar output)
    """
    
    print("\n" + "="*60)
    print("🧠 BUILDING LSTM MODEL")
    print("="*60)
    
    model = Sequential([
        # First LSTM layer
        LSTM(64, return_sequences=True, input_shape=input_shape, name='lstm_1'),
        Dropout(0.2, name='dropout_1'),
        
        # Second LSTM layer
        LSTM(32, return_sequences=False, name='lstm_2'),
        Dropout(0.2, name='dropout_2'),
        
        # Dense layers
        Dense(16, activation='relu', name='dense_1'),
        Dropout(0.1, name='dropout_3'),
        
        # Output layer
        Dense(1, name='output')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # Mean Squared Error
        metrics=['mae']  # Mean Absolute Error
    )
    
    print("\n📋 Model Summary:")
    model.summary()
    
    total_params = model.count_params()
    print(f"\n✅ Model built successfully!")
    print(f"   Total parameters: {total_params:,}")
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train the LSTM model
    """
    
    print("\n" + "="*60)
    print("🎯 TRAINING MODEL")
    print("="*60)
    
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Training samples: {len(X_train):,}")
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        '../models/best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
    
    # Train
    print("\n🚀 Starting training...\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    print("\n✅ Training complete!")
    
    return history


def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate model performance on test set
    """
    
    print("\n" + "="*60)
    print("📊 EVALUATING MODEL")
    print("="*60)
    
    # Make predictions
    y_pred_normalized = model.predict(X_test, verbose=0)
    
    # Denormalize predictions and actual values
    # Create dummy array with same shape as original features
    dummy = np.zeros((len(y_pred_normalized), scaler.n_features_in_))
    
    # Put predictions in last column (solar_output_mw position)
    dummy[:, -1] = y_pred_normalized.flatten()
    y_pred = scaler.inverse_transform(dummy)[:, -1]
    
    # Do the same for actual values
    dummy[:, -1] = y_test
    y_actual = scaler.inverse_transform(dummy)[:, -1]
    
    # Calculate metrics
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = y_actual > 1  # Only calculate for values > 1 MW
    mape = np.mean(np.abs((y_actual[mask] - y_pred[mask]) / y_actual[mask])) * 100
    
    print(f"\n📈 Performance Metrics:")
    print(f"   MAE (Mean Absolute Error): {mae:.2f} MW")
    print(f"   RMSE (Root Mean Squared Error): {rmse:.2f} MW")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    # Accuracy interpretation
    print(f"\n💡 Interpretation:")
    if mae < 5:
        print(f"   ✅ Excellent! Predictions are typically within {mae:.1f} MW")
    elif mae < 10:
        print(f"   ✅ Good! Predictions are typically within {mae:.1f} MW")
    elif mae < 15:
        print(f"   ⚠️  Moderate. Predictions are typically within {mae:.1f} MW")
    else:
        print(f"   ❌ Needs improvement. Predictions are off by {mae:.1f} MW on average")
    
    return y_actual, y_pred, mae, rmse, r2, mape


def plot_training_history(history):
    """Plot training and validation loss"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Model Loss During Training', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE plot
    ax2.plot(history.history['mae'], label='Training MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('Mean Absolute Error During Training', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../visualizations/training_history.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 Training history saved: ../visualizations/training_history.png")


def plot_predictions(y_actual, y_pred, num_samples=500):
    """Plot actual vs predicted values"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Time series comparison (first 500 samples)
    samples = min(num_samples, len(y_actual))
    x = range(samples)
    
    ax1.plot(x, y_actual[:samples], label='Actual', linewidth=2, alpha=0.7)
    ax1.plot(x, y_pred[:samples], label='Predicted', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Sample', fontsize=12)
    ax1.set_ylabel('Solar Output (MW)', fontsize=12)
    ax1.set_title('Actual vs Predicted Solar Output (First 500 samples)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot (predicted vs actual)
    ax2.scatter(y_actual, y_pred, alpha=0.5, s=10)
    
    # Perfect prediction line
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Solar Output (MW)', fontsize=12)
    ax2.set_ylabel('Predicted Solar Output (MW)', fontsize=12)
    ax2.set_title('Prediction Accuracy Scatter Plot', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../visualizations/predictions_comparison.png', dpi=150, bbox_inches='tight')
    print(f"📈 Predictions comparison saved: ../visualizations/predictions_comparison.png")


if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("   MONSOON SOLAR PREDICTOR - LSTM TRAINING")
    print("="*60)
    
    # 1. Load data
    df = load_data('../data/monsoon_solar_data.csv')
    
    # 2. Prepare features
    data_normalized, scaler, feature_columns = prepare_features(df)
    
    # 3. Create sequences
    X, y = create_sequences(
        data_normalized, 
        lookback=12,          # 1 hour of history (12 × 5 min)
        forecast_horizon=6    # Predict 30 min ahead (6 × 5 min)
    )
    
    # 4. Split data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    
    # 5. Build model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # 6. Train model
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    # 7. Evaluate model
    y_actual, y_pred, mae, rmse, r2, mape = evaluate_model(model, X_test, y_test, scaler)
    
    # 8. Save final model
    model.save('../models/monsoon_solar_lstm.keras')
    print(f"\n💾 Final model saved: ../models/monsoon_solar_lstm.keras")
    
    # 9. Create visualizations
    plot_training_history(history)
    plot_predictions(y_actual, y_pred)
    
    # 10. Summary
    print("\n" + "="*60)
    print("✅ PHASE 2 COMPLETE - MODEL TRAINING SUCCESSFUL!")
    print("="*60)
    print(f"\n📦 Generated Files:")
    print(f"   1. ../models/monsoon_solar_lstm.keras (trained model)")
    print(f"   2. ../models/best_model.keras (best checkpoint)")
    print(f"   3. ../models/scaler.pkl (data normalizer)")
    print(f"   4. ../visualizations/training_history.png")
    print(f"   5. ../visualizations/predictions_comparison.png")
    
    print(f"\n📊 Final Performance:")
    print(f"   MAE: {mae:.2f} MW")
    print(f"   RMSE: {rmse:.2f} MW")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    
    print("\n🎯 Next: Phase 3 - Build Dashboard!")
    print("="*60 + "\n")
