"""
Monsoon Solar Predictor - Data Generation Script
Simulates 90 days of solar generation data with monsoon patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def simulate_solar_output(cloud_cover, hour, base_capacity=100, season='monsoon'):
    """
    Simulates solar power output based on cloud cover and time of day
    
    Args:
        cloud_cover: 0-100% cloud coverage
        hour: 0-23 hour of day
        base_capacity: Maximum plant capacity in MW
        season: 'monsoon' or 'normal'
    
    Returns:
        Solar output in MW
    """
    
    # Solar intensity based on time (0 at night, max at noon)
    if 6 <= hour <= 18:
        # Sine curve peaking at noon
        time_factor = np.sin(np.pi * (hour - 6) / 12)
    else:
        time_factor = 0
    
    # Cloud impact on solar output
    # Clear sky (0% clouds) = 100% output
    # Overcast (100% clouds) = 10-20% output (some diffuse radiation)
    cloud_factor = 1 - (0.85 * cloud_cover / 100)
    
    # Monsoon has more atmospheric moisture even without clouds
    if season == 'monsoon':
        atmospheric_attenuation = 0.92  # 8% loss due to humidity
    else:
        atmospheric_attenuation = 0.98
    
    # Add realistic random variations (±5%)
    noise = np.random.normal(1, 0.05)
    
    # Calculate final output
    output = base_capacity * time_factor * cloud_factor * atmospheric_attenuation * noise
    
    return max(0, min(output, base_capacity))  # Clamp between 0 and capacity


def generate_cloud_patterns(num_points, monsoon=True):
    """
    Generates realistic cloud cover patterns
    
    Args:
        num_points: Number of data points to generate
        monsoon: If True, generates monsoon-like cloud patterns
    
    Returns:
        Array of cloud cover percentages
    """
    clouds = []
    current_cloud = np.random.randint(10, 40)
    
    for i in range(num_points):
        
        if monsoon:
            # Monsoon: sudden cloud bursts, rapid changes
            if np.random.random() < 0.08:  # 8% chance of sudden clouds
                current_cloud = np.random.randint(70, 100)  # Heavy clouds
            elif np.random.random() < 0.15:  # 15% chance of clearing
                current_cloud = np.random.randint(0, 30)   # Clear sky
            else:
                # Gradual change with some randomness
                change = np.random.randint(-15, 15)
                current_cloud = np.clip(current_cloud + change, 0, 100)
        else:
            # Normal season: slower, more predictable changes
            change = np.random.randint(-5, 5)
            current_cloud = np.clip(current_cloud + change, 0, 60)
        
        clouds.append(current_cloud)
    
    return np.array(clouds)


def create_monsoon_dataset(days=90, interval_minutes=5, base_capacity=100):
    """
    Creates complete monsoon solar generation dataset
    
    Args:
        days: Number of days to simulate
        interval_minutes: Time interval between readings
        base_capacity: Solar plant capacity in MW
    
    Returns:
        pandas DataFrame with all features
    """
    
    print(f"🌦️ Generating {days} days of monsoon solar data...")
    print(f"   Interval: {interval_minutes} minutes")
    print(f"   Plant capacity: {base_capacity} MW")
    
    # Calculate number of data points
    points_per_day = (24 * 60) // interval_minutes
    num_points = days * points_per_day
    
    print(f"   Total data points: {num_points:,}")
    
    # Start date: June 1, 2024 (monsoon season)
    start_date = datetime(2024, 6, 1, 0, 0)
    
    # Generate timestamps
    timestamps = [start_date + timedelta(minutes=interval_minutes*i) for i in range(num_points)]
    
    # Generate cloud patterns
    cloud_cover = generate_cloud_patterns(num_points, monsoon=True)
    
    # Generate weather features
    # Temperature: Higher during day, varies 25-38°C in monsoon
    base_temp = 30
    temp_variation = [base_temp + 6*np.sin(2*np.pi*t.hour/24) + np.random.normal(0, 2) 
                      for t in timestamps]
    temperature = np.clip(temp_variation, 25, 38)
    
    # Humidity: High during monsoon (60-95%)
    humidity = np.random.randint(60, 95, num_points)
    
    # Wind speed: 5-25 km/h, higher during cloud bursts
    wind_speed = []
    for i, cloud in enumerate(cloud_cover):
        if cloud > 70:  # Storm conditions
            wind = np.random.randint(15, 30)
        else:
            wind = np.random.randint(5, 15)
        wind_speed.append(wind)
    
    # Rainfall: 0 most of time, heavy when clouds >80%
    rainfall = []
    for cloud in cloud_cover:
        if cloud > 80 and np.random.random() < 0.6:
            rainfall.append(np.random.randint(5, 50))  # mm/hour
        else:
            rainfall.append(0)
    
    # Generate solar output based on all factors
    solar_output = []
    for i, ts in enumerate(timestamps):
        output = simulate_solar_output(
            cloud_cover[i], 
            ts.hour, 
            base_capacity,
            season='monsoon'
        )
        solar_output.append(output)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'hour': [t.hour for t in timestamps],
        'day_of_year': [t.timetuple().tm_yday for t in timestamps],
        'cloud_cover_percent': cloud_cover,
        'temperature_celsius': temperature,
        'humidity_percent': humidity,
        'wind_speed_kmh': wind_speed,
        'rainfall_mmh': rainfall,
        'solar_output_mw': solar_output
    })
    
    # Add derived features
    df['is_daytime'] = df['hour'].apply(lambda h: 1 if 6 <= h <= 18 else 0)
    
    print("✅ Dataset generation complete!")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Average solar output: {df['solar_output_mw'].mean():.2f} MW")
    print(f"   Peak solar output: {df['solar_output_mw'].max():.2f} MW")
    print(f"   Average cloud cover: {df['cloud_cover_percent'].mean():.1f}%")
    print(f"   Rainy periods: {(df['rainfall_mmh'] > 0).sum()} out of {len(df)}")
    
    return df


def visualize_sample_data(df, days_to_show=7):
    """
    Creates visualization of generated data
    """
    
    # Select first week
    sample = df.head(days_to_show * 288)  # 288 = 5-min intervals per day
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot 1: Solar Output
    axes[0].plot(sample['timestamp'], sample['solar_output_mw'], color='orange', linewidth=1)
    axes[0].set_ylabel('Solar Output (MW)', fontsize=12)
    axes[0].set_title('Simulated Monsoon Solar Generation - First Week', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(sample['timestamp'], sample['solar_output_mw'], alpha=0.3, color='orange')
    
    # Plot 2: Cloud Cover
    axes[1].plot(sample['timestamp'], sample['cloud_cover_percent'], color='gray', linewidth=1)
    axes[1].set_ylabel('Cloud Cover (%)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(sample['timestamp'], sample['cloud_cover_percent'], alpha=0.3, color='gray')
    
    # Plot 3: Temperature and Rainfall
    ax3 = axes[2]
    ax3.plot(sample['timestamp'], sample['temperature_celsius'], color='red', linewidth=1, label='Temperature')
    ax3.set_ylabel('Temperature (°C)', fontsize=12, color='red')
    ax3.tick_params(axis='y', labelcolor='red')
    ax3.grid(True, alpha=0.3)
    
    ax3_rain = ax3.twinx()
    ax3_rain.bar(sample['timestamp'], sample['rainfall_mmh'], color='blue', alpha=0.3, width=0.003, label='Rainfall')
    ax3_rain.set_ylabel('Rainfall (mm/h)', fontsize=12, color='blue')
    ax3_rain.tick_params(axis='y', labelcolor='blue')
    
    axes[2].set_xlabel('Date/Time', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../visualizations/monsoon_data_sample.png', dpi=150, bbox_inches='tight')
    print("\n📈 Visualization saved: ../visualizations/monsoon_data_sample.png")
    
    return fig


if __name__ == "__main__":
    
    print("=" * 60)
    print("   MONSOON SOLAR PREDICTOR - DATA GENERATION")
    print("=" * 60)
    
    # Generate dataset
    df = create_monsoon_dataset(
        days=90,              # 3 months of monsoon data
        interval_minutes=5,   # One reading every 5 minutes
        base_capacity=100     # 100 MW solar plant
    )
    
    # Save to CSV
    output_file = '../data/monsoon_solar_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Data saved to: {output_file}")
    print(f"   File size: {len(df):,} rows × {len(df.columns)} columns")
    
    # Create visualization
    visualize_sample_data(df, days_to_show=7)
    
    # Print sample data
    print("\n📋 Sample data (first 10 rows):")
    print(df.head(10).to_string())
    
    print("\n" + "=" * 60)
    print("✅ WEEK 1 COMPLETE - Data ready for model training!")
    print("=" * 60)