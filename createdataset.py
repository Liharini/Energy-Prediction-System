import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate timestamps for April 2025
time = pd.date_range(start='2025-04-01 00:00:00', end='2025-04-30 23:00:00', freq='H')

# Create energy values with realistic daily patterns
energy_values = []
for ts in time:
    hour = ts.hour
    
    # Base energy usage (in kWh)
    if 0 <= hour <= 5:       # Late night
        base = np.random.uniform(0.2, 0.5)
    elif 6 <= hour <= 9:     # Morning peak
        base = np.random.uniform(1.0, 2.5)
    elif 10 <= hour <= 17:   # Daytime (people out)
        base = np.random.uniform(0.7, 1.5)
    elif 18 <= hour <= 22:   # Evening peak
        base = np.random.uniform(1.5, 3.0)
    else:                   # 23:00
        base = np.random.uniform(0.5, 1.0)
    
    # Add some random daily variation
    variation = np.random.normal(0, 0.1)
    value = max(base + variation, 0)  # ensure no negative energy
    energy_values.append(value)

# Occasionally inject small anomalies (spikes)
anomaly_indices = np.random.choice(range(len(energy_values)), size=5, replace=False)
for idx in anomaly_indices:
    energy_values[idx] *= np.random.uniform(1.8, 2.5)  # spike

# Create the DataFrame
df_april = pd.DataFrame({
    'Time': time,
    'Energy_kWh': energy_values
})

# Save to CSV
df_april.to_csv('april_2025_energy.csv', index=False)

print("✅ Synthetic dataset for April 2025 created: april_2025_energy.csv")
