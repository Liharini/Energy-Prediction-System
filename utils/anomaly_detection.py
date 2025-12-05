import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def detect_anomalies_by_range(actual_rescaled, threshold_percent=0.85):
    """
    Detect anomalies based on a dynamic range-based threshold.

    :param actual_rescaled: np.array, actual energy values (rescaled)
    :param threshold_percent: float, proportion of the range to set the threshold (e.g., 0.85 = top 15%)
    :return: np.array of bools indicating anomaly positions, threshold value
    """
    values = actual_rescaled.flatten()
    data_range = np.max(values) - np.min(values)
    dynamic_threshold = np.min(values) + threshold_percent * data_range

    anomalies = values > dynamic_threshold
    return anomalies, dynamic_threshold


def save_range_based_anomalies(anomalies, data, seq_length, output_path="results/anomalies.csv"):
    """
    Save detected anomalies to a CSV file.

    :param anomalies: np.array of bools, anomaly labels
    :param data: DataFrame, original data with datetime index
    :param seq_length: int, LSTM sequence length offset
    :param output_path: str, path to save the CSV
    """
    anomaly_indices = np.where(anomalies)[0]
    if len(anomaly_indices) == 0:
        print("✅ No anomalies detected.")
        return

    anomalous_data = data.iloc[anomaly_indices + seq_length].copy() if seq_length > 0 else data.iloc[anomaly_indices].copy()
    anomalous_data['Anomaly'] = True
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    anomalous_data.to_csv(output_path)
    print(f"✅ Anomalous data saved to '{output_path}'")
    print("\nSample of anomalies detected:")
    print(anomalous_data.head())


def plot_range_based_anomalies(actual_rescaled, anomalies, threshold, output_path="results/anomaly_detection.png"):
    """
    Plot actual values with range-based anomalies highlighted.

    :param actual_rescaled: np.array, actual energy values
    :param anomalies: np.array of bools, anomaly labels
    :param threshold: float, threshold used to detect anomalies
    :param output_path: str, path to save the plot
    """
    values = actual_rescaled.flatten()
    anomaly_indices = np.where(anomalies)[0]

    plt.figure(figsize=(12, 5))
    plt.plot(values, label='Actual Energy', color='blue', alpha=0.7)
    plt.scatter(anomaly_indices, values[anomaly_indices], color='red', label='Anomalies', s=20)
    plt.axhline(y=threshold, color='green', linestyle='--', label=f'Threshold = {threshold:.2f} kWh')
    plt.xlabel("Time Step")
    plt.ylabel("Energy (kWh)")
    plt.title("Anomaly Detection Using Dynamic Range-Based Threshold")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    
def get_anomaly_timeframes(anomalies, dates, tolerance=3):
    """
    Group consecutive anomalies into timeframes.
    
    :param anomalies: np.array of bools, anomaly labels
    :param dates: array-like of datetime objects corresponding to each data point
    :param tolerance: int, maximum gap between anomalies to be considered part of the same timeframe
    :return: list of tuples with start and end dates of anomaly timeframes
    """
    if not any(anomalies):
        return []
        
    anomaly_indices = np.where(anomalies)[0]
    timeframes = []
    
    # Initialize with the first anomaly
    start_idx = anomaly_indices[0]
    prev_idx = start_idx
    
    for idx in anomaly_indices[1:]:
        # If there's a gap larger than tolerance, this is a new timeframe
        if idx - prev_idx > tolerance:
            timeframes.append((dates[start_idx], dates[prev_idx]))
            start_idx = idx
        prev_idx = idx
    
    # Add the last timeframe
    timeframes.append((dates[start_idx], dates[prev_idx]))
    
    return timeframes


def calculate_energy_metrics(energy_data):
    """
    Calculate various energy usage metrics from the data.
    
    :param energy_data: pandas Series or numpy array of energy values
    :return: dict of metrics
    """
    metrics = {
        'mean': np.mean(energy_data),
        'max': np.max(energy_data),
        'min': np.min(energy_data),
        'std': np.std(energy_data),
        'total': np.sum(energy_data),
        'peak_hour': np.argmax(np.bincount([hour for hour in pd.DatetimeIndex(energy_data.index).hour])) if hasattr(energy_data, 'index') else None
    }
    
    return metrics