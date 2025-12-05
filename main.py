import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from models.lstm_model import EnergyLSTM
from utils.anomaly_detection import (
    detect_anomalies_by_range,
    save_range_based_anomalies,
    plot_range_based_anomalies,
    calculate_energy_metrics
)
import os
import warnings
warnings.filterwarnings("ignore")

def train_model(data_path, output_dir='models', save_model=True):
    """
    Train an energy forecasting model based on the data.
    
    :param data_path: path to the CSV data file
    :param output_dir: directory to save the model
    :param save_model: whether to save the trained model
    :return: trained model, scaler, evaluation metrics
    """
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_length = 24
    batch_size = 64
    num_epochs = 50
    lr = 0.001
    
    # Load dataset
    data = pd.read_csv(data_path)
    if 'Timestamp' not in data.columns:
        data['Timestamp'] = pd.date_range(start='2022-01-01', periods=len(data), freq='H')
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data.set_index('Timestamp', inplace=True)
    
    # Scale energy values
    energy_values = data['Energy_kWh'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    energy_scaled = scaler.fit_transform(energy_values)
    
    # Create sequences
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            xs.append(data[i:i+seq_length])
            ys.append(data[i+seq_length])
        return np.array(xs), np.array(ys)
    
    X, y = create_sequences(energy_scaled, seq_length)
    
    # Split train/test
    split_ratio = 0.7
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor  = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor  = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model setup
    model = EnergyLSTM(input_size=1, hidden_size=128, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()
        actual = y_test_tensor.cpu().numpy()
    
    # Inverse transform
    predictions_rescaled = scaler.inverse_transform(predictions)
    actual_rescaled = scaler.inverse_transform(actual)
    
    # Metrics
    mae = mean_absolute_error(actual_rescaled, predictions_rescaled)
    rmse = np.sqrt(mean_squared_error(actual_rescaled, predictions_rescaled))
    r2 = r2_score(actual_rescaled, predictions_rescaled)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    # Save model
    if save_model:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'lstm_energy_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    return model, scaler, metrics


def predict_next_month(model, scaler, recent_data, seq_length=24):
    """
    Predict the next month's energy consumption.
    
    :param model: trained PyTorch model
    :param scaler: fitted scaler
    :param recent_data: recent energy data (at least seq_length points)
    :param seq_length: sequence length used for the model
    :return: predicted next month's total energy consumption
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Scale the input data
    scaled_data = scaler.transform(recent_data.reshape(-1, 1))
    
    # Create input sequence
    input_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(device)
    
    # Make predictions for next 30 days (720 hours)
    next_month_predictions = []
    current_input = input_tensor.clone()
    
    for _ in range(720):  # 30 days * 24 hours
        with torch.no_grad():
            next_step = model(current_input).cpu().numpy()
            next_month_predictions.append(next_step[0, 0])
            
            # Update input sequence for next prediction
            current_input = torch.cat([
                current_input[:, 1:, :],
                torch.tensor(next_step.reshape(1, 1, 1), dtype=torch.float32).to(device)
            ], dim=1)
    
    # Inverse transform predictions
    next_month_predictions = np.array(next_month_predictions).reshape(-1, 1)
    next_month_rescaled = scaler.inverse_transform(next_month_predictions)
    
    # Sum up to get total monthly consumption
    next_month_total = np.sum(next_month_rescaled)
    
    return next_month_total


def analyze_energy_data(data_path, output_dir='results'):
    """
    Analyze energy data file and return predictions and anomalies.
    
    :param data_path: path to energy data CSV
    :param output_dir: directory to save results
    :return: dict with analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    data = pd.read_csv(data_path)
    if 'Timestamp' not in data.columns:
        data['Timestamp'] = pd.date_range(start='2022-01-01', periods=len(data), freq='H')
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data.set_index('Timestamp', inplace=True)
    
    if 'Energy_kWh' not in data.columns:
        # Try to use the first numeric column as Energy_kWh
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            data['Energy_kWh'] = data[numeric_cols[0]]
        else:
            raise ValueError("No numeric column found for energy data")
    
    # Scale energy values
    energy_values = data['Energy_kWh'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    energy_scaled = scaler.fit_transform(energy_values)
    
    # Train model or load pre-trained model
    try:
        model_path = os.path.join('models', 'lstm_energy_model.pth')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EnergyLSTM(input_size=1, hidden_size=128, num_layers=2).to(device)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded pre-trained model from {model_path}")
        else:
            print("No pre-trained model found. Training a new model...")
            model, scaler, metrics = train_model(data_path, save_model=True)
    except Exception as e:
        print(f"Error loading or training model: {str(e)}")
        # Fallback to a simple prediction method
        last_month_total = np.sum(energy_values[-720:])  # Last 30 days
        next_month_forecast = last_month_total * 1.05  # Simple 5% increase
    else:
        # Predict next month's energy consumption
        next_month_forecast = predict_next_month(model, scaler, energy_values)
    
    # Detect anomalies
    threshold_percent = 0.70
    anomalies, threshold = detect_anomalies_by_range(energy_values, threshold_percent=threshold_percent)
    anomaly_indices = np.where(anomalies)[0]
    anomaly_count = len(anomaly_indices)
    
    # Get anomaly dates
    anomaly_dates = []
    if anomaly_count > 0:
        for idx in anomaly_indices:
            if idx < len(data):
                date_str = data.index[idx].strftime('%Y-%m-%d %H:%M')
                anomaly_dates.append(date_str)
    
    # Create visualizations
    # 1. Energy usage plot
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, energy_values, label='Energy Usage')
    plt.title('Energy Usage Over Time')
    plt.xlabel('Date')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.tight_layout()
    usage_plot_path = os.path.join(output_dir, 'energy_usage.png')
    plt.savefig(usage_plot_path)
    plt.close()
    
    # 2. Anomaly detection plot
    plot_range_based_anomalies(
        energy_values, 
        anomalies, 
        threshold, 
        output_path=os.path.join(output_dir, 'anomaly_detection.png')
    )
    
    # Calculate energy metrics
    metrics = calculate_energy_metrics(data['Energy_kWh'])
    
    # Generate energy saving suggestions based on patterns
    suggestions = generate_suggestions(metrics, anomaly_count)
    
    # Compile results
    results = {
        'next_month_forecast': float(next_month_forecast),
        'anomaly_count': anomaly_count,
        'anomaly_dates': anomaly_dates,
        'metrics': metrics,
        'suggestions': suggestions,
        'plots': {
            'usage': usage_plot_path,
            'anomalies': os.path.join(output_dir, 'anomaly_detection.png')
        }
    }
    
    return results


def generate_suggestions(metrics, anomaly_count):
    """
    Generate energy saving suggestions based on energy usage patterns.
    
    :param metrics: dict of energy metrics
    :param anomaly_count: number of anomalies detected
    :return: list of suggestions
    """
    suggestions = []
    
    # Basic suggestions
    suggestions.append("Replace traditional light bulbs with LED bulbs to save up to 75% energy")
    suggestions.append("Use smart power strips to eliminate phantom energy usage from devices on standby")
    
    # Based on peak hour
    if metrics.get('peak_hour') is not None:
        peak_hour = metrics['peak_hour']
        if 7 <= peak_hour <= 10 or 17 <= peak_hour <= 21:
            suggestions.append(f"Your peak energy usage is at {peak_hour}:00. Consider programming appliances to run during off-peak hours")
    
    # Based on anomaly count
    if anomaly_count > 5:
        suggestions.append("You have several unusual energy spikes. Consider an energy audit to identify potential issues")
    elif anomaly_count > 0:
        suggestions.append(f"We detected {anomaly_count} energy usage anomalies. Check for devices that might be malfunctioning")
    
    # Based on overall consumption
    if metrics['mean'] > 2.0:  # Arbitrary threshold, adjust as needed
        suggestions.append("Your average energy consumption is high. Consider upgrading to ENERGY STAR certified appliances")
    
    # Seasonal suggestions
    current_month = pd.Timestamp.now().month
    if 5 <= current_month <= 9:  # Summer months
        suggestions.append("Set your thermostat to 78°F (26°C) when you're home and higher when you're away to reduce cooling costs")
    elif current_month in [12, 1, 2, 3]:  # Winter months
        suggestions.append("Set your thermostat to 68°F (20°C) when you're home and lower when you're away or sleeping")
    
    # Additional general tips
    suggestions.append("Ensure proper insulation around windows and doors to prevent energy waste")
    suggestions.append("Clean or replace HVAC filters monthly for optimal efficiency")
    
    return suggestions


if __name__ == "__main__":
    # Example usage
    data_path = 'data/hue.csv'
    results = analyze_energy_data(data_path)
    
    print(f"\n✅ Next Month's Forecasted Energy: {results['next_month_forecast']:.2f} kWh")
    print(f"✅ Anomalies Detected: {results['anomaly_count']}")
    
    print("\n✅ Energy Saving Suggestions:")
    for suggestion in results['suggestions']:
        print(f"  • {suggestion}")