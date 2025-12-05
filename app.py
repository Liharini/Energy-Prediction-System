from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Import your model and utility functions
from models.lstm_model import EnergyLSTM
from utils.anomaly_detection import detect_anomalies_by_range, plot_range_based_anomalies

# Constants and configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'csv'}
SEQ_LENGTH = 24

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

# Prediction History model
class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    month = db.Column(db.String(20), nullable=False)
    forecast_value = db.Column(db.Float, nullable=False)
    anomaly_count = db.Column(db.Integer, nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    result_path = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f'<Prediction {self.month} for user {self.user_id}>'

# Create tables
with app.app_context():
    db.create_all()

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_energy_saving_tips(usage_pattern, anomaly_count):
    # Basic suggestions based on data patterns and anomalies
    tips = []

    # General tips
    tips.append("Install LED bulbs which use up to 75% less energy than incandescent lighting")
    tips.append("Use smart power strips to reduce standby power consumption")
    tips.append("Unplug chargers and electronics when not in use")
    tips.append("Make the most of natural daylight and turn off lights during the day")
    tips.append("Set your refrigerator temperature between 35°F and 38°F for optimal efficiency")

    # Based on anomaly count
    if anomaly_count > 3:
        tips.append("Consider a smart thermostat to optimize heating and cooling")
        tips.append("Check for devices that might be malfunctioning and causing energy spikes")
        tips.append("Use energy monitoring devices to track real-time consumption and detect unusual spikes")
        tips.append("Inspect insulation and seal leaks to prevent loss of heating/cooling energy")

    # Based on average usage
    if usage_pattern > 50:  # Arbitrary threshold
        tips.append("Your energy usage is high. Consider an energy audit")
        tips.append("Upgrade to energy-efficient appliances")
        tips.append("Limit usage of high-power devices during peak hours")
        tips.append("Use ceiling fans to reduce the load on air conditioners")
        tips.append("Wash clothes with cold water and air dry when possible")
    else:
        tips.append("Your energy consumption is relatively low. Keep up the good practices!")
        tips.append("Continue monitoring your usage to catch future anomalies early")
        tips.append("Consider installing solar panels to further reduce dependency on grid electricity")

    return tips


# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists!')
            return redirect(url_for('register'))
        
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash('Email already registered!')
            return redirect(url_for('register'))
        
        # Create new user
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Error: {str(e)}')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash(f'Welcome back, {username}!')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please login first!')
        return redirect(url_for('login'))
    
    # Fetch user's prediction history
    history = PredictionHistory.query.filter_by(user_id=session['user_id']).order_by(PredictionHistory.prediction_date.desc()).all()
    
    return render_template('dashboard.html', username=session['username'], history=history)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'user_id' not in session:
        flash('Please login first!')
        return redirect(url_for('login'))
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('dashboard'))
    
    file = request.files['file']
    month = request.form.get('month')
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('dashboard'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session['user_id']}_{month}_{filename}")
        file.save(file_path)
        
        # Process the file and get predictions
        result_path, forecast_value, anomaly_count, anomalies_dates, prediction_accuracy = process_file(file_path, month)
        
        # Extract just the filename part for database storage
        result_filename = os.path.basename(result_path)
        
        # Save prediction to history - store only the filename
        new_prediction = PredictionHistory(
            user_id=session['user_id'],
            month=month,
            forecast_value=float(forecast_value),
            anomaly_count=anomaly_count,
            file_path=file_path,
            result_path=result_filename  # Store only the filename
        )
        
        try:
            db.session.add(new_prediction)
            db.session.commit()
        except Exception as e:
            flash(f'Error saving prediction: {str(e)}')
        
        # Generate energy saving tips
        usage_pattern = forecast_value  # You can adjust this based on your metrics
        tips = get_energy_saving_tips(usage_pattern, anomaly_count)
        
        return render_template(
            'result.html',
            forecast_value=forecast_value,
            anomaly_count=anomaly_count,
            anomalies_dates=anomalies_dates,
            tips=tips,
            month=month,
            result_file=result_filename,
            prediction_accuracy=prediction_accuracy
        )
    
    flash('Invalid file type. Please upload a CSV file.')
    return redirect(url_for('dashboard'))

@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
def delete_prediction(prediction_id):
    if 'user_id' not in session:
        flash('Please login first!')
        return redirect(url_for('login'))
    
    # Get the prediction from the database
    prediction = PredictionHistory.query.get_or_404(prediction_id)
    
    # Ensure the user can only delete their own predictions
    if prediction.user_id != session['user_id']:
        flash('You do not have permission to delete this prediction.')
        return redirect(url_for('dashboard'))
    
    try:
        # Delete the associated files if they exist
        if prediction.file_path and os.path.exists(prediction.file_path):
            os.remove(prediction.file_path)
        
        # Build the full result path
        result_full_path = os.path.join(app.config['RESULTS_FOLDER'], prediction.result_path)
        if os.path.exists(result_full_path):
            os.remove(result_full_path)
        
        # Delete the database record
        db.session.delete(prediction)
        db.session.commit()
        
        flash('Prediction deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting prediction: {str(e)}', 'danger')
    
    return redirect(url_for('dashboard'))

@app.route('/delete_all_predictions', methods=['POST'])
def delete_all_predictions():
    if 'user_id' not in session:
        flash('Please login first!')
        return redirect(url_for('login'))
    
    try:
        # Get all predictions for the current user
        predictions = PredictionHistory.query.filter_by(user_id=session['user_id']).all()
        
        # Delete associated files for each prediction
        for prediction in predictions:
            if prediction.file_path and os.path.exists(prediction.file_path):
                os.remove(prediction.file_path)
            
            # Build the full result path
            result_full_path = os.path.join(app.config['RESULTS_FOLDER'], prediction.result_path)
            if os.path.exists(result_full_path):
                os.remove(result_full_path)
        
        # Delete all records for this user
        PredictionHistory.query.filter_by(user_id=session['user_id']).delete()
        db.session.commit()
        
        flash('All prediction history deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting predictions: {str(e)}', 'danger')
    
    return redirect(url_for('dashboard'))

# Diagnostic route to help debug path issues
@app.route('/debug/paths')
def debug_paths():
    if 'user_id' not in session:
        return "Please log in first"
    
    # Get all predictions for this user
    predictions = PredictionHistory.query.filter_by(user_id=session['user_id']).all()
    
    debug_info = {
        "RESULTS_FOLDER": app.config['RESULTS_FOLDER'],
        "predictions": []
    }
    
    for pred in predictions:
        # Check if the file exists in various locations
        filename = os.path.basename(pred.result_path)
        potential_paths = [
            pred.result_path,  # As stored in DB
            os.path.join(app.config['RESULTS_FOLDER'], filename),  # Using just filename
            os.path.join("static/results", filename),  # Direct path
            os.path.join("static", "results", filename)  # Alternative path format
        ]
        
        exists_at = []
        for path in potential_paths:
            if os.path.exists(path):
                exists_at.append(path)
        
        debug_info["predictions"].append({
            "id": pred.id,
            "month": pred.month,
            "stored_path": pred.result_path,
            "filename": filename,
            "exists_at": exists_at
        })
    
    return jsonify(debug_info)

def process_file(file_path, month):
    try:
        # Setup output directories
        os.makedirs("models", exist_ok=True)
        os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
        
        # Generate unique result names
        user_id = os.path.basename(file_path).split('_')[0]
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        result_filename = f"{user_id}_{month}_{timestamp}.png"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        # Load the data
        data = pd.read_csv(file_path)
        
        # Ensure the data has the required format
        if 'Energy_kWh' not in data.columns:
            # Try to use the first numeric column as Energy_kWh
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data['Energy_kWh'] = data[numeric_cols[0]]
            else:
                raise ValueError("No numeric column found for energy data")
        
        # Add a timestamp column if not present
        if 'Timestamp' not in data.columns:
            data['Timestamp'] = pd.date_range(start='2022-01-01', periods=len(data), freq='H')
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data.set_index('Timestamp', inplace=True)
        
        # Scale energy values
        energy_values = data['Energy_kWh'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        energy_scaled = scaler.fit_transform(energy_values)
        
        # Load or train the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EnergyLSTM(input_size=1, hidden_size=128, num_layers=2).to(device)
        model_path = os.path.join('models', 'lstm_energy_model.pth')
        
        # For prediction accuracy comparison
        prediction_accuracy = {}
        current_month_prediction = 0
        current_month_actual = np.sum(energy_values)
        
        if os.path.exists(model_path):
            try:
                # Try loading the pre-trained model
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Successfully loaded model from {model_path}")
                
                # Calculate accuracy for current month (hindcast)
                # We'll use first half of the data to predict second half
                seq_length = 24
                if len(energy_values) >= seq_length * 2:
                    # Use first half to predict second half
                    half_point = len(energy_values) // 2
                    train_data = energy_scaled[:half_point]
                    test_data = energy_values[half_point:]
                    
                    # Prepare input sequence for hindcast
                    input_seq = train_data[-seq_length:].reshape(1, seq_length, 1)
                    input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(device)
                    
                    # Generate predictions for second half
                    hindcast_predictions = []
                    current_input = input_tensor.clone()
                    
                    for _ in range(len(test_data)):
                        with torch.no_grad():
                            next_step = model(current_input).cpu().numpy()
                            hindcast_predictions.append(next_step[0, 0])
                            
                            # Update input sequence
                            current_input = torch.cat([
                                current_input[:, 1:, :],
                                torch.tensor(next_step.reshape(1, 1, 1), dtype=torch.float32).to(device)
                            ], dim=1)
                    
                    # Convert predictions back to original scale
                    hindcast_predictions = np.array(hindcast_predictions).reshape(-1, 1)
                    hindcast_rescaled = scaler.inverse_transform(hindcast_predictions)
                    
                    # Calculate metrics
                    current_month_prediction = np.sum(hindcast_rescaled)
                    current_month_actual = np.sum(test_data)
                    
                    # Calculate accuracy percentage (how close our prediction is to actual)
                    if current_month_actual > 0:
                        accuracy_percent = 100 - min(100, abs((current_month_prediction - current_month_actual) / current_month_actual * 100))
                    else:
                        accuracy_percent = 0
                        
                    prediction_accuracy = {
                        'actual': float(current_month_actual),
                        'predicted': float(current_month_prediction),
                        'accuracy': float(accuracy_percent)
                    }
                
                # Predict next month's energy using the model
                if len(energy_values) >= seq_length:
                    # Scale the input data
                    input_seq = energy_scaled[-seq_length:].reshape(1, seq_length, 1)
                    input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(device)
                    
                    # Make predictions for next 30 days (720 hours)
                    next_month_predictions = []
                    current_input = input_tensor.clone()
                    
                    for _ in range(720):  # 30 days * 24 hours
                        with torch.no_grad():
                            next_step = model(current_input).cpu().numpy()
                            next_month_predictions.append(next_step[0, 0])
                            
                            # Update input sequence
                            current_input = torch.cat([
                                current_input[:, 1:, :],
                                torch.tensor(next_step.reshape(1, 1, 1), dtype=torch.float32).to(device)
                            ], dim=1)
                    
                    # Inverse transform and sum
                    next_month_predictions = np.array(next_month_predictions).reshape(-1, 1)
                    next_month_rescaled = scaler.inverse_transform(next_month_predictions)
                    next_month_forecast = np.sum(next_month_rescaled)
                else:
                    # Fallback for short data
                    next_month_forecast = np.mean(energy_values) * 720
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                # Fallback to simple forecasting
                next_month_forecast = np.mean(energy_values) * 720
        else:
            print("No pre-trained model found, using simple forecasting method")
            # Fallback to simple forecasting 
            next_month_forecast = np.mean(energy_values) * 720
        
        # Detect anomalies
        anomalies, threshold = detect_anomalies_by_range(energy_values, threshold_percent=0.70)
        anomaly_count = np.sum(anomalies)
        
        # Get the dates of anomalies
        anomaly_indices = np.where(anomalies)[0]
        anomalies_dates = []
        if len(anomaly_indices) > 0:
            for idx in anomaly_indices:
                if idx < len(data):
                    date_str = data.index[idx].strftime('%Y-%m-%d %H:%M')
                    anomalies_dates.append(date_str)
        
        # Create the visualization with three subplots
        plt.figure(figsize=(12, 12))
        
        # Plot 1: Energy usage
        plt.subplot(3, 1, 1)
        plt.plot(data.index, energy_values, label='Energy Usage', color='blue')
        plt.title(f'Energy Usage for {month}')
        plt.xlabel('Date')
        plt.ylabel('Energy (kWh)')
        plt.legend()
        
        # Plot 2: Anomalies
        plt.subplot(3, 1, 2)
        plt.plot(data.index, energy_values, color='blue', alpha=0.7)
        if len(anomaly_indices) > 0:
            anomaly_values = [energy_values[i] for i in anomaly_indices if i < len(energy_values)]
            anomaly_dates = [data.index[i] for i in anomaly_indices if i < len(data.index)]
            plt.scatter(anomaly_dates, anomaly_values, color='red', label='Anomalies')
        plt.axhline(y=threshold, color='green', linestyle='--', label=f'Threshold')
        plt.title('Anomaly Detection')
        plt.xlabel('Date')
        plt.ylabel('Energy (kWh)')
        plt.legend()
        
        # Plot 3: Prediction Accuracy (for current month)
        plt.subplot(3, 1, 3)
        if prediction_accuracy:
            # Create bar chart for actual vs predicted
            labels = ['Actual', 'Predicted']
            values = [prediction_accuracy['actual'], prediction_accuracy['predicted']]
            
            plt.bar(labels, values, color=['#3366cc', '#ff9900'])
            plt.axhline(y=prediction_accuracy['actual'], color='#3366cc', linestyle='--', alpha=0.6)
            
            # Add value labels on top of each bar
            for i, v in enumerate(values):
                plt.text(i, v + 0.5, f'{v:.2f} kWh', ha='center')
                
            plt.title(f'Model Accuracy for {month} (Hindcast)')
            plt.ylabel('Total Energy (kWh)')
            
            # Add accuracy text to the plot
            accuracy_text = f"Accuracy: {prediction_accuracy['accuracy']:.2f}%"
            plt.figtext(0.7, 0.4, accuracy_text, fontsize=12, 
                       bbox=dict(facecolor='white', alpha=0.8))
        else:
            plt.text(0.5, 0.5, 'Insufficient data for accuracy calculation', 
                    ha='center', va='center', fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.8))
            plt.title('Prediction Accuracy')
        
        plt.tight_layout()
        plt.savefig(result_path)
        plt.close()
        
        return result_path, float(next_month_forecast), int(anomaly_count), anomalies_dates, prediction_accuracy if prediction_accuracy else None
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return default values in case of error
        return '', 0, 0, [], None

if __name__ == '__main__':
    app.run(debug=True)