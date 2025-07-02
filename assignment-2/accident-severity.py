# Accident Severity Prediction using Linear Regression

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Create a synthetic dataset (replace this with your real dataset)
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    
    # Generate random features
    weather = np.random.randint(1, 5, num_samples)  # 1=Clear, 2=Rain, 3=Fog, 4=Snow
    road_type = np.random.randint(1, 5, num_samples) # 1=Urban, 2=Rural, 3=Highway, 4=Construction
    speed = np.random.randint(30, 121, num_samples)  # Speed limit from 30-120 km/h
    alcohol = np.random.randint(0, 2, num_samples)    # 0=No, 1=Yes
    
    # Calculate severity (base + weighted factors + noise)
    severity = (
        1.0 + 
        0.5 * weather + 
        0.8 * road_type + 
        0.02 * speed + 
        1.5 * alcohol + 
        np.random.normal(0, 0.5, num_samples)
    )
    
    # Clip severity between 1-5 and round
    severity = np.clip(severity, 1, 5)
    severity = np.round(severity)
    
    # Create DataFrame
    data = pd.DataFrame({
        'weather': weather,
        'road_type': road_type,
        'speed_limit': speed,
        'alcohol_involved': alcohol,
        'severity': severity
    })
    
    return data

# Generate and explore the dataset
df = generate_synthetic_data(1000)
print("Dataset head:")
print(df.head())
print("\nDataset description:")
print(df.describe())

# Visualize the data
plt.figure(figsize=(12, 8))
sns.pairplot(df, hue='severity', palette='viridis')
plt.suptitle('Feature Relationships by Accident Severity', y=1.02)
plt.show()

# Prepare data for modeling
X = df.drop('severity', axis=1)
y = df['severity']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Creating and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Actual Severity')
    plt.ylabel('Predicted Severity')
    plt.title('Actual vs Predicted Accident Severity')
    plt.grid(True)
    plt.show()
    
    return y_pred

# Evaluate the model
y_pred = evaluate_model(model, X_test, y_test)

# Save the model for future use
model_filename = 'accident_severity_model.pkl'
joblib.dump(model, model_filename)
print(f"\nModel saved as {model_filename}")

# Example prediction function
def predict_severity(model, weather, road_type, speed, alcohol):
    input_data = pd.DataFrame({
        'weather': [weather],
        'road_type': [road_type],
        'speed_limit': [speed],
        'alcohol_involved': [alcohol]
    })
    
    severity = model.predict(input_data)[0]
    severity = max(1, min(5, round(severity)))  # Ensure within 1-5 range
    
    print("\nPrediction Results:")
    print(f"Weather: {['Clear', 'Rain', 'Fog', 'Snow'][weather-1]}")
    print(f"Road Type: {['Urban', 'Rural', 'Highway', 'Construction'][road_type-1]}")
    print(f"Speed Limit: {speed} km/h")
    print(f"Alcohol Involved: {'Yes' if alcohol else 'No'}")
    print(f"Predicted Accident Severity: {severity} (on scale of 1-5)")
    
    return severity

# Make a sample prediction
sample_prediction = predict_severity(
    model=model,
    weather=2,      # Rain
    road_type=3,    # Highway
    speed=110,      # 110 km/h
    alcohol=1       # Yes
)

# Feature importance
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

print("\nFeature Importance:")
print(coefficients)