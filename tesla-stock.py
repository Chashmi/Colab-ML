# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset
spreadsheet = "/content/drive/MyDrive/Dr. Ali Shahzadi/Machine Learning - 4031/Python Homework/HW2/tesla-stock-price_Ex.xlsx"
data = pd.read_excel(spreadsheet)

# Step 2: Preprocessing
# Remove rows with missing values if any
data = data.dropna()

# Check if 'volume' column contains commas and is a string
if data['volume'].dtype == 'object':  # If it's a string with commas
    data['volume'] = data['volume'].str.replace(",", "").astype(float)
else:
    data['volume'] = data['volume'].astype(float)  # Convert to float if it's already numeric

# Convert 'date' column to datetime (optional, if needed for analysis)
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Convert 'date' column to datetime (optional, if needed for analysis)
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Step 3: Define features (X) and target (y)
X = data[['open', 'high', 'low', 'volume']]  # Features
y = data['close']  # Target

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Step 8: Visualize Actual vs Predicted Close Prices
plt.scatter(y_test, y_pred, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linewidth=2)
plt.xlabel("Actual Close Prices")
plt.ylabel("Predicted Close Prices")
plt.title("Actual vs Predicted Close Prices")
plt.show()
