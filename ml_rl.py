# Linear Regression Machine Learning example:
# Uses data for machine age and time between failures
# Predict a model for the data (supervised ML)

## Import packages
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define your spreadsheet
spreadsheet = "/content/drive/MyDrive/Dr. Ali Shahzadi/Machine Learning - 4031/Python Homework/HW1/Dataset/LR_ML.xlsx"
data = pd.read_excel(spreadsheet)

# Define your useful columns of data
months = data['Machine Age (Months)'].values
MTBF = data['Mean Time Between Failure (Days)'].values

# Normalize the data for better training stability
train_X = np.asarray(months, dtype=np.float32)
train_Y = np.asarray(MTBF, dtype=np.float32)

X_mean, X_std = train_X.mean(), train_X.std()
Y_mean, Y_std = train_Y.mean(), train_Y.std()

train_X = (train_X - X_mean) / X_std
train_Y = (train_Y - Y_mean) / Y_std

# Hyperparameters
learning_rate = 0.05  # Adjusted learning rate
training_epochs = 1000  # Increased epochs for better convergence

# Parameters (weights and bias) initialized to zero
W = tf.Variable(0.0, name="weight", dtype=tf.float32)
b = tf.Variable(0.0, name="bias", dtype=tf.float32)

# Linear model function: y = W * X + b
def linear_model(X):
    return W * X + b

# Loss function: Mean Squared Error (MSE)
def mean_squared_error(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Gradient Descent Optimizer
optimizer = tf.optimizers.SGD(learning_rate)

# Training loop
for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        # Forward pass: compute predictions and loss
        predictions = linear_model(train_X)
        loss = mean_squared_error(predictions, train_Y)

    # Compute gradients
    gradients = tape.gradient(loss, [W, b])

    # Apply gradients to update weights and bias
    optimizer.apply_gradients(zip(gradients, [W, b]))

    # Display logs every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {loss.numpy()}, W: {W.numpy()}, b: {b.numpy()}")

print("Optimization Finished!")
print(f"Final Loss: {loss.numpy()}, W: {W.numpy()}, b: {b.numpy()}")

# Plot training data and fitted line
# Reverse normalization for plotting
train_X_original = train_X * X_std + X_mean
train_Y_original = train_Y * Y_std + Y_mean
fitted_Y = (linear_model(train_X) * Y_std + Y_mean).numpy()

plt.plot(train_X_original, train_Y_original, 'ro', label='Original data')
plt.plot(train_X_original, fitted_Y, label='Fitted line')
plt.legend()
plt.show()

# Testing example
test_X = np.asarray([2, 4, 6, 8, 10], dtype=np.float32)
test_Y = np.asarray([25, 23, 21, 19, 17], dtype=np.float32)

# Normalize test data
test_X = (test_X - X_mean) / X_std
test_Y = (test_Y - Y_mean) / Y_std

# Testing loss
test_predictions = linear_model(test_X)
testing_loss = mean_squared_error(test_predictions, test_Y)
print(f"Testing Loss: {testing_loss.numpy()}")
