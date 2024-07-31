# Linear Regression with TensorFlow

This repository demonstrates a simple linear regression using TensorFlow and Keras.

## Overview

In this example, we generate synthetic data for the linear function \( y = 3x + 2 \) with some Gaussian noise. A neural network model is then created to learn and predict this function. The model is trained using stochastic gradient descent (SGD) with a mean squared error loss function.

## Code Overview

The code is divided into the following sections:

1. **Data Generation**: Generate synthetic data for the linear function with some noise.
2. **Model Creation**: Define a simple neural network model with one dense layer.
3. **Model Training**: Compile and train the model using stochastic gradient descent.
4. **Visualization**: Plot the original data and the model's predictions.

## Code

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for a linear function y = 3x + 2
# Create 1000 evenly spaced values between -10 and 10
x = np.linspace(-10, 10, 1000) 

# Calculate corresponding y values for the linear function with added Gaussian noise
y = 3*x + 2 + np.random.normal(scale=5, size=x.shape)

# Create the model
# Define a Sequential model with a single Dense layer
# The Dense layer has 1 unit and an input shape of 1
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]),
])

# Model hyperparameters
# Compile the model with stochastic gradient descent (SGD) optimizer and mean squared error loss
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='mean_squared_error')

# Train the model
# Fit the model to the data with 5 epochs and display training progress
model.fit(x, y, epochs=5, verbose=1)

# Plot the original data and the model prediction
# Scatter plot of original data points in blue
plt.scatter(x, y, color='blue', label='Original Data')

# Line plot of model predictions in red
plt.plot(x, model.predict(x), color='red', label='Model Prediction')

# Add a legend to the plot
plt.legend()

# Display the plot
plt.show()
