import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for a linear function y = 3x + 2
x = np.linspace(-10, 10, 1000) 
y = 3*x + 2 + np.random.normal(scale=5, size=x.shape)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]),
])

# Model hyperparameters
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(x, y, epochs=5, verbose=1)



# Plot the original data and the model prediction
plt.scatter(x, y, color='blue', label='Original Data')
plt.plot(x, model.predict(x), color='red', label='Model Prediction')
plt.legend()
plt.show()