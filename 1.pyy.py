import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build a simple neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to 1D array of 784 elements
    Dense(128, activation='relu'),  # First hidden layer with 128 neurons
    Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each digit)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Predict on a sample from the test set
predictions = model.predict(X_test)

# Display the first 5 predictions and the corresponding images
for i in range(5):
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f'Predicted: {np.argmax(predictions[i])}, Actual: {y_test[i]}')
    plt.show()
