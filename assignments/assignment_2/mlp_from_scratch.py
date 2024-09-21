# mlp_from_scratch.py

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import EcoliDataLoader

class MLPFromScratch:
    def __init__(self, dataset_path='datasets/ecoli/ecoli.data', test_size=0.2, random_state=42, learning_rate=0.01, epochs=1000):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Load data
        data_loader = EcoliDataLoader(self.dataset_path, self.test_size, self.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = data_loader.load_and_preprocess_data()

        # Initialize weights
        input_size = self.X_train.shape[1]
        hidden_size = 5  # You can adjust this
        output_size = 1
        self.init_weights(input_size, hidden_size, output_size)
        

    def init_weights(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with standard normal distribution
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        # Derivative of sigmoid function with respect to activation
        return a * (1 - a)

    def forward(self, X):
        # Forward pass
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        # Compute loss (binary cross-entropy)
        loss = -np.sum(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8)) / m
        return loss

    def backward(self, X, y):
        m = y.shape[0]

        # Output layer error
        dZ2 = self.A2 - y.reshape(-1, 1)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer error
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self):
        for epoch in range(self.epochs):
            # Forward pass
            output = self.forward(self.X_train)
            # Compute loss
            loss = self.compute_loss(self.y_train, output)
            # Backward pass and update
            self.backward(self.X_train, self.y_train)
            # Calculate accuracy
            predictions = (output > 0.5).astype(int).flatten()
            accuracy = np.mean(predictions == self.y_train)
            # Print loss and accuracy every epoch
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

    def predict(self, X):
        # Predict function
        output = self.forward(X)
        predictions = (output > 0.5).astype(int).flatten()
        return predictions

    def run(self):
        # Train the model
        self.train()

        # Evaluate the model
        y_pred = self.predict(self.X_test)
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

if __name__ == "__main__":
    # Initialize and run the MLP from scratch
    mlp = MLPFromScratch(dataset_path='datasets/ecoli/ecoli.data', learning_rate=0.01, epochs=1000)
    mlp.run()
