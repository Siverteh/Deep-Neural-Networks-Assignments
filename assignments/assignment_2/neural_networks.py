import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

class EcoliClassifier:
    def __init__(self, dataset_path='datasets/ecoli/ecoli.data', test_size=0.2, random_state=42, learning_rate=0.01, epochs=1000, batch_size=32):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Load and preprocess data
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_preprocess_ecoli_data()

    def load_and_preprocess_ecoli_data(self):
        # Load the dataset
        df = pd.read_csv(self.dataset_path, delim_whitespace=True, header=None)

        # Assign column names
        df.columns = ['sequence_name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']

        # Filter the dataset to only include 'cp' and 'im' classes
        df = df[df['class'].isin(['cp', 'im'])]

        # Map the class labels to binary values: 'cp' -> 0, 'im' -> 1
        df['class'] = df['class'].map({'cp': 0, 'im': 1})

        # Drop the sequence_name column as it's not needed
        df = df.drop(columns=['sequence_name'])

        # Separate features and labels
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Standardize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split the data into training and test sets
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def run_scratch(self):
        print("Running MLP from Scratch...")
        input_size = self.X_train.shape[1]
        hidden_size = 5
        output_size = 1

        mlp_scratch = self.MLPFromScratch(input_size, hidden_size, output_size, self.learning_rate)
        mlp_scratch.train(self.X_train, self.y_train, self.epochs)

        # Evaluate MLP from Scratch
        y_pred_scratch = mlp_scratch.predict(self.X_test)
        print("\nMLP From Scratch Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred_scratch))
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred_scratch))

    def run_pytorch(self):
        print("Running MLP using PyTorch...")
        # Convert data to torch tensors
        X_train_tensor = torch.from_numpy(self.X_train.astype(np.float32))
        y_train_tensor = torch.from_numpy(self.y_train.astype(np.float32)).view(-1, 1)
        X_test_tensor = torch.from_numpy(self.X_test.astype(np.float32))
        y_test_tensor = torch.from_numpy(self.y_test.astype(np.float32)).view(-1, 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_size = self.X_train.shape[1]
        hidden_sizes = [10, 5]
        output_size = 1

        mlp_pytorch = self.MLPPyTorch(input_size, hidden_sizes, output_size).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(mlp_pytorch.parameters(), lr=self.learning_rate)

        # Train the PyTorch MLP
        epochs = self.epochs
        mlp_pytorch.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = mlp_pytorch(X_train_tensor.to(device))
            loss = criterion(outputs, y_train_tensor.to(device))
            loss.backward()
            optimizer.step()
            # Optionally print loss
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        # Evaluate the PyTorch MLP
        mlp_pytorch.eval()
        with torch.no_grad():
            y_pred_pytorch = mlp_pytorch(X_test_tensor.to(device))
            y_pred_pytorch = (y_pred_pytorch > 0.5).int().cpu().numpy()
            y_test_np = y_test_tensor.numpy()
            print("\nPyTorch MLP Confusion Matrix:")
            print(confusion_matrix(y_test_np, y_pred_pytorch))
            print("\nClassification Report:")
            print(classification_report(y_test_np, y_pred_pytorch))

    class MLPFromScratch:
        def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
            # Initialize weights and biases
            self.W1 = np.random.randn(input_size, hidden_size)
            self.b1 = np.zeros((1, hidden_size))
            self.W2 = np.random.randn(hidden_size, output_size)
            self.b2 = np.zeros((1, output_size))
            self.learning_rate = learning_rate

        def sigmoid(self, z):
            # Sigmoid activation function
            return 1 / (1 + np.exp(-z))

        def sigmoid_derivative(self, z):
            # Derivative of sigmoid function
            s = self.sigmoid(z)
            return s * (1 - s)

        def forward(self, X):
            # Forward pass
            self.Z1 = np.dot(X, self.W1) + self.b1
            self.A1 = self.sigmoid(self.Z1)
            self.Z2 = np.dot(self.A1, self.W2) + self.b2
            self.A2 = self.sigmoid(self.Z2)
            return self.A2

        def backward(self, X, y):
            # Backward pass
            m = y.shape[0]

            # Output layer error
            dZ2 = self.A2 - y.reshape(-1, 1)
            dW2 = np.dot(self.A1.T, dZ2) / m
            db2 = np.sum(dZ2, axis=0, keepdims=True) / m

            # Hidden layer error
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self.sigmoid_derivative(self.Z1)
            dW1 = np.dot(X.T, dZ1) / m
            db1 = np.sum(dZ1, axis=0, keepdims=True) / m

            # Update weights and biases
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

        def train(self, X, y, epochs=1000):
            for epoch in range(epochs):
                # Forward pass
                output = self.forward(X)
                # Compute loss (binary cross-entropy)
                loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
                # Backward pass and update
                self.backward(X, y)
                # Optionally print loss every 100 epochs
                if (epoch + 1) % 100 == 0:
                    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

        def predict(self, X):
            # Predict function
            output = self.forward(X)
            predictions = (output > 0.5).astype(int).flatten()
            return predictions

    class MLPPyTorch(nn.Module):
        def __init__(self, input_size, hidden_sizes, output_size):
            super(EcoliClassifier.MLPPyTorch, self).__init__()
            # Define layers
            layers = []
            layer_sizes = [input_size] + hidden_sizes
            for i in range(len(hidden_sizes)):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_sizes[-1], output_size))
            layers.append(nn.Sigmoid())
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

if __name__ == "__main__":
    # Initialize the classifier
    classifier = EcoliClassifier(dataset_path='ecoli.data', learning_rate=0.01, epochs=1000)

    # Run MLP from Scratch
    classifier.run_scratch()

    # Alternatively, run MLP using PyTorch
    # classifier = EcoliClassifier(dataset_path='ecoli.data', learning_rate=0.01, epochs=500)
    # classifier.run_pytorch()
