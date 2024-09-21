# mlp_pytorch.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import EcoliDataLoader

class PyTorchMLPClassifier:
    def __init__(self, dataset_path='datasets/ecoli/ecoli.data', test_size=0.2, random_state=42, learning_rate=0.01, epochs=500, batch_size=32):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Load data
        self.load_data()

        # Initialize the model
        input_size = self.X_train_tensor.shape[1]
        hidden_sizes = [10, 5]  # You can adjust this
        output_size = 1
        self.init_model(input_size, hidden_sizes, output_size)

    def load_data(self):
        # Load data
        data_loader = EcoliDataLoader(self.dataset_path, test_size=self.test_size, random_state=self.random_state)
        X_train, X_test, y_train, y_test = data_loader.load_and_preprocess_data()

        # Convert data to torch tensors
        self.X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
        self.y_train_tensor = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
        self.X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
        self.y_test_tensor = torch.from_numpy(y_test.astype(np.float32)).view(-1, 1)

    def init_model(self, input_size, hidden_sizes, output_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.Sigmoid()
        ).to(self.device)

        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.X_train_tensor.to(self.device))
            loss = self.criterion(outputs, self.y_train_tensor.to(self.device))
            loss.backward()
            self.optimizer.step()
            # Compute accuracy
            predictions = (outputs > 0.5).int()
            correct = (predictions.cpu() == self.y_train_tensor.cpu()).sum().item()
            accuracy = correct / self.y_train_tensor.size(0)
            # Optionally print loss and accuracy
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')


    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.X_test_tensor.to(self.device))
            y_pred = (y_pred > 0.5).int().cpu().numpy()
            y_test_np = self.y_test_tensor.numpy()
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test_np, y_pred))
            print("\nClassification Report:")
            print(classification_report(y_test_np, y_pred))

    def run(self):
        self.train()
        self.evaluate()

if __name__ == "__main__":
    # Initialize and run the PyTorch MLP classifier
    classifier = PyTorchMLPClassifier(dataset_path='datasets/ecoli/ecoli.data', learning_rate=0.01, epochs=500)
    classifier.run()
