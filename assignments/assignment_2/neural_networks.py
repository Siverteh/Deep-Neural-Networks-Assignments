import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

class MLPModels:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = self._load_and_preprocess_ecoli_data()

    def _load_and_preprocess_ecoli_data(self):
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

        return df

    def run_simple_mlp(self):
        input_size = self.df.shape[1] - 1  # Number of features
        hidden_size = 5  # Arbitrary hidden layer size
        output_size = 1  # Binary classification

        mlp = self.SimpleMLP(input_size, hidden_size, output_size)
        X = self.df.iloc[:, :-1].values  # Features
        output = mlp.forward(X)
        print("Simple MLP output (not trained):", output[:5])  # Print first 5 outputs

    def run_keras_mlp(self):
        self._build_and_train_mlp()

    def _build_and_train_mlp(self):
        # Prepare the data
        X = self.df.iloc[:, :-1].values
        y = self.df.iloc[:, -1].values

        # Standardize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build the MLP model
        model = Sequential([
            Dense(10, input_dim=X_train.shape[1], activation='relu'),  # First hidden layer
            Dense(5, activation='relu'),  # Second hidden layer
            Dense(1, activation='sigmoid')  # Output layer for binary classification
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.1)

        # Evaluate the model
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    class SimpleMLP:
        def __init__(self, input_size, hidden_size, output_size):
            # Initialize weights and biases for a simple 2-layer MLP
            self.W1 = np.random.randn(input_size, hidden_size)
            self.b1 = np.zeros((1, hidden_size))
            self.W2 = np.random.randn(hidden_size, output_size)
            self.b2 = np.zeros((1, output_size))

        def forward(self, X):
            # Forward pass through the network
            self.Z1 = np.dot(X, self.W1) + self.b1
            self.A1 = self.sigmoid(self.Z1)
            self.Z2 = np.dot(self.A1, self.W2) + self.b2
            output = self.sigmoid(self.Z2)
            return output
        
        def sigmoid(self, z):
            # Sigmoid activation function
            return 1 / (1 + np.exp(-z))