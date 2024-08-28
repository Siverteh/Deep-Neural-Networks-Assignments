import pandas as pd
import numpy as np
from collections import Counter
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.model_selection import train_test_split
from pathlib import Path

# K-Nearest Neighbors Classifier
class KNN:
    def __init__(self, dataset: Path = 'datasets/pima_indians_diabetes/pima-indians-diabetes.csv'):
        """
        Initialize the KNN classifier with the number of neighbors (k).
        """
        self.k = None
        self.dataset = dataset
        self.df = pd.read_csv(self.dataset)

    def fit(self, X_train, y_train):
        """
        Fit the model with the training data.
        X_train: Training features.
        y_train: Training labels.
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def euclidean_distance(self, x1, x2):
        """
        Calculate the Euclidean distance between two points.
        x1: The first point.
        x2: The second point.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        """
        Predict the class of a single sample x.
        x: A single test instance.
        """
        predictions = []

        for x in X_test:
            # Compute distances between x and all examples in the training set
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            
            # Sort by distance and get the indices of the k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Extract the labels of the k nearest neighbor training samples
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Return the most common class label
            most_common = Counter(k_nearest_labels).most_common(1)

            predictions.append(most_common[0][0])

        return np.array(predictions)
    
    def _calculate_metrics(self, k_values: int) -> list[dict[float, float, float, float, float]]:
        metrics_results = []
        
        # Splitting the dataset into features and labels
        X = self.df.iloc[:, :-1].values  # Features
        y = self.df.iloc[:, -1].values   # Labels

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for k in k_values:
            metrics = {}
            self.k = k
            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)

            metrics["accuracy"] = accuracy_score(y_test, y_pred)

    def run_knn(self, max_k):
        """
        Run the KNN algorithm for values of k from 1 to max_k, and plot the results.
        max_k: The maximum value of k to evaluate.
        """
        # Load dataset
        df = pd.read_csv(self.dataset)

        # Splitting the dataset into features and labels
        X = df.iloc[:, :-1].values  # Features
        y = df.iloc[:, -1].values   # Labels

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Lists to store metrics for each k
        k_values = list(range(1, max_k + 1))
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        mses = []

        # Loop through each value of k from 1 to max_k
        for k in k_values:
            self.k = k
            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Store metrics
            accuracies.append(accuracy)
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            mses.append(mse)

            # Print metrics for the current k
            print(f'k={k}: Accuracy={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, MSE={mse:.4f}')

        # Find the best k based on accuracy
        best_k = k_values[np.argmax(accuracies)]
        print(f'\nBest k based on accuracy: k={best_k} with Accuracy={max(accuracies):.4f}')

        # Create a single plot for all metrics
        fig = go.Figure()

        # Plot Accuracy
        fig.add_trace(go.Scatter(x=k_values, y=accuracies, mode='lines+markers', name='Accuracy'))

        # Plot F1 Score
        fig.add_trace(go.Scatter(x=k_values, y=f1_scores, mode='lines+markers', name='F1 Score'))

        # Plot Precision
        fig.add_trace(go.Scatter(x=k_values, y=precisions, mode='lines+markers', name='Precision'))

        # Plot Recall
        fig.add_trace(go.Scatter(x=k_values, y=recalls, mode='lines+markers', name='Recall'))

        # Plot MSE (inverted to show lower is better)
        fig.add_trace(go.Scatter(x=k_values, y=[-mse for mse in mses], mode='lines+markers', name='MSE (Inverted)'))

        # Update layout to make the plot fill the entire window
        fig.update_layout(
            title="KNN Metrics by k",
            xaxis_title="k",
            yaxis_title="Metric Value",
            legend_title="Metrics",
            autosize=True,
            height=None,  # Set height to None for automatic full-screen height
            width=None,   # Set width to None for automatic full-screen width
            margin=dict(l=0, r=0, t=30, b=0),  # Minimal margins for full-screen effect
            template="plotly_white",  # Optional: Use a clean white background
        )

        # Make the plot responsive
        fig.update_layout(
            autosize=True,
            height=None,
            width=None,
            margin=dict(l=0, r=0, t=30, b=0),
        )

        # Show plot
        fig.show()

# Example of using the class
knn = KNN()
knn.run_knn(100)
