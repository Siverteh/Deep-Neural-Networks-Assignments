import pandas as pd
import numpy as np
from collections import Counter
import plotly.graph_objs as go
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Metrics:
    accuracy: float
    f1: float
    precision: float
    recall: float
    mse: float

    def __str__(self):
        return (f"Accuracy={self.accuracy:.4f}\nF1={self.f1:.4f}\n"
                f"Precision={self.precision:.4f}\nRecall={self.recall:.4f}\n"
                f"MSE={self.mse:.4f}")

class KNN:
    def __init__(self, dataset: Path = 'datasets/pima_indians_diabetes/pima-indians-diabetes.csv'):
        """
        Initialize the KNN classifier with a dataset.
        """
        self.dataset = np.loadtxt(dataset, delimiter=",")
        self.k = None
        self.X_train = None
        self.y_train = None

    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate the Euclidean distance between two points using np.linalg.norm.
        """
        return np.linalg.norm(x1 - x2)


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict the class for each sample in X_test.
        Handles ties using the same method as the first code with Counter.most_common(1).
        """
        predictions = []
        for x in X_test:
            # Calculate distances between the query point and all training points
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            # Sort distances and get the k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Use Counter.most_common(1) to get the label with the highest count (majority voting)
            predict_label = Counter(k_nearest_labels).most_common(1)[0][0]
            
            predictions.append(predict_label)
        
        return np.array(predictions)



    @staticmethod
    def manual_train_test_split(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the dataset into training and test sets manually with shuffling.
        Ensures the same behavior as the first code.
        """
        np.random.seed(6)  # Set random seed for reproducibility
        indices = np.random.permutation(len(X))  # Shuffle the indices
        X = X[indices]
        y = y[indices]
        # split 80/20
        split_ratio = 0.8
        split_index = int(len(X) * split_ratio)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        
        return X_train, X_test, y_train, y_test


    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float, float]:
        """
        Calculate accuracy, precision, recall, F1 score, and mean squared error.
        """
        # Precision, Recall, F1 Score
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = np.mean(y_true == y_pred)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        mse = np.mean((y_true - y_pred) ** 2)
        confusion_matrix = np.array([[tn, fp], [fn, tp]])        
        
        return accuracy, precision, recall, f1, mse, confusion_matrix

    def get_all_metrics(self, k_values: List[int]) -> List[Metrics]:
        X = self.dataset[:, :-1]
        y = self.dataset[:, -1]

        X_train, X_test, y_train, y_test = self.manual_train_test_split(X, y)

        all_metrics = []
        for k in k_values:
            self.k = k
            self.X_train, self.y_train = X_train, y_train
            y_pred = self.predict(X_test)

            accuracy, precision, recall, f1, mse, cm = self.calculate_metrics(y_test, y_pred)

            metrics = Metrics(accuracy=accuracy, f1=f1, precision=precision, recall=recall, mse=mse)
            
            # Print the results for the current k
            print('--------------------------------------------------------')
            print("K VALUE:")
            print(f'k={k}')
            print()
            print("RESULTS:")
            print(metrics)
            print()
            print('CONFUSION MATRIX:')
            print('                Predicted Negative   Predicted Positive')
            print(f'Actual Negative       {cm[0, 0]:<18} {cm[0, 1]}')
            print(f'Actual Positive       {cm[1, 0]:<18} {cm[1, 1]}')
            print('--------------------------------------------------------\n')
            
            all_metrics.append(metrics)

        return all_metrics

    def _visualize(self, k_values: List[int], metrics: List[Metrics]):
        fig = go.Figure()
        # Plot Accuracy
        fig.add_trace(go.Scatter(x=k_values, y=[m.accuracy for m in metrics], mode='lines+markers', name='Accuracy'))
        # Plot Precision
        fig.add_trace(go.Scatter(x=k_values, y=[m.precision for m in metrics], mode='lines+markers', name='Precision'))
        # Plot F1 Score
        fig.add_trace(go.Scatter(x=k_values, y=[m.f1 for m in metrics], mode='lines+markers', name='F1 Score'))
        # Plot Recall
        fig.add_trace(go.Scatter(x=k_values, y=[m.recall for m in metrics], mode='lines+markers', name='Recall'))
        # Plot MSE (Inverted to show lower is better)
        fig.add_trace(go.Scatter(x=k_values, y=[m.mse for m in metrics], mode='lines+markers', name='MSE'))

        # Add a secondary y-axis for loss
        fig.update_layout(
            title={
                'text': "KNN Metrics by k",
                'y': 0.95,  # Move title slightly higher to avoid overlap with the plot
                'x': 0.5,  # Centers the title horizontally
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 40}  # Adjusts the title font size
            },
            xaxis_title={
                'text': "k",
                'font': {'size': 40}  # Adjusts the x-axis title font size
            },
            yaxis_title={
                'text': "Metric Value",
                'font': {'size': 40}  # Adjusts the y-axis title font size
            },
            yaxis2=dict(
                title="Loss (MSE)",
                overlaying="y",
                side="right",
                titlefont={'size': 40}  # Adjusts the secondary y-axis title font size
            ),
            # Make the x and y tick values (numbers) larger
            xaxis=dict(
                tickfont=dict(size=40)  # Adjust the x-axis tick font size
            ),
            yaxis=dict(
                tickfont=dict(size=40)  # Adjust the y-axis tick font size
            ),
            legend_title="Metrics",
            template="plotly_white",
            autosize=True,
            margin=dict(l=0, r=0, t=50, b=0),  # Adjust margin to give more space for the title
            # Increase the font size of the legend
            legend=dict(
                font=dict(size=40)  # Adjust the size to your preference
            )
        )
        
        fig.show()

    def run_knn(self, max_k: int):
        k_values = list(range(1, max_k + 1))
        metrics = self.get_all_metrics(k_values)
        best_k = np.argmax([m.accuracy for m in metrics]) + 1

        print(f'Best k based on accuracy: k={best_k} with Accuracy={max(m.accuracy for m in metrics):.4f}')
        self._visualize(k_values, metrics)
