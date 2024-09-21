# data_loader.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class EcoliDataLoader:
    def __init__(self, dataset_path='/workspaces/Small_Assignments/datasets/ecoli/ecoli.data', test_size=0.2, random_state=42):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.random_state = random_state

    def load_and_preprocess_data(self):
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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        return X_train, X_test, y_train, y_test
