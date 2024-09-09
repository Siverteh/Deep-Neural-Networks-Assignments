import torch
print(torch.cuda.is_available())
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from food_dataset import create_data_loaders, class_mapping

# Define the Custom CNN Model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 11)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the Classifier
class CustomCNNClassifier:
    def __init__(self, data_dir='datasets/food-11', batch_size=32, lr=0.001, num_epochs=20):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare data
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(self.data_dir, self.batch_size)

    def train(self, model, optimizer, criterion):
        model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {running_loss/len(self.train_loader):.4f}")

    def evaluate(self, model, loader):
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
        return all_labels, all_preds

    def validate_and_test(self, model):
        # Validation
        print("Validating the model...")
        val_labels, val_preds = self.evaluate(model, self.val_loader)
        print("Validation Classification Report:")
        print(classification_report(val_labels, val_preds, target_names=[class_mapping[i] for i in range(11)]))
        print("Confusion Matrix:")
        print(confusion_matrix(val_labels, val_preds, labels=[i for i in range(11)]))

        # Testing
        print("Testing the model...")
        test_labels, test_preds = self.evaluate(model, self.test_loader)
        print("Test Classification Report:")
        print(classification_report(test_labels, test_preds, target_names=[class_mapping[i] for i in range(11)]))
        print("Confusion Matrix:")
        print(confusion_matrix(test_labels, test_preds, labels=[i for i in range(11)]))


    def run(self):
        # Initialize the model
        print("Running with Custom CNN...")
        model = CustomCNN().to(self.device)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Train and evaluate the model
        self.train(model, optimizer, criterion)
        self.validate_and_test(model)

# Usage
if __name__ == "__main__":
    classifier = CustomCNNClassifier(data_dir='datasets/food-11', batch_size=32, lr=0.001, num_epochs=20)
    classifier.run()
