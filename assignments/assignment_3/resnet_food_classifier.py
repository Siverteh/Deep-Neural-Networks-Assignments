import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from food_dataset import create_data_loaders, class_mapping  
from torchvision import models

class ResNetClassifier:
    def __init__(self, data_dir='datasets/food-11', batch_size=32, lr=0.001, num_epochs=10):
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
        print("Validating the model...")
        val_labels, val_preds = self.evaluate(model, self.val_loader)
        print("Validation Classification Report:")
        print(classification_report(val_labels, val_preds, target_names=[class_mapping[i] for i in range(11)]))
        print("Confusion Matrix:")
        print(confusion_matrix(val_labels, val_preds, labels=[i for i in range(11)]))

        print("Testing the model...")
        test_labels, test_preds = self.evaluate(model, self.test_loader)
        print("Test Classification Report:")
        print(classification_report(test_labels, test_preds, target_names=[class_mapping[i] for i in range(11)]))
        print("Confusion Matrix:")
        print(confusion_matrix(test_labels, test_preds, labels=[i for i in range(11)]))

    def run(self):
        # Use ResNet18
        print("Running with ResNet18...")
        weights = models.ResNet18_Weights.IMAGENET1K_V1  
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 11) 
        model = model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        self.train(model, optimizer, criterion)
        self.validate_and_test(model)

if __name__ == "__main__":
    classifier = ResNetClassifier(data_dir='datasets/food-11', batch_size=32, lr=0.001, num_epochs=20)
    classifier.run()