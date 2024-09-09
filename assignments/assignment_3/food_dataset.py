import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Dictionary used for mapping the image numbers (0-10) to their respective food categories.
class_mapping = {
    0: 'Bread',
    1: 'Dairy product',
    2: 'Dessert',
    3: 'Egg',
    4: 'Fried food',
    5: 'Meat',
    6: 'Noodles-Pasta',
    7: 'Rice',
    8: 'Seafood',
    9: 'Soup',
    10: 'Vegetable-Fruit'
}

# Custom dataset class to handle images and their associated labels.
class FoodDataset(Dataset):
    # The constructor takes the directory of images and a set of transformations.
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # Root directory where the images are stored.
        self.transform = transform  # Optional image transformations.
        self.image_paths = []  # List to store paths of all images.
        self.labels = []  # List to store the labels (classes) of the images.
        
        # Loop through the directory to extract labels from filenames.
        for fname in os.listdir(root_dir):
            # Check if the file is an image with .jpg or .png extension.
            if fname.endswith(".jpg") or fname.endswith(".png"):
                # Extract the class label (the number before the '_') from the filename.
                label = fname.split('_')[0]
                # Add the full path of the image to the image_paths list.
                self.image_paths.append(os.path.join(root_dir, fname))
                # Convert the extracted label to an integer and add it to the labels list.
                self.labels.append(int(label))

    # Return the total number of images in the dataset.
    def __len__(self):
        return len(self.image_paths)

    # Return a single image and its corresponding label.
    def __getitem__(self, idx):
        # Retrieve the path of the image at index 'idx'.
        img_path = self.image_paths[idx]
        # Open the image file using the PIL library.
        image = Image.open(img_path)
        
        # Apply any transformations if provided (e.g., resizing, normalization).
        if self.transform:
            image = self.transform(image)
        
        # Retrieve the label for the image at index 'idx'.
        label = self.labels[idx]
        # Return both the image and its label.
        return image, label

# Function to create DataLoader objects for training, validation, and testing datasets.
def create_data_loaders(data_dir, batch_size=32):
    # Define the set of transformations to apply to the images (resize, flip, convert to tensor, normalize).
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (common size for CNNs like ResNet).
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally for augmentation.
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image to match pre-trained model statistics.
    ])

    # Create datasets for training, validation, and testing, applying the defined transformations.
    train_dataset = FoodDataset(root_dir=os.path.join(data_dir, 'training'), transform=transform)
    val_dataset = FoodDataset(root_dir=os.path.join(data_dir, 'validation'), transform=transform)
    test_dataset = FoodDataset(root_dir=os.path.join(data_dir, 'evaluation'), transform=transform)

    # Create DataLoader objects to handle batching and shuffling of the data.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle the training data.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Do not shuffle validation data.
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Do not shuffle test data.

    # Return the three data loaders.
    return train_loader, val_loader, test_loader
