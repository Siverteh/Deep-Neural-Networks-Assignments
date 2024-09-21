import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.ops as ops
from balloon_dataset import BalloonDataset

class ObjectDetection:
    def __init__(self, train_dir, val_dir, batch_size=4, lr=0.0001, num_epochs=50, device=None):
        """
        Object Detection class to handle Mask R-CNN training and validation.
        """
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the Mask R-CNN model pre-trained on COCO
        self.model = maskrcnn_resnet50_fpn(pretrained=True)

        # Replace the pre-trained head with a new one (2 classes: background + balloon)
        num_classes = 2  # 1 class (balloon) + background
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace the mask predictor with a new one (2 classes: background + balloon)
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        self.model.to(self.device)

        # Set up the optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(params, lr=self.lr)

        # Load the datasets
        self.train_dataset = BalloonDataset(train_dir, train=True)
        self.val_dataset = BalloonDataset(val_dir, train=False)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def train(self):
        """Train the model using the training set."""
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for images, targets in self.train_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                self.optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                self.optimizer.step()

                running_loss += losses.item()

            epoch_loss = running_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")

    def validate(self):
        """Validate the model using Intersection over Union (IoU)."""
        self.model.eval()
        total_iou = 0.0
        num_boxes = 0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Get predictions
                outputs = self.model(images)

                for i in range(len(images)):
                    pred_boxes = outputs[i]['boxes']
                    true_boxes = targets[i]['boxes']

                    if pred_boxes.shape[0] > 0 and true_boxes.shape[0] > 0:
                        # Calculate IoU between predicted and true boxes
                        iou = ops.box_iou(pred_boxes, true_boxes)
                        max_iou, _ = iou.max(dim=0)
                        total_iou += max_iou.sum().item()
                        num_boxes += max_iou.shape[0]

        avg_iou = total_iou / num_boxes if num_boxes > 0 else 0.0
        print(f"Mean IoU: {avg_iou:.4f}")


# Example usage
if __name__ == "__main__":
    detector = ObjectDetection(train_dir='datasets/balloon/train', val_dir='datasets/balloon/val', batch_size=4, lr=0.0005, num_epochs=30)
    
    # Train the model
    detector.train()

    # Validate the model
    detector.validate()
