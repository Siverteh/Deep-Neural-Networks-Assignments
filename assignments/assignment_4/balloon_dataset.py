import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import random
import torchvision.transforms.functional as F

class BalloonDataset(Dataset):
    def __init__(self, data_dir, train=True):
        """
        Initialize dataset for a specific directory (train or val), and setup the transformations.
        """
        self.train = train
        self.data_dir = data_dir
        self.transforms = self.get_transform(train)
        self.imgs = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jpg")])

        # Load annotations from the JSON file in the directory
        with open(os.path.join(data_dir, 'via_region_data.json')) as f:
            annotations = json.load(f)

        # Build a mapping from image filenames to annotations
        self.annotations = {}
        for annotation in annotations.values():
            filename = annotation['filename']
            regions = annotation['regions']
            self.annotations[filename] = regions

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img_name = os.path.basename(img_path)
        img = Image.open(img_path).convert("RGB")

        # Load corresponding annotations
        if img_name in self.annotations:
            regions = self.annotations[img_name]
        else:
            regions = {}

        # Extract masks and bounding boxes from polygons
        masks = []
        boxes = []
        for region in regions.values():  # Changed from 'for region in regions:' to 'for region in regions.values():'
            shape_attributes = region['shape_attributes']
            polygon_x = shape_attributes['all_points_x']
            polygon_y = shape_attributes['all_points_y']

            # Create a mask from the polygon
            mask = Image.new('L', (img.width, img.height), 0)
            ImageDraw.Draw(mask).polygon(list(zip(polygon_x, polygon_y)), outline=1, fill=1)
            mask = np.array(mask)
            masks.append(mask)

            # Compute bounding box from the polygon
            x_min = min(polygon_x)
            x_max = max(polygon_x)
            y_min = min(polygon_y)
            y_max = max(polygon_y)
            boxes.append([x_min, y_min, x_max, y_max])

        num_objs = len(boxes)

        # Convert everything into torch tensors
        if num_objs > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            labels = torch.ones((num_objs,), dtype=torch.int64)  # All balloons have label 1
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, img.height, img.width), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        image_id = torch.tensor([idx])

        # Create the target dictionary
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        # Apply transformations
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_transform(self, train):
        """
        Define the transformations: converting to tensor and applying data augmentations.
        """
        transforms = []
        #transforms.append(Resize((512, 512)))  # Ensure you pass a tuple for size
        transforms.append(ToTensor())
        #transforms.append(Normalize(mean=[0.485, 0.456, 0.406],
                                    #std=[0.229, 0.224, 0.225]))
        if train:
            transforms.append(RandomHorizontalFlip(0.5))
            # Optionally add RandomResize if desired
            # transforms.append(RandomResize())
        return Compose(transforms)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Resize(object):
    def __init__(self, size):
        self.size = size  # size should be a tuple (height, width)

    def __call__(self, image, target):
        orig_size = image.size  # PIL format: (width, height)
        image = F.resize(image, self.size)

        # Calculate resize ratios
        ratio_width = self.size[0] / orig_size[0]
        ratio_height = self.size[1] / orig_size[1]

        # Resize boxes
        if 'boxes' in target and target['boxes'].shape[0] > 0:
            boxes = target['boxes']
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio_width
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio_height
            target['boxes'] = boxes

        # Resize masks
        if 'masks' in target and target['masks'].shape[0] > 0:
            masks = target['masks']
            masks = masks.unsqueeze(1).float()  # Add channel dimension
            masks = F.resize(masks, self.size)
            masks = masks.squeeze(1).byte()
            target['masks'] = masks

        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            width = image.size(2)  # image shape: [C, H, W]

            # Flip bounding boxes
            if 'boxes' in target and target['boxes'].shape[0] > 0:
                boxes = target['boxes']
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target['boxes'] = boxes

            # Flip masks
            if 'masks' in target and target['masks'].shape[0] > 0:
                target['masks'] = target['masks'].flip(-1)
        return image, target

class Resize(object):
    def __init__(self, size):
        self.size = size  # size should be a tuple (height, width)

    def __call__(self, image, target):
        orig_size = image.size  # PIL format: (width, height)
        image = F.resize(image, self.size)

        # Calculate resize ratios
        ratio_width = self.size[0] / orig_size[0]
        ratio_height = self.size[1] / orig_size[1]

        # Resize boxes
        if 'boxes' in target and target['boxes'].shape[0] > 0:
            boxes = target['boxes']
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio_width
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio_height
            target['boxes'] = boxes

        # Resize masks
        if 'masks' in target and target['masks'].shape[0] > 0:
            masks = target['masks']
            masks = masks.unsqueeze(1).float()  # Add channel dimension
            masks = F.resize(masks, self.size)
            masks = masks.squeeze(1).byte()
            target['masks'] = masks

        return image, target
