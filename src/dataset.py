import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PCBDataset(Dataset):
    """Dataset class for PCB defect detection."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        img_base = os.path.join(root_dir, 'images')
        ann_base = os.path.join(root_dir, 'Annotations')

        self.class_names = [
            'missing_hole', 'mouse_bite', 'open_circuit', 
            'short', 'spur', 'spurious_copper'
        ]
        self.label_map = {name: i + 1 for i, name in enumerate(self.class_names)}
        
        all_xmls = []
        for root, _, files in os.walk(ann_base):
            for f in files:
                if f.endswith('.xml'):
                    all_xmls.append(os.path.join(root, f))

        for xml_path in all_xmls:
            rel_path = os.path.relpath(xml_path, ann_base)
            img_path = os.path.join(img_base, rel_path).replace('.xml', '.JPG')
            
            if not os.path.exists(img_path):
                img_path = img_path.replace('.JPG', '.jpg')
            
            if os.path.exists(img_path):
                self.samples.append((img_path, xml_path))
        
    def __getitem__(self, idx):
        img_path, xml_path = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tree = ET.parse(xml_path)
        boxes, labels = [], []
        for obj in tree.findall('object'):
            name = obj.find('name').text.lower().strip()
            label_id = self.label_map.get(name)
            if label_id is not None:
                labels.append(label_id)
                bbox = obj.find('bndbox')
                boxes.append([
                    float(bbox.find('xmin').text),
                    float(bbox.find('ymin').text),
                    float(bbox.find('xmax').text),
                    float(bbox.find('ymax').text)
                ])

        boxes = np.array(boxes, dtype=np.float32) if len(boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if len(labels) > 0 else np.zeros((0,), dtype=np.int64)

        if self.transform:
            sample = self.transform(image=image, bboxes=boxes, labels=labels)
            image = sample['image']
            boxes = torch.as_tensor(sample['bboxes'], dtype=torch.float32).reshape(-1, 4)
            labels = torch.as_tensor(sample['labels'], dtype=torch.int64)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return image, target

    def __len__(self):
        return len(self.samples)


class SubsetWithTransform(Dataset):
    """Dataset wrapper for train/val/test splits with different transforms."""
    
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path, xml_path = self.base_dataset.samples[actual_idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        tree = ET.parse(xml_path)
        boxes, labels = [], []
        for obj in tree.findall('object'):
            name = obj.find('name').text.lower().strip()
            label_id = self.base_dataset.label_map.get(name)
            if label_id is not None:
                labels.append(label_id)
                bbox = obj.find('bndbox')
                boxes.append([
                    float(bbox.find('xmin').text),
                    float(bbox.find('ymin').text),
                    float(bbox.find('xmax').text),
                    float(bbox.find('ymax').text)
                ])
        
        boxes = np.array(boxes, dtype=np.float32) if len(boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if len(labels) > 0 else np.zeros((0,), dtype=np.int64)
        
        if self.transform:
            sample = self.transform(image=image, bboxes=boxes, labels=labels)
            image = sample['image']
            boxes = np.array(sample['bboxes'])
            labels = np.array(sample['labels'])

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        
        if len(boxes_tensor) > 0:
            h, w = image.shape[-2:]
            boxes_tensor[:, [0, 2]] = boxes_tensor[:, [0, 2]].clamp(0, w)
            boxes_tensor[:, [1, 3]] = boxes_tensor[:, [1, 3]].clamp(0, h)
            
            keep = (boxes_tensor[:, 2] > boxes_tensor[:, 0]) & (boxes_tensor[:, 3] > boxes_tensor[:, 1])
            boxes_tensor = boxes_tensor[keep]
            labels_tensor = labels_tensor[keep]

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([actual_idx])
        }
        return image, target
    
    def __len__(self):
        return len(self.indices)


def get_transforms():
    """Returns training and validation transforms."""
    
    train_transform = A.Compose([
        A.Resize(800, 800),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.CLAHE(p=0.5),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(10, 32), 
                       hole_width_range=(10, 32), p=0.2),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'],
                                min_visibility=0.1, min_area=1.0))

    val_transform = A.Compose([
        A.Resize(800, 800),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    return train_transform, val_transform
