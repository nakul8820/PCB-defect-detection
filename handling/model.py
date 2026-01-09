import torch
import numpy as np
import lightning as L
import wandb
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def get_pcb_model(num_classes=7, use_pretrained=True):
    """Create Faster R-CNN model optimized for PCB defect detection."""
    
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    
    anchor_sizes = ((4,), (8,), (16,), (32,), (64,)) 
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    
    model.roi_heads.detections_per_img = 15
    model.roi_heads.score_thresh = 0.4
    model.roi_heads.nms_thresh = 0.3
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


class PCBInspector(L.LightningModule):
    """PyTorch Lightning module for PCB defect detection."""
    
    def __init__(self, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        self.class_names = [
            'background','missing_hole', 'mouse_bite', 
            'open_circuit', 'short', 'spur', 'spurious_copper'
        ]
        
        self.model = get_pcb_model(num_classes=len(self.class_names), use_pretrained=True)
        self.lr = lr
        self.map_metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
        
        self.val_pred_counts = []
        self.val_gt_counts = []

    def training_step(self, batch, batch_idx):
        images, targets = batch
        
        for i, target in enumerate(targets):
            if len(target['boxes']) == 0:
                continue
            assert target['boxes'].shape[1] == 4, f"Invalid box shape: {target['boxes'].shape}"
            assert (target['boxes'][:, 2] > target['boxes'][:, 0]).all(), "Invalid boxes in training"
            assert (target['boxes'][:, 3] > target['boxes'][:, 1]).all(), "Invalid boxes in training"
        
        loss_dict = self.model(list(images), targets)
        
        for loss_name, loss_value in loss_dict.items():
            self.log(f"train/{loss_name}", loss_value)

        losses = sum(loss for loss in loss_dict.values())
        self.log("train/total_loss", losses, prog_bar=True)
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(list(images))

        filtered_outputs = []
        for out in outputs:
            mask = out['scores'] > 0.2
            filtered_outputs.append({
                'boxes': out['boxes'][mask][:15],
                'labels': out['labels'][mask][:15],
                'scores': out['scores'][mask][:15]
            })

        self.map_metric.update(filtered_outputs, targets)
    
        if batch_idx == 0:
            self._log_inspection_data(images[0], filtered_outputs[0], targets[0])
            
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        metrics = self.map_metric.compute()
        self.log("test/mAP", metrics["map"].item())
        
        if self.trainer.is_global_zero:
            mAP_val = float(metrics['map'])
            print(f"DONE. Test mAP: {mAP_val:.4f}")
        
        self.map_metric.reset()
    
    def on_validation_epoch_end(self):
        metrics = self.map_metric.compute()
        self.log("val/mAP", metrics["map"], prog_bar=True)
        self.log("val/mAP_50", metrics["map_50"])
        self.log("val/mAP_75", metrics["map_75"])
        
        if len(self.val_pred_counts) > 0:
            avg_preds = np.mean(self.val_pred_counts)
            avg_gts = np.mean(self.val_gt_counts)
            self.log("debug/avg_pred_count", avg_preds)
            self.log("debug/avg_gt_count", avg_gts)
            print(f"\nEpoch {self.current_epoch}: Avg Predictions={avg_preds:.1f}, Avg GT={avg_gts:.1f}")
        
        if "map_per_class" in metrics:
            for i, name in enumerate(self.class_names[1:]): 
                if i < metrics["map_per_class"].size(0):
                    class_score = metrics["map_per_class"][i]
                    self.log(f"val/mAP_{name}", class_score)

        self.map_metric.reset()
        self.val_pred_counts = []
        self.val_gt_counts = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=0.0001
        )
        
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }

    def _log_inspection_data(self, img, pred, target, key="val/visual_report"):
        img_np = img.permute(1, 2, 0).cpu().numpy()
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = (std * img_np + mean).clip(0, 1) 

        class_id_to_name = {i: name for i, name in enumerate(self.class_names)}

        self.logger.experiment.log({
            key: wandb.Image(img_np, boxes={
                "predictions": {
                    "box_data": self._format_boxes(pred, is_pred=True),
                    "class_labels": class_id_to_name
                },
                "ground_truth": {
                    "box_data": self._format_boxes(target, is_pred=False),
                    "class_labels": class_id_to_name
                }
            })
        })

    def _format_boxes(self, output, is_pred=True):
        if len(output['boxes']) == 0:
            return []
            
        formatted_boxes = []
        boxes = output['boxes'].detach().cpu().numpy()
        labels = output['labels'].detach().cpu().numpy()
        
        if is_pred and 'scores' in output:
            scores = output['scores'].detach().cpu().numpy()
        else:
            scores = np.ones(len(labels))
            
        max_boxes = 20
        count = 0
    
        for box, label, score in zip(boxes, labels, scores):
            if count >= max_boxes:
                break
                
            if is_pred and score < 0.15:
                continue
                
            formatted_boxes.append({
                "position": {
                    "minX": float(box[0]), 
                    "minY": float(box[1]), 
                    "maxX": float(box[2]), 
                    "maxY": float(box[3])
                },
                "class_id": int(label), 
                "scores": {"confidence": float(score)},
                "domain": "pixel"
            })
            count += 1
    
        return formatted_boxes
