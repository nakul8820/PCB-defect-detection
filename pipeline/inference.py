import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from torchvision.ops import nms

from model import PCBInspector


def plot_pcb_random_results(model_path, test_ds, device='cuda', 
                            conf_thresh=0.45, sample_count=8):
    """Generate and log inference results for random test samples."""
    
    model = PCBInspector.load_from_checkpoint(model_path)
    model.to(device)
    model.eval()

    cols = 2
    rows = (sample_count + cols - 1) // cols
    fig, axes = plt.subplots(4, 3, figsize=(20, 24))
    axes = axes.flatten()
    
    indices = random.sample(range(len(test_ds)), sample_count)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    print(f"Generating report grid for {sample_count} images...")

    for i, idx in enumerate(indices):
        image_tensor, target = test_ds[idx]
        
        with torch.no_grad():
            prediction = model.model([image_tensor.to(device)])[0]

        boxes = prediction['boxes']
        scores = prediction['scores']
        labels = prediction['labels']
        
        mask = scores > conf_thresh
        keep = nms(boxes[mask], scores[mask], iou_threshold=0.2)
        
        final_boxes = boxes[mask][keep].cpu().numpy()
        final_labels = labels[mask][keep].cpu().numpy()
        final_scores = scores[mask][keep].cpu().numpy()

        img_plot = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_plot = (img_plot * std + mean).clip(0, 1)
        
        ax = axes[i]
        ax.imshow(img_plot)
        
        colors = plt.cm.get_cmap('rainbow', 7)

        for box, label, score in zip(final_boxes, final_labels, final_scores):
            name = model.class_names[label]
            c = colors(label)
            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                 fill=False, color=c, linewidth=2)
            ax.add_patch(rect)
            ax.text(box[0], box[1]-4, f"{name} {score:.2f}", 
                    fontsize=8, color='white', bbox=dict(facecolor=c, alpha=0.5, pad=0))

        ax.set_title(f"Test Image Index: {idx}", fontsize=12)
        ax.axis('off')

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    wandb.log({"Inference_Report_Grid": [wandb.Image(plt, caption="Final Inference Grid")]})
    plt.savefig("pcb_report_grid.png", dpi=300, bbox_inches='tight')
    plt.show()


def infer_single_image(model_path, image_tensor, device='cuda', 
                      conf_thresh=0.45, nms_thresh=0.2):
    """Run inference on a single image."""
    
    model = PCBInspector.load_from_checkpoint(model_path)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        prediction = model.model([image_tensor.to(device)])[0]
    
    boxes = prediction['boxes']
    scores = prediction['scores']
    labels = prediction['labels']
    
    mask = scores > conf_thresh
    keep = nms(boxes[mask], scores[mask], iou_threshold=nms_thresh)
    
    return {
        'boxes': boxes[mask][keep].cpu().numpy(),
        'labels': labels[mask][keep].cpu().numpy(),
        'scores': scores[mask][keep].cpu().numpy()
    }
