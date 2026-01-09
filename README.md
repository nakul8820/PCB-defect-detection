
# PCB Defect Detection with Faster R-CNN

A PyTorch Lightning-based implementation for automated PCB defect detection using Faster R-CNN with ResNet50-FPN backbone.

## WandB Report:
Link :https://wandb.ai/nakupatel-indus-university/PCB_Inspection/reports/PCB-Defect-Detection-with-Faster-R-CNN--VmlldzoxNTU4MTc1Nw?accessToken=9neui0ovr4r4ed9w23v87vuhjpbtqf7pi9nmzep0zj7uffdfwtjkwqcdqklarsl7

## 🎯 Project Overview

This project implements a deep learning pipeline for detecting six types of PCB defects:
- Missing hole
- Mouse bite
- Open circuit
- Short
- Spur
- Spurious copper

## 📊 Key Features

- **Model**: Faster R-CNN with ResNet50-FPN backbone
- **Framework**: PyTorch Lightning for clean, scalable training
- **Augmentation**: Albumentations for robust data augmentation
- **Logging**: Weights & Biases (W&B) for experiment tracking
- **Optimizations**:
  - Custom anchor sizes for small defect detection
  - AdamW optimizer with OneCycleLR scheduling
  - Gradient accumulation for larger effective batch size
  - Early stopping and model checkpointing

## 📁 Project Structure

```
pcb-defect-detection/
├── src/
│   ├── dataset.py          # Dataset and data loading
│   ├── model.py            # Model architecture
│   ├── train.py            # Training pipeline
│   └── inference.py        # Inference and visualization
├── config/
│   └── config.yaml         # Configuration file
├── requirements.txt        # Dependencies
├── README.md              # This file
└── .gitignore            # Git ignore file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Weights & Biases account (for logging)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pcb-defect-detection.git
cd pcb-defect-detection

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

1. Download the PCB defects dataset
2. Organize it with the following structure:
```
PCB_DATASET/
├── images/
│   └── *.jpg
└── Annotations/
    └── *.xml
```

### Training

```bash
python src/train.py
```


## 🔧 Configuration

Key hyperparameters can be modified in the training script:
- `lr`: Learning rate (default: 0.0005)
- `max_epochs`: Maximum training epochs (default: 50)
- `batch_size`: Batch size (default: 16)
- `img_size`: Input image size (default: 800x800)

## 📊 Monitoring

Training metrics are logged to Weights & Biases:
- Training/validation loss
- Mean Average Precision (mAP)
- Per-class performance
- Sample predictions with bounding boxes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch Lightning for the training framework
- Torchvision for pre-trained models
- Albumentations for data augmentation
- Weights & Biases for experiment tracking

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/nakul8820/pcb-defect-detection](https://github.com/nakul8820/pcb-defect-detection)
