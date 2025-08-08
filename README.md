# PyramidViT Detector - Hierarchical Hybrid Vision Transformer with Self-Supervised Pre-training for Object Detection

*A comprehensive object detection framework using Vision Transformers with YOLO-style functionality for training, evaluation, and inference.*

## Overview

ViT Detector is a modern object detection framework that combines Vision Transformer with the simplicity and effectiveness of YOLO-style detection. It provides a complete pipeline for training, evaluation, and inference with comprehensive metrics and visualization tools.

### Key Features

| **Training** | **Evaluation** | **Inference** | **Architecture** |
|:---:|:---:|:---:|:---:|
| YOLO-style training loops | Comprehensive mAP metrics | Command-line interface | PVT-v2-B2 backbone |
| Real-time progress tracking | Precision-Recall curves | Batch processing | DETR-style detection head |
| Automatic checkpointing | Confusion matrices | Multiple image formats | Hungarian matching |
| Training curves visualization | COCO-style evaluation | JSON results export | Progressive unfreezing |

## Architecture

The model combines modern Vision Transformer techniques with proven detection methodologies:

- **Backbone**: PVT-v2-B2 (Pyramid Vision Transformer v2) for hierarchical feature extraction
- **Neck**: Feature Pyramid Network (FPN) for multi-scale feature fusion  
- **Head**: DETR-style transformer decoder with object queries
- **Matching**: Hungarian algorithm for optimal prediction-to-ground-truth assignment
- **Loss**: Multi-task loss with classification, bounding box regression, and GIoU

## Project Structure

```
vit_detector/
├── 📄 config.py                 # Central configuration file
├── 📄 main.py                   # Complete end-to-end training script
├── 📁 scripts/                  # Core framework modules
│   ├── dataset.py               # YOLO-format dataset handling
│   ├── model.py                 # ViT Detector architecture
│   ├── train.py                 # Advanced training pipeline
│   ├── engine.py                # Training/validation engines
│   ├── loss.py                  # Loss functions and Hungarian matcher
│   ├── metrics.py               # Evaluation metrics (mAP, PR, etc.)
│   ├── inference.py             # YOLO-style inference interface
│   ├── mae_pretrain.py          # Masked Auto-Encoder pretraining
│   └── utils.py                 # Utility functions
└── 📁 outputs/                  # Training artifacts
    ├── 📁 checkpoints/          # Model checkpoints by dataset & timestamp
    │   └── 📁 TUD-GV_YYYYMMDD_HHMMSS/
    │       ├── best_model.pth           # Best performing model
    │       ├── training_curves.png      # Loss and mAP progression
    │       ├── precision_recall_curves.png # Per-class PR curves
    │       ├── confusion_matrix.png     # Classification confusion matrix
    │       ├── training_summary.json    # Complete training results
    │       └── training.log             # Detailed training logs
    └── 📁 logs/                 # Additional logging outputs
```

## Quick Start

### Installation Instructions

```bash
# Clone the repository
git clone https://github.com/saomyaraj/PyramidViT.git
cd PyramidViT

# Install dependencies
pip install torch torchvision timm opencv-python matplotlib seaborn scikit-learn tqdm pillow albumentations
```

### Dataset Preparation

Organize your dataset in YOLO format:

```
dataset/
├── images/
│   ├── train/          # Training images
│   └── val/            # Validation images
└── labels/
    ├── train/          # Training labels (.txt files)
    └── val/            # Validation labels (.txt files)
```

### Configuration

Update `config.py` with your dataset information:

```python
# Dataset Configuration
CURRENT_DATASET = 'DATASET'
DATA_PATH = '/path/to/dataset'
CLASS_NAMES = ['class1', 'class2', 'class3']  # Class names
NUM_CLASSES = len(CLASS_NAMES)

# Training Parameters
EPOCHS = 150
BATCH_SIZE = 4          # Adjust based on GPU memory
LEARNING_RATE = 1e-4
BACKBONE = 'pvt_v2_b2'  # pvt_v2_b2, swin_tiny, swin_small
```

### Training

#### Simple Training (Recommended for beginners)

```bash
# All-in-one training script
python main.py
```

#### Advanced Training (Full pipeline)

```bash
# Advanced training with comprehensive features
python scripts/train.py
```

### Inference

#### Single Image Detection

```bash
python scripts/inference.py \
    --model outputs/checkpoints/DATASET_TIMESTAMP/best_model.pth \
    --input path/to/image.jpg \
    --confidence 0.5 \
    --save \
    --show
```

#### Batch Processing

```bash
python scripts/inference.py \
    --model best_model.pth \
    --input path/to/images/ \
    --output results/ \
    --confidence 0.5 \
    --save
```

#### Programmatic Inference

```python
from scripts.inference import ViTDetectorInference

detector = ViTDetectorInference('best_model.pth', confidence_threshold=0.5)
results = detector.predict_image('image.jpg')
```

### Generated Artifacts

| File | Description | Use Case |
|------|-------------|----------|
| `best_model.pth` | Best performing model weights | Production inference |
| `training_curves.png` | Loss and mAP progression plots | Training analysis |
| `precision_recall_curves.png` | Per-class PR curves | Performance evaluation |
| `confusion_matrix.png` | Classification confusion matrix | Error analysis |
| `training_summary.json` | Complete training statistics | Experiment tracking |
| `checkpoint_epoch_N.pth` | Periodic checkpoints | Resume training |

## Features

### Progressive Unfreezing

Implements sophisticated training strategies:

```python
UNFREEZE_SCHEDULE = [
    {'epoch': 0, 'lr': 1e-3, 'layers': ['head', 'fpn']},      # Start with head
    {'epoch': 10, 'lr': 1e-4, 'layers': ['backbone_last_2']}, # Add backbone layers  
    {'epoch': 30, 'lr': 5e-5, 'layers': ['backbone_all']}     # Full model training
]
```

### Multi-Dataset Support

Configure and train on multiple datasets:

```python
DATASETS = {
    'TUD-GV': {'path': '/path/to/tudgv', 'class_names': ['litter'], 'num_classes': 1},
    'FloW-IMG': {'path': '/path/to/flow', 'class_names': ['bottle'], 'num_classes': 1},
    'Custom': {'path': '/path/to/custom', 'class_names': ['obj1', 'obj2'], 'num_classes': 2}
}
```

### Self-Supervised Pretraining

Optional MAE (Masked Auto-Encoder) pretraining for improved performance:

```bash
python scripts/mae_pretrain.py --dataset unlabeled_images/ --epochs 100
```

### Model Export & Optimization

```python
# Export to TorchScript for production
model = ViTDetector.load_from_checkpoint('best_model.pth')
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model_optimized.pt')
```

## Requirements & Dependencies

### System Requirements

- **Python**: 3.8+ (3.9+ recommended)
- **GPU**: CUDA-compatible GPU with 6GB+ VRAM
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space for datasets and outputs

### Installation

```bash
# Install PyTorch (check https://pytorch.org for your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install timm opencv-python matplotlib seaborn scikit-learn tqdm pillow albumentations
```

### Performance Tips

- **Batch Size**: Use largest batch size that fits in memory (4-16 recommended)
- **Mixed Precision**: Enable AMP for faster training with `--amp`
- **Data Loading**: Use SSD storage for datasets to reduce I/O bottleneck
- **Backbone**: Try different backbones (`swin_tiny` for speed, `pvt_v2_b2` for accuracy)

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`  
5. **Open** a Pull Request

### Development Setup

```bash
git clone https://github.com/saomyaraj/PyramidViT.git
cd PyramidViT
pip install -e .  # Editable install
pre-commit install  # Code formatting hooks
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **PVT-v2**: [Pyramid Vision Transformer v2](https://github.com/whai362/PVT) for the excellent backbone
- **DETR**: [Detection Transformer](https://github.com/facebookresearch/detr) for the detection methodology
- **timm**: [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) for model implementations
- **YOLO**: For inspiration in creating user-friendly detection interfaces

---
