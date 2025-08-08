# -----------------------------------------------------------------------------
# Config file for the ViT Detector
# -----------------------------------------------------------------------------

import os

# --- Dataset Configuration ---
# Multi-dataset configuration for automatic training
DATASETS = {
    # 'FloW-IMG': {
    #     'path': 'e:/version_2/Datasets/FloW-IMG',
    #     'class_names': ['bottle'],
    #     'num_classes': 1
    # },
    # 'IWHR': {
    #     'path': 'e:/version_2/Datasets/IWHR', 
    #     'class_names': ['litter'],
    #     'num_classes': 1
    # },
    'TUD-GV': {
        'path': 'e:/version_2/Datasets/TUD-GV',
        'class_names': ['litter'], 
        'num_classes': 1
    }
}

# Current dataset configuration - TUD-GV
CURRENT_DATASET = 'TUD-GV'
DATA_PATH = DATASETS[CURRENT_DATASET]['path']
CLASS_NAMES = DATASETS[CURRENT_DATASET]['class_names']
NUM_CLASSES = DATASETS[CURRENT_DATASET]['num_classes']

# Legacy config paths
UNLABELED_DATA_PATH = None # Path to unlabeled frames for MAE, set to None if not used

# --- Model Architecture ---
# Alternative options: 'swin_tiny', 'swin_small', 'pvt_v2_b2', 'swin_base'
BACKBONE = 'pvt_v2_b2' # 'swin_tiny', 'swin_small', 'pvt_v2_b2'
PRETRAINED_BACKBONE_PATH = 'e:/version_2/vit_detector/outputs/mae_vit_backbone.pth' # Path to self-supervised weights
FROZEN_STAGES = -1 # -1 means no frozen stages, 1 means freeze stage 1, etc.

# Stem
CONV_STEM_CHANNELS = [64, 128]

# FPN
FPN_CHANNELS = 256
FPN_SCALES = ['P3', 'P4', 'P5'] # Corresponds to different backbone stages

# Head (DETR-style)    
NUM_QUERIES = 300
HIDDEN_DIM = 256
NHEADS = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

# Refinement Module (DiffusionDet-lite)
NUM_REFINEMENT_STEPS = 2

# --- Training Parameters ---
# General
EPOCHS = 150
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
OPTIMIZER = 'adamw'
LR_SCHEDULER = 'cosine'

# Progressive Unfreezing Schedule
# (epoch, learning_rate, layers_to_unfreeze)
# 'all', 'head', 'fpn', 'backbone_last_2', 'backbone_all'
UNFREEZE_SCHEDULE = [
    {'epoch': 0, 'lr': 1e-3, 'layers': ['head', 'fpn']},
    {'epoch': 10, 'lr': 1e-4, 'layers': ['backbone_last_2']},
    {'epoch': 30, 'lr': 5e-5, 'layers': ['backbone_all']}
]

# Additional Training Parameters
LR = LEARNING_RATE  # Alias for compatibility
# Training parameters with aliases
LR = LEARNING_RATE  # Alias for compatibility
LR_BACKBONE = LEARNING_RATE * 0.1  # Lower learning rate for backbone
LR_DROP = 40  # Learning rate drop epoch
CLIP_MAX_NORM = 0.1  # Gradient clipping

# Loss coefficients (aliases for compatibility)
CLS_LOSS_COEF = 1.0
BBOX_LOSS_COEF = 5.0
GIOU_LOSS_COEF = 2.0

# Loss Weights
LOSS_CE_WEIGHT = 1.0
LOSS_BBOX_WEIGHT = 5.0
LOSS_GIOU_WEIGHT = 2.0
EOS_COEF = 0.1 # Relative weight for 'no object' class

# --- Validation Parameters ---
EVAL_INTERVAL = 1  # Evaluate every N epochs
SAVE_INTERVAL = 10  # Save checkpoint every N epochs

# --- Augmentations ---
# Standard Augs
HFLIP_PROB = 0.5
VFLIP_PROB = 0.0
RESIZE_MIN = 800
RESIZE_MAX = 1333

# Advanced Augs
TOKEN_DROP_PROB = 0.15
DIFFUSION_AUG_MIX_RATIO = 0.5 # Ratio of synthetic to real images per batch

# --- Inference ---
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5
MULTI_SCALE_INFERENCE = True
INFERENCE_SCALES = [800, 1000, 1200] # Short edge sizes for multi-scale test

# --- Hardware ---
DEVICE = 'cuda'
NUM_WORKERS = 0  # Set to 0 for Windows compatibility (multiprocessing issues)

# --- Logging ---
LOG_DIR = 'e:/version_2/vit_detector/outputs/logs'
SAVE_DIR = 'e:/version_2/vit_detector/outputs/checkpoints'
LOG_INTERVAL = 25 # Log every 50 iterations

# Multi-dataset output directories
MULTI_DATASET_OUTPUT_BASE = 'e:/version_2/vit_detector/outputs'

# --- Directory Settings ---
import os
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
