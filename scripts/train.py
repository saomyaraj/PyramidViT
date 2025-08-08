# scripts/train.py

import torch
from torch.utils.data import DataLoader
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tqdm import tqdm
import logging

import sys
sys.path.append('e:/version_2/vit_detector')
import config
from scripts.dataset import ViTDetDataset, SimpleTransform
from scripts.model import ViTDetector
from scripts.loss import HungarianMatcher, SetCriterion
from scripts.utils import collate_fn
from scripts.engine import train_one_epoch, evaluate
from scripts.metrics import MetricsCalculator

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def plot_training_curves(train_losses, val_losses, val_maps, output_dir):
    """Plot training curves similar to YOLO"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # mAP curve
    ax2.plot(epochs, val_maps, 'g-', label='mAP@0.5', linewidth=2)
    ax2.set_title('Mean Average Precision')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP@0.5')
    ax2.legend()
    ax2.grid(True)
    
    # Model info
    ax3.text(0.5, 0.5, f'Model Architecture:\\nViT Detector\\nBackbone: {config.BACKBONE}', 
             ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    ax3.set_title('Model Info')
    ax3.axis('off')
    
    # Dataset info
    ax4.text(0.5, 0.5, f'Dataset: {config.CURRENT_DATASET}\\nClasses: {config.CLASS_NAMES}\\nNum Classes: {config.NUM_CLASSES}', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Dataset Info')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_training_summary(output_dir, best_map, train_losses, val_losses, val_maps, total_time):
    """Save comprehensive training summary"""
    summary = {
        'dataset': config.CURRENT_DATASET,
        'dataset_path': config.DATA_PATH,
        'class_names': config.CLASS_NAMES,
        'num_classes': config.NUM_CLASSES,
        'model_config': {
            'backbone': config.BACKBONE,
            'num_queries': config.NUM_QUERIES,
            'hidden_dim': config.HIDDEN_DIM,
            'num_encoder_layers': config.NUM_ENCODER_LAYERS,
            'num_decoder_layers': config.NUM_DECODER_LAYERS,
        },
        'training_config': {
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LR,
            'backbone_lr': config.LR_BACKBONE,
            'weight_decay': config.WEIGHT_DECAY,
        },
        'results': {
            'best_map': best_map,
            'final_train_loss': train_losses[-1] if train_losses else 0,
            'final_val_loss': val_losses[-1] if val_losses else 0,
            'training_time_hours': total_time / 3600,
        },
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_maps': val_maps,
        }
    }
    
    with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    print(f"Training on dataset: {config.CURRENT_DATASET}")
    print(f"Dataset path: {config.DATA_PATH}")
    print(f"Classes: {config.CLASS_NAMES}")
    
    # --- Setup ---
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.SAVE_DIR, f"{config.CURRENT_DATASET}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting training on {config.CURRENT_DATASET} dataset")
    logger.info(f"Output directory: {output_dir}")

    # --- Datasets and DataLoaders ---
    transform_train = SimpleTransform(normalize=True)
    transform_val = SimpleTransform(normalize=True)

    dataset_train = ViTDetDataset(root_dir=config.DATA_PATH, split='train', transforms=transform_train)
    dataset_val = ViTDetDataset(root_dir=config.DATA_PATH, split='val', transforms=transform_val)

    logger.info(f"Training with {len(dataset_train)} images.")
    logger.info(f"Validating with {len(dataset_val)} images.")

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS
    )

    # --- Model Setup ---
    model = ViTDetector(
        backbone_name=config.BACKBONE,
        num_classes=config.NUM_CLASSES,
        num_queries=config.NUM_QUERIES,
        hidden_dim=config.HIDDEN_DIM,
        nheads=config.NHEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS
    )
    model.to(device)

    # --- Loss and Optimizer Setup ---
    matcher = HungarianMatcher(
        cost_class=config.CLS_LOSS_COEF,
        cost_bbox=config.BBOX_LOSS_COEF,
        cost_giou=config.GIOU_LOSS_COEF
    )
    
    weight_dict = {'loss_ce': config.CLS_LOSS_COEF, 'loss_bbox': config.BBOX_LOSS_COEF, 'loss_giou': config.GIOU_LOSS_COEF}
    losses = ['labels', 'boxes', 'cardinality']
    
    criterion = SetCriterion(
        num_classes=config.NUM_CLASSES,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=config.EOS_COEF,
        losses=losses
    )
    criterion.to(device)

    # Setup optimizer with different learning rates for backbone and head
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": config.LR_BACKBONE},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.LR_DROP)

    # --- Training Loop ---
    logger.info("Starting training...")
    start_time = datetime.now()
    
    train_losses = []
    val_losses = []
    val_maps = []
    best_map = 0.0
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator(num_classes=config.NUM_CLASSES)
    
    for epoch in range(config.EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{config.EPOCHS}")
        
        # Training
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, config.CLIP_MAX_NORM)
        lr_scheduler.step()
        
        # Validation
        val_stats = evaluate(model, criterion, data_loader_val, device)
        
        # Calculate detailed metrics
        metrics_calc.reset()
        model.eval()
        with torch.no_grad():
            for samples, targets in tqdm(data_loader_val, desc="Computing metrics"):
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                outputs = model(samples)
                # Convert outputs to evaluation format
                predictions = []
                for i, output in enumerate(outputs):
                    # Get top predictions based on confidence
                    scores = output['pred_logits'].softmax(-1)[:, 0]  # Assuming single class
                    top_indices = scores > 0.5  # Confidence threshold
                    
                    pred = {
                        'boxes': output['pred_boxes'][top_indices],
                        'scores': scores[top_indices],
                        'labels': torch.zeros(top_indices.sum(), dtype=torch.long, device=device)
                    }
                    predictions.append(pred)
                
                metrics_calc.add_batch(predictions, targets)
        
        # Compute metrics
        map_50 = metrics_calc.compute_map(iou_threshold=0.5)
        precision, recall, f1 = metrics_calc.compute_precision_recall_f1()
        
        # Store metrics
        train_losses.append(train_stats['loss'])
        val_losses.append(val_stats['loss'])
        val_maps.append(map_50)
        
        # Log progress
        logger.info(f"Train Loss: {train_stats['loss']:.4f}")
        logger.info(f"Val Loss: {val_stats['loss']:.4f}")
        logger.info(f"mAP@0.5: {map_50:.4f}")
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Save best model
        if map_50 > best_map:
            best_map = map_50
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'map': map_50,
                'config': {
                    'backbone': config.BACKBONE,
                    'num_classes': config.NUM_CLASSES,
                    'num_queries': config.NUM_QUERIES,
                    'hidden_dim': config.HIDDEN_DIM,
                    'class_names': config.CLASS_NAMES,
                }
            }, os.path.join(output_dir, 'best_model.pth'))
            logger.info(f"New best model saved! mAP@0.5: {best_map:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Calculate total training time
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # Generate final metrics and plots
    logger.info("Generating final metrics and plots...")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, val_maps, output_dir)
    
    # Generate precision-recall curves
    metrics_calc.plot_precision_recall_curves(output_dir)
    
    # Generate confusion matrix
    metrics_calc.plot_confusion_matrix(output_dir)
    
    # Save training summary
    save_training_summary(output_dir, best_map, train_losses, val_losses, val_maps, total_time)
    
    logger.info(f"Training completed! Best mAP@0.5: {best_map:.4f}")
    logger.info(f"Total training time: {total_time/3600:.2f} hours")
    logger.info(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    main()
