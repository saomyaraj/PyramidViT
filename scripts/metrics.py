# scripts/metrics.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import json
import os
from typing import Dict, List, Tuple
from collections import defaultdict

import sys
sys.path.append('e:/version_2/vit_detector')
import config

class MetricsCalculator:
    """
    Comprehensive metrics calculator for object detection evaluation.
    Compatible with YOLO-style evaluation for comparison.
    """
    
    def __init__(self, num_classes: int, iou_thresholds: List[float] = None):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.predictions = []
        self.ground_truths = []
        self.confidences = []
        
    def add_batch(self, predictions: List[Dict], targets: List[Dict]):
        """
        Add a batch of predictions and targets for evaluation
        
        Args:
            predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
            targets: List of target dicts with 'boxes', 'labels'
        """
        self.predictions.extend(predictions)
        self.ground_truths.extend(targets)
    
    def calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Calculate IoU between two boxes in format [cx, cy, w, h] (normalized)"""
        # Convert from center format to corner format
        def cxcywh_to_xyxy(box):
            cx, cy, w, h = box
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            return torch.stack([x1, y1, x2, y2])
        
        box1_xyxy = cxcywh_to_xyxy(box1)
        box2_xyxy = cxcywh_to_xyxy(box2)
        
        # Intersection
        x1 = torch.max(box1_xyxy[0], box2_xyxy[0])
        y1 = torch.max(box1_xyxy[1], box2_xyxy[1])
        x2 = torch.min(box1_xyxy[2], box2_xyxy[2])
        y2 = torch.min(box1_xyxy[3], box2_xyxy[3])
        
        # Check if there's an intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate areas
        area1 = box1[2] * box1[3]  # w * h
        area2 = box2[2] * box2[3]  # w * h
        
        # Union
        union = area1 + area2 - intersection
        
        # IoU
        iou = intersection / (union + 1e-16)
        return iou.item() if torch.is_tensor(iou) else iou
    
    def compute_map(self, iou_threshold: float = 0.5) -> float:
        """
        Compute mean Average Precision (mAP) at specified IoU threshold
        
        Args:
            iou_threshold: IoU threshold for determining true positives
            
        Returns:
            mAP value
        """
        if not self.predictions or not self.ground_truths:
            return 0.0
        
        total_ap = 0.0
        valid_classes = 0
        
        for class_id in range(self.num_classes):
            ap = self.compute_ap_for_class(class_id, iou_threshold)
            if ap is not None:
                total_ap += ap
                valid_classes += 1
        
        return total_ap / max(valid_classes, 1)
    
    def compute_ap_for_class(self, class_id: int, iou_threshold: float) -> float:
        """Compute AP for a single class"""
        # Collect all predictions and ground truths for this class
        class_predictions = []
        class_gts = []
        
        for i, (pred, gt) in enumerate(zip(self.predictions, self.ground_truths)):
            # Filter predictions for this class
            if 'labels' in pred and len(pred['labels']) > 0:
                class_mask = pred['labels'] == class_id
                if class_mask.sum() > 0:
                    class_pred = {
                        'boxes': pred['boxes'][class_mask],
                        'scores': pred['scores'][class_mask],
                        'image_id': i
                    }
                    class_predictions.extend([
                        {'box': box, 'score': score, 'image_id': i}
                        for box, score in zip(class_pred['boxes'], class_pred['scores'])
                    ])
            
            # Filter ground truths for this class
            if 'labels' in gt and len(gt['labels']) > 0:
                gt_mask = gt['labels'] == class_id
                if gt_mask.sum() > 0:
                    class_gts.extend([
                        {'box': box, 'image_id': i, 'used': False}
                        for box in gt['boxes'][gt_mask]
                    ])
        
        if not class_predictions or not class_gts:
            return None
        
        # Sort predictions by confidence
        class_predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Calculate precision and recall
        tp = torch.zeros(len(class_predictions))
        fp = torch.zeros(len(class_predictions))
        
        for i, pred in enumerate(class_predictions):
            # Find matching ground truths in the same image
            image_gts = [gt for gt in class_gts if gt['image_id'] == pred['image_id']]
            
            if not image_gts:
                fp[i] = 1
                continue
            
            # Calculate IoU with all ground truths in the image
            max_iou = 0
            max_idx = -1
            
            for j, gt in enumerate(image_gts):
                if gt['used']:
                    continue
                
                iou = self.calculate_iou(pred['box'], gt['box'])
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            
            if max_iou >= iou_threshold and max_idx >= 0:
                tp[i] = 1
                image_gts[max_idx]['used'] = True
            else:
                fp[i] = 1
        
        # Calculate cumulative precision and recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        
        recalls = tp_cumsum / len(class_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in torch.linspace(0, 1, 11):
            mask = recalls >= t
            if mask.sum() > 0:
                ap += torch.max(precisions[mask]) / 11
        
        return ap.item()
    
    def compute_precision_recall_f1(self, iou_threshold: float = 0.5) -> Tuple[float, float, float]:
        """
        Compute overall precision, recall, and F1 score
        
        Args:
            iou_threshold: IoU threshold for determining true positives
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        if not self.predictions or not self.ground_truths:
            return 0.0, 0.0, 0.0
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            if len(pred['boxes']) == 0 and len(gt['boxes']) == 0:
                continue
            
            if len(pred['boxes']) == 0:
                total_fn += len(gt['boxes'])
                continue
            
            if len(gt['boxes']) == 0:
                total_fp += len(pred['boxes'])
                continue
            
            # Calculate IoU matrix
            iou_matrix = torch.zeros(len(pred['boxes']), len(gt['boxes']))
            for i, pred_box in enumerate(pred['boxes']):
                for j, gt_box in enumerate(gt['boxes']):
                    iou_matrix[i, j] = self.calculate_iou(pred_box, gt_box)
            
            # Match predictions to ground truths
            matched_gt = set()
            matched_pred = set()
            
            # Sort by confidence and match greedily
            if 'scores' in pred:
                pred_indices = torch.argsort(pred['scores'], descending=True)
            else:
                pred_indices = torch.arange(len(pred['boxes']))
            
            for pred_idx in pred_indices:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx in range(len(gt['boxes'])):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = iou_matrix[pred_idx, gt_idx]
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx >= 0:
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(pred_idx.item())
                    total_tp += 1
            
            # Count false positives and false negatives
            total_fp += len(pred['boxes']) - len(matched_pred)
            total_fn += len(gt['boxes']) - len(matched_gt)
        
        precision = total_tp / (total_tp + total_fp + 1e-16)
        recall = total_tp / (total_tp + total_fn + 1e-16)
        f1 = 2 * precision * recall / (precision + recall + 1e-16)
        
        return precision, recall, f1
    
    def plot_precision_recall_curves(self, output_dir: str):
        """Generate and save precision-recall curves like YOLO"""
        if not self.predictions or not self.ground_truths:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot PR curve for each class
        for class_id in range(min(self.num_classes, 4)):  # Limit to 4 classes for display
            ax = axes[class_id // 2, class_id % 2]
            
            # Collect class-specific data
            class_predictions = []
            class_gts = []
            
            for i, (pred, gt) in enumerate(zip(self.predictions, self.ground_truths)):
                if 'labels' in pred and len(pred['labels']) > 0:
                    class_mask = pred['labels'] == class_id
                    if class_mask.sum() > 0:
                        scores = pred['scores'][class_mask] if 'scores' in pred else torch.ones(class_mask.sum())
                        class_predictions.extend(scores.cpu().numpy())
                
                if 'labels' in gt and len(gt['labels']) > 0:
                    gt_mask = gt['labels'] == class_id
                    class_gts.extend([1] * gt_mask.sum().item())
            
            if class_predictions and class_gts:
                # Create binary labels
                y_true = [1] * len(class_gts) + [0] * (len(class_predictions) - len(class_gts))
                y_scores = class_predictions + [0] * max(0, len(class_gts) - len(class_predictions))
                
                precision, recall, _ = precision_recall_curve(y_true[:len(y_scores)], y_scores)
                ap = average_precision_score(y_true[:len(y_scores)], y_scores)
                
                ax.plot(recall, precision, linewidth=2, label=f'AP = {ap:.3f}')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(f'Class {class_id} PR Curve')
                ax.legend()
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Class {class_id} - No Data')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, output_dir: str, iou_threshold: float = 0.5):
        """Generate and save confusion matrix"""
        if not self.predictions or not self.ground_truths:
            return
        
        # Create confusion matrix
        conf_matrix = np.zeros((self.num_classes + 1, self.num_classes + 1))  # +1 for background
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            if len(gt['boxes']) == 0:
                # True negative case
                if len(pred['boxes']) == 0:
                    conf_matrix[self.num_classes, self.num_classes] += 1
                else:
                    # False positive
                    for pred_label in pred['labels']:
                        conf_matrix[pred_label.item(), self.num_classes] += 1
                continue
            
            if len(pred['boxes']) == 0:
                # False negative
                for gt_label in gt['labels']:
                    conf_matrix[self.num_classes, gt_label.item()] += 1
                continue
            
            # Match predictions to ground truths
            matched_gt = set()
            for i, pred_box in enumerate(pred['boxes']):
                best_iou = 0
                best_gt_idx = -1
                
                for j, gt_box in enumerate(gt['boxes']):
                    if j in matched_gt:
                        continue
                    
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = j
                
                pred_label = pred['labels'][i].item() if i < len(pred['labels']) else 0
                
                if best_gt_idx >= 0:
                    gt_label = gt['labels'][best_gt_idx].item()
                    conf_matrix[pred_label, gt_label] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    # False positive
                    conf_matrix[pred_label, self.num_classes] += 1
            
            # Count unmatched ground truths as false negatives
            for j, gt_label in enumerate(gt['labels']):
                if j not in matched_gt:
                    conf_matrix[self.num_classes, gt_label.item()] += 1
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        
        # Create class names
        class_names = [f'Class {i}' for i in range(self.num_classes)] + ['Background']
        
        sns.heatmap(conf_matrix, annot=True, fmt='.0f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix (IoU > {iou_threshold})')
        plt.xlabel('True Label')
        plt.ylabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_yolo_like_results(self, output_dir: str):
        """Generate YOLO-like results summary"""
        results = {}
        
        # Calculate mAP at different IoU thresholds
        map_50 = self.compute_map(0.5)
        map_75 = self.compute_map(0.75)
        
        # Calculate mAP@0.5:0.95 (average over IoU thresholds)
        map_50_95 = np.mean([self.compute_map(iou) for iou in self.iou_thresholds])
        
        # Calculate precision, recall, F1
        precision, recall, f1 = self.compute_precision_recall_f1()
        
        results = {
            'metrics': {
                'mAP@0.5': map_50,
                'mAP@0.75': map_75,
                'mAP@0.5:0.95': map_50_95,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
            },
            'per_class_metrics': {}
        }
        
        # Per-class metrics
        for class_id in range(self.num_classes):
            class_ap_50 = self.compute_ap_for_class(class_id, 0.5)
            if class_ap_50 is not None:
                results['per_class_metrics'][f'class_{class_id}'] = {
                    'AP@0.5': class_ap_50,
                    'AP@0.75': self.compute_ap_for_class(class_id, 0.75),
                }
        
        # Save results
        with open(os.path.join(output_dir, 'detection_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
        y2 = torch.min(box1[3], box2[3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Areas
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Union
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def convert_cxcywh_to_xyxy(self, boxes: torch.Tensor, img_size: Tuple[int, int] = (800, 800)) -> torch.Tensor:
        """Convert boxes from [cx, cy, w, h] normalized to [x1, y1, x2, y2] absolute"""
        if boxes.numel() == 0:
            return boxes
        
        h, w = img_size
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Convert to absolute coordinates
        cx = cx * w
        cy = cy * h
        bw = bw * w
        bh = bh * h
        
        # Convert to xyxy format
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def calculate_ap_at_iou(self, iou_threshold: float = 0.5) -> Dict[str, float]:
        """Calculate Average Precision at specific IoU threshold"""
        ap_per_class = {}
        
        for class_id in range(self.num_classes):
            # Collect all predictions and ground truths for this class
            all_pred_boxes = []
            all_pred_scores = []
            all_gt_boxes = []
            
            for i, (pred, gt) in enumerate(zip(self.predictions, self.ground_truths)):
                # Filter predictions for this class
                if 'labels' in pred:
                    class_mask = pred['labels'] == class_id
                    if class_mask.any():
                        pred_boxes = self.convert_cxcywh_to_xyxy(pred['boxes'][class_mask])
                        pred_scores = pred['scores'][class_mask] if 'scores' in pred else torch.ones(class_mask.sum())
                        
                        for box, score in zip(pred_boxes, pred_scores):
                            all_pred_boxes.append(box)
                            all_pred_scores.append(score.item())
                
                # Filter ground truths for this class
                if 'labels' in gt:
                    gt_class_mask = gt['labels'] == class_id
                    if gt_class_mask.any():
                        gt_boxes = self.convert_cxcywh_to_xyxy(gt['boxes'][gt_class_mask])
                        for box in gt_boxes:
                            all_gt_boxes.append(box)
            
            if len(all_gt_boxes) == 0:
                ap_per_class[f'class_{class_id}'] = 0.0
                continue
            
            if len(all_pred_boxes) == 0:
                ap_per_class[f'class_{class_id}'] = 0.0
                continue
            
            # Sort predictions by confidence
            sorted_indices = np.argsort(all_pred_scores)[::-1]
            sorted_pred_boxes = [all_pred_boxes[i] for i in sorted_indices]
            sorted_scores = [all_pred_scores[i] for i in sorted_indices]
            
            # Calculate precision and recall
            tp = np.zeros(len(sorted_pred_boxes))
            fp = np.zeros(len(sorted_pred_boxes))
            
            for i, pred_box in enumerate(sorted_pred_boxes):
                best_iou = 0
                for gt_box in all_gt_boxes:
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                
                if best_iou >= iou_threshold:
                    tp[i] = 1
                else:
                    fp[i] = 1
            
            # Calculate cumulative precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / len(all_gt_boxes)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            
            # Calculate AP using the 11-point interpolation
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11
            
            ap_per_class[f'class_{class_id}'] = ap
        
        # Calculate mean AP
        ap_per_class['mAP'] = np.mean(list(ap_per_class.values()))
        return ap_per_class
    
    def calculate_map(self) -> Dict[str, float]:
        """Calculate mAP across all IoU thresholds (mAP@0.5:0.95)"""
        maps = []
        detailed_results = {}
        
        for iou_thresh in self.iou_thresholds:
            ap_results = self.calculate_ap_at_iou(iou_thresh)
            maps.append(ap_results['mAP'])
            detailed_results[f'mAP@{iou_thresh}'] = ap_results['mAP']
        
        detailed_results['mAP@0.5:0.95'] = np.mean(maps)
        detailed_results['mAP@0.5'] = detailed_results.get('mAP@0.5', 0.0)
        
        return detailed_results
    
    def plot_precision_recall_curve(self, save_path: str):
        """Plot precision-recall curves for each class"""
        plt.figure(figsize=(12, 8))
        
        for class_id in range(self.num_classes):
            # Collect predictions and ground truths for this class
            y_true = []
            y_scores = []
            
            for pred, gt in zip(self.predictions, self.ground_truths):
                # Ground truth labels
                if 'labels' in gt:
                    gt_labels = (gt['labels'] == class_id).float()
                    y_true.extend(gt_labels.cpu().numpy())
                
                # Prediction scores
                if 'labels' in pred and 'scores' in pred:
                    pred_mask = pred['labels'] == class_id
                    if pred_mask.any():
                        scores = pred['scores'][pred_mask]
                        y_scores.extend(scores.cpu().numpy())
                    else:
                        y_scores.extend([0.0] * len(gt_labels))
                else:
                    y_scores.extend([0.0] * len(gt_labels))
            
            if len(y_true) > 0 and sum(y_true) > 0:
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                ap = average_precision_score(y_true, y_scores)
                plt.plot(recall, precision, label=f'Class {class_id} (AP={ap:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def calculate_f1_scores(self, confidence_threshold: float = 0.5) -> Dict[str, float]:
        """Calculate F1 scores at given confidence threshold"""
        f1_scores = {}
        
        for class_id in range(self.num_classes):
            y_true = []
            y_pred = []
            
            for pred, gt in zip(self.predictions, self.ground_truths):
                # Ground truth
                if 'labels' in gt:
                    gt_mask = gt['labels'] == class_id
                    y_true.extend(gt_mask.cpu().numpy())
                
                # Predictions
                if 'labels' in pred and 'scores' in pred:
                    pred_mask = (pred['labels'] == class_id) & (pred['scores'] > confidence_threshold)
                    y_pred.extend(pred_mask.cpu().numpy())
                else:
                    y_pred.extend([False] * len(gt_mask))
            
            if len(y_true) > 0:
                f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
                f1_scores[f'class_{class_id}'] = f1
        
        f1_scores['macro_f1'] = np.mean(list(f1_scores.values()))
        return f1_scores
    
    def save_metrics_to_json(self, metrics: Dict, save_path: str):
        """Save metrics to JSON file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    def generate_comprehensive_report(self, output_dir: str, dataset_name: str = ""):
        """Generate comprehensive evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate all metrics
        map_results = self.calculate_map()
        ap_at_50 = self.calculate_ap_at_iou(0.5)
        f1_results = self.calculate_f1_scores()
        
        # Combine all metrics
        comprehensive_metrics = {
            'dataset': dataset_name,
            'mAP_results': map_results,
            'AP_at_50': ap_at_50,
            'F1_scores': f1_results,
            'evaluation_summary': {
                'mAP@0.5': map_results.get('mAP@0.5', 0.0),
                'mAP@0.5:0.95': map_results.get('mAP@0.5:0.95', 0.0),
                'macro_F1': f1_results.get('macro_f1', 0.0)
            }
        }
        
        # Save metrics
        metrics_path = os.path.join(output_dir, f'{dataset_name}_metrics.json')
        self.save_metrics_to_json(comprehensive_metrics, metrics_path)
        
        # Plot PR curves
        pr_curve_path = os.path.join(output_dir, f'{dataset_name}_pr_curves.png')
        self.plot_precision_recall_curve(pr_curve_path)
        
        return comprehensive_metrics

def evaluate_model_predictions(model_outputs: List[Dict], targets: List[Dict], 
                             num_classes: int, output_dir: str, dataset_name: str = "") -> Dict:
    """
    Evaluate model predictions and generate comprehensive metrics report
    
    Args:
        model_outputs: List of model prediction dictionaries
        targets: List of ground truth dictionaries
        num_classes: Number of classes in the dataset
        output_dir: Directory to save evaluation results
        dataset_name: Name of the dataset for labeling
    
    Returns:
        Dictionary containing comprehensive evaluation metrics
    """
    calculator = MetricsCalculator(num_classes)
    calculator.add_batch(model_outputs, targets)
    return calculator.generate_comprehensive_report(output_dir, dataset_name)
