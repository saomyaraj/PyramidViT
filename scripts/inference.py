# scripts/inference.py

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import numpy as np
import os
import json
import cv2
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

import sys
sys.path.append('e:/version_2/vit_detector')
import config
from scripts.model import ViTDetector
from scripts.utils import box_cxcywh_to_xyxy

class ViTDetectorInference:
    """
    YOLO-style inference for ViT Detector
    Provides similar functionality to YOLO for object detection with bounding boxes
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, device: str = 'cuda'):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.class_names = config.CLASS_NAMES
        self.colors = self._generate_colors()
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _generate_colors(self):
        """Generate random colors for each class"""
        np.random.seed(42)  # For consistent colors
        colors = []
        for _ in range(len(self.class_names)):
            colors.append(tuple(np.random.randint(0, 255, 3).tolist()))
        return colors
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        model = ViTDetector(
            backbone_name=config.BACKBONE,
            num_classes=config.NUM_CLASSES,
            num_queries=config.NUM_QUERIES,
            hidden_dim=config.HIDDEN_DIM,
            nheads=config.NHEADS,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess image for inference"""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Transform image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        return input_tensor, original_size
    
    def postprocess_predictions(self, outputs: Dict, original_size: Tuple[int, int], 
                              confidence_threshold: float = None) -> List[Dict]:
        """
        Convert model outputs to YOLO-style detection results
        
        Returns:
            List of detections with format:
            [{'bbox': [x1, y1, x2, y2], 'confidence': score, 'class': class_name, 'class_id': id}]
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        # outputs is now a dict directly
        pred_logits = outputs['pred_logits'][0]  # Remove batch dimension
        pred_boxes = outputs['pred_boxes'][0]    # Remove batch dimension
        
        # Get probabilities and filter by confidence
        probs = F.softmax(pred_logits, dim=-1)
        max_probs, classes = probs[:, :-1].max(dim=-1)  # Exclude background class
        
        # Filter by confidence threshold
        keep = max_probs > confidence_threshold
        
        filtered_boxes = pred_boxes[keep]
        filtered_probs = max_probs[keep]
        filtered_classes = classes[keep]
        
        # Convert to absolute coordinates
        orig_w, orig_h = original_size
        detections = []
        
        for box, prob, cls in zip(filtered_boxes, filtered_probs, filtered_classes):
            # Convert from cxcywh to xyxy format
            cx, cy, w, h = box.cpu().numpy()
            
            # Convert to absolute coordinates
            x1 = (cx - w/2) * orig_w
            y1 = (cy - h/2) * orig_h
            x2 = (cx + w/2) * orig_w
            y2 = (cy + h/2) * orig_h
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))
            
            detection = {
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(prob.cpu().numpy()),
                'class': self.class_names[cls.item()] if cls.item() < len(self.class_names) else 'unknown',
                'class_id': int(cls.item())
            }
            detections.append(detection)
        
        return detections
    
    def predict(self, image_path: str, save_path: str = None) -> List[Dict]:
        """
        Run inference on a single image
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save annotated image
            
        Returns:
            List of detections
        """
        with torch.no_grad():
            # Preprocess
            input_tensor, original_size = self.preprocess_image(image_path)
            
            # Run inference
            outputs = self.model(input_tensor)
            
            # Postprocess
            detections = self.postprocess_predictions(outputs, original_size)
            
            # Save annotated image if requested
            if save_path:
                self.visualize_detections(image_path, detections, save_path)
            
            return detections
    
    def visualize_detections(self, image_path: str, detections: List[Dict], save_path: str):
        """Visualize detections on image (YOLO-style)"""
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            class_id = detection['class_id']
            
            # Get color for this class
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            draw.rectangle(bbox, outline=color, width=3)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size for background
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            # Draw background for text
            text_bg = [bbox[0], bbox[1] - text_height - 5, 
                      bbox[0] + text_width + 10, bbox[1]]
            draw.rectangle(text_bg, fill=color)
            
            # Draw text
            draw.text((bbox[0] + 5, bbox[1] - text_height - 2), label, 
                     fill='white', font=font)
        
        # Save image
        image.save(save_path)
        print(f"Annotated image saved to: {save_path}")
    
    def predict_batch(self, image_paths: List[str], batch_size: int = 4) -> List[List[Dict]]:
        """Run inference on a batch of images"""
        all_detections = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_detections = []
            
            for path in batch_paths:
                detections = self.predict(path)
                batch_detections.append(detections)
            
            all_detections.extend(batch_detections)
        
        return all_detections
    
    def export_detections_to_json(self, detections: List[Dict], image_path: str, save_path: str):
        """Export detections in JSON format (similar to YOLO results)"""
        result = {
            'image': image_path,
            'detections': detections,
            'model_info': {
                'backbone': config.BACKBONE,
                'num_classes': config.NUM_CLASSES,
                'confidence_threshold': self.confidence_threshold
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Detection results saved to: {save_path}")

# Legacy class for backward compatibility
class ViTInference(ViTDetectorInference):
    def __init__(self, model_path: str, device: str = 'cuda', confidence_threshold: float = 0.5):
        super().__init__(model_path, confidence_threshold, device)

def batch_inference(input_dir: str, output_dir: str, model_path: str, confidence_threshold: float = 0.5):
    """
    Run inference on a directory of images
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save results
        model_path: Path to trained model
        confidence_threshold: Confidence threshold for detections
    """
    detector = ViTDetectorInference(model_path, confidence_threshold)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all images
    image_files = [f for f in input_path.iterdir() 
                  if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    all_results = []
    
    for image_file in image_files:
        print(f"Processing: {image_file.name}")
        
        # Run inference
        output_image_path = output_path / f"annotated_{image_file.name}"
        detections = detector.predict(str(image_file), str(output_image_path))
        
        # Save JSON results
        json_path = output_path / f"{image_file.stem}_detections.json"
        detector.export_detections_to_json(detections, str(image_file), str(json_path))
        
        all_results.append({
            'image': str(image_file),
            'detections': detections,
            'detection_count': len(detections)
        })
    
    # Save summary
    summary_path = output_path / "inference_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'total_images': len(image_files),
            'total_detections': sum(len(r['detections']) for r in all_results),
            'results': all_results
        }, f, indent=2)
    
    print(f"Batch inference complete! Results saved to {output_dir}")

def main():
    """Example usage of the inference system"""
    parser = argparse.ArgumentParser(description='ViT Detector Inference - YOLO-style interface')
    parser.add_argument('--model', '-m', required=True, help='Path to trained model')
    parser.add_argument('--input', '-i', required=True, help='Input image or directory')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', '-d', default='cuda', help='Device to run on')
    
    args = parser.parse_args()
    
    # Initialize inference
    detector = ViTDetectorInference(args.model, args.confidence, args.device)
    
    # Check if input is file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image inference
        print(f"Running inference on: {args.input}")
        
        # Set output path for annotated image
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"annotated_{input_path.name}"
        else:
            save_path = f"annotated_{input_path.name}"
        
        detections = detector.predict(args.input, str(save_path))
        
        print(f"Found {len(detections)} objects:")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['class']}: {det['confidence']:.3f} at {det['bbox']}")
        
        # Save JSON result
        if args.output:
            json_path = output_dir / f"{input_path.stem}_detections.json"
            detector.export_detections_to_json(detections, args.input, str(json_path))
    
    elif input_path.is_dir():
        # Batch inference
        output_dir = args.output or str(input_path / "results")
        print(f"Running batch inference on directory: {args.input}")
        batch_inference(args.input, output_dir, args.model, args.confidence)
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")

if __name__ == '__main__':
    main()
