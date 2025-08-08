# ==============================================================================
# 0. SETUP AND IMPORTS
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F

import os
import cv2
import numpy as np
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json

# Third-party libraries
import timm
from scipy.optimize import linear_sum_assignment
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    # --- Dataset Paths ---
    # IMPORTANT: Change this to the root directory of your TUD-GV dataset
    DATASET_PATH = "/content/TUD-GV"
    
    # --- Model Hyperparameters ---
    IMG_SIZE = 800
    BACKBONE = 'swin_tiny_patch4_window7_224' # Powerful and efficient backbone
    NUM_CLASSES = 1 # Only "litter"
    NUM_QUERIES = 100 # Max number of objects to detect per image
    HIDDEN_DIM = 256 # Dimension of the transformer
    NHEADS = 8 # Number of attention heads
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6 # Includes refinement stages
    
    # --- Training Hyperparameters ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 50 # Adjust as needed
    LR = 1e-4
    LR_BACKBONE = 1e-5
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 4 # Adjust based on your GPU memory
    NUM_WORKERS = 2
    CLIP_MAX_NORM = 0.1 # Gradient clipping

    # --- Loss Coefficients ---
    CLS_LOSS_COEF = 2
    BBOX_LOSS_COEF = 5
    GIOU_LOSS_COEF = 2
    EOS_COEF = 0.1 # Relative weight for the "no object" class

    # --- Output/Logging ---
    OUTPUT_DIR = "outputs"

# Create output directory
Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)

# ==============================================================================
# 2. DATASET AND DATALOADER
# ==============================================================================
# Bounding box utility functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=1)

# Albumentations for powerful data augmentation. This simulates the "domain randomization" part of our proposal
def get_transforms(train=True):
    if train:
        return A.Compose([
            A.RandomSizedBBoxSafeCrop(width=Config.IMG_SIZE, height=Config.IMG_SIZE, erosion_rate=0.2),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.MotionBlur(p=0.2),
            A.CoarseDropout(max_holes=8, max_height=Config.IMG_SIZE//20, max_width=Config.IMG_SIZE//20, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    else:
        return A.Compose([
            A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

class TUDGVDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.img_dir = self.root_dir / 'images' / split
        self.label_dir = self.root_dir / 'labels' / split
        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.img_dir / img_name
        label_path = self.label_dir / f"{img_name.split('.')[0]}.txt"

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([x_center, y_center, width, height])
                    labels.append(int(class_id))
        
        target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32), 
                  'labels': torch.as_tensor(labels, dtype=torch.int64)}
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=target['boxes'], class_labels=target['labels'])
            image = transformed['image']
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32).reshape(-1, 4)
            target['labels'] = torch.as_tensor(transformed['class_labels'], dtype=torch.int64)

        # Convert YOLO to xyxy format for GIoU loss calculation
        if len(target['boxes']) > 0:
            target['boxes'] = box_xyxy_to_cxcywh(box_cxcywh_to_xyxy(target['boxes'])) # yolo -> xyxy -> cxcywh
        else: # Handle images with no objects
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros(0, dtype=torch.int64)

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ==============================================================================
# 3. MODEL ARCHITECTURE (Hybrid Conv-ViT with DETR Head)
# ==============================================================================
class HybridDetector(nn.Module):
    def __init__(self, num_classes, num_queries, hidden_dim, nheads, num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.num_queries = num_queries
        
        # --- Backbone (Swin Transformer) ---
        # Using timm to get a powerful pre-trained backbone. This replaces the ConvStem+ViT. features_only=True gives us intermediate feature maps, perfect for FPN.
        self.backbone = timm.create_model(Config.BACKBONE, pretrained=True, features_only=True, out_indices=(1, 2, 3))
        num_channels = self.backbone.feature_info.channels()[-1]

        # --- Transformer ---
        self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        
        # --- Prediction Heads ---
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1) # +1 for "no object"
        self.bbox_embed = nn.Linear(hidden_dim, 4) # cx, cy, w, h
        
        # --- Query Embeddings ---
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, x):
        # Backbone
        features = self.backbone(x)
        src = features[-1] # Use the last feature map from the backbone
        
        # Project backbone output to the transformer's dimension
        proj_src = self.input_proj(src)
        
        # Reshape for transformer
        h, w = proj_src.shape[-2:]
        bs = proj_src.shape[0]
        proj_src = proj_src.flatten(2).permute(2, 0, 1) # (h*w, bs, hidden_dim)

        # Positional encoding (learned) - simple version
        pos_embed = nn.Parameter(torch.randn(h * w, 1, Config.HIDDEN_DIM)).to(proj_src.device)
        
        # Get query embeddings
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) # (num_queries, bs, hidden_dim)
        
        # Transformer
        # The decoder output is what we use for prediction. It's a form of iterative refinement.
        hs = self.transformer(proj_src + pos_embed, query_embed).transpose(1, 2)
        
        # Prediction
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid() # Use sigmoid for normalized coords
        
        # hs shape: (num_decoder_layers, bs, num_queries, hidden_dim)
        # We take the output from the last decoder layer
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        
        return out

# ==============================================================================
# 4. LOSS FUNCTION (Bipartite Matching)
# ==============================================================================
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_upleft = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_botright = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    inter_wh = (inter_botright - inter_upleft).clamp(min=0)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]
    
    union_area = boxes1_area[:, None] + boxes2_area - inter_area
    
    iou = inter_area / union_area
    
    enclose_upleft = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enclose_botright = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enclose_wh = (enclose_botright - enclose_upleft).clamp(min=0)
    enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1]
    
    giou = iou - (enclose_area - union_area) / enclose_area
    
    return giou

class SetCriterion(nn.Module):
    """
    This class computes the loss for DETR.
    The process happens in two steps:
    1) we compute hungarian assignment between ground truth boxes and the outputs of the model
    2) we supervise each pair of matched ground-truth / prediction
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = nn.functional.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = nn.functional.l1_loss(src_boxes, target_boxes, reduction='none')
        
        # GIoU Loss
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))

        return {'loss_bbox': loss_bbox.sum() / len(target_boxes), 
                'loss_giou': loss_giou.sum() / len(target_boxes)}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        # Compute the matching between predictions and targets
        indices = self.matcher(outputs, targets)

        # Compute all the requested losses
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices))
        
        return losses

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]
        
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

# ==============================================================================
# 5. TRAINING AND EVALUATION
# ==============================================================================
def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch):
    model.train()
    criterion.train()
    
    total_loss = 0
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1} Training")
    
    for images, targets in pbar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(torch.stack(images))
        
        losses_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(losses_dict[k] * weight_dict[k] for k in losses_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()
        if Config.CLIP_MAX_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_MAX_NORM)
        optimizer.step()
        
        total_loss += losses.item()
        pbar.set_postfix(loss=losses.item())
        
    return total_loss / len(data_loader)

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, coco_gt):
    model.eval()
    criterion.eval()

    coco_results = []
    img_ids = coco_gt.getImgIds()

    pbar = tqdm(data_loader, desc="Evaluating")
    for images, targets in pbar:
        images = list(img.to(device) for img in images)
        
        outputs = model(torch.stack(images))
        
        orig_target_sizes = torch.stack([torch.tensor([Config.IMG_SIZE, Config.IMG_SIZE], device=device) for i in range(len(images))])
        
        results = post_process(outputs, orig_target_sizes)
        
        for i, res in enumerate(results):
            # Assuming image IDs in val loader are sequential. A more robust solution would pass image_ids from the dataset
            img_id = img_ids[i] 
            
            for label, score, box in zip(res['labels'], res['scores'], res['boxes']):
                box = box.tolist()
                coco_results.append({
                    "image_id": img_id,
                    "category_id": label.item(),
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]], # xyxy -> xywh
                    "score": score.item(),
                })
    
    if not coco_results:
        print("No predictions made, skipping COCO eval.")
        return None

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval

def post_process(outputs, target_sizes):
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    prob = nn.functional.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)

    boxes = box_cxcywh_to_xyxy(out_bbox)
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    return results

def create_coco_gt(dataset_path, split='val'):
    """Creates a COCO-like ground truth object for evaluation."""
    data_yaml_path = Path(dataset_path) / 'data.yaml'
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    class_names = data_cfg['names']
    
    coco_gt_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name, "supercategory": "litter"} for i, name in enumerate(class_names)]
    }
    
    img_dir = Path(dataset_path) / 'images' / split
    label_dir = Path(dataset_path) / 'labels' / split
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    ann_id = 0
    for img_id, img_name in enumerate(image_files):
        img_path = img_dir / img_name
        h, w, _ = cv2.imread(str(img_path)).shape
        coco_gt_dict["images"].append({"id": img_id, "width": w, "height": h, "file_name": img_name})
        
        label_path = label_dir / f"{img_name.split('.')[0]}.txt"
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_c, y_c, width, height = map(float, line.strip().split())
                    x_min = (x_c - width / 2) * w
                    y_min = (y_c - height / 2) * h
                    bbox_w = width * w
                    bbox_h = height * h
                    
                    coco_gt_dict["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(class_id),
                        "bbox": [x_min, y_min, bbox_w, bbox_h],
                        "area": bbox_w * bbox_h,
                        "iscrowd": 0
                    })
                    ann_id += 1
                    
    # Save to a temporary file to load with COCO api
    gt_path = Path(Config.OUTPUT_DIR) / 'temp_gt.json'
    with open(gt_path, 'w') as f:
        json.dump(coco_gt_dict, f)
        
    return COCO(str(gt_path))

# ==============================================================================
# 6. UTILITY FUNCTIONS (Plotting, Inference)
# ==============================================================================
def plot_logs(logs):
    plt.figure(figsize=(12, 5))
    
    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(logs['train_loss'], label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot mAP
    plt.subplot(1, 2, 2)
    plt.plot(logs['map'], label='mAP @ 0.5:0.95')
    plt.plot(logs['map_50'], label='mAP @ 0.5')
    plt.title('Validation mAP per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'training_curves.png'))
    plt.show()

def run_inference(model_path, image_path, output_path, conf_threshold=0.7):
    print("\n--- Running Inference ---")
    # Load model
    model = HybridDetector(
        num_classes=Config.NUM_CLASSES,
        num_queries=Config.NUM_QUERIES,
        hidden_dim=Config.HIDDEN_DIM,
        nheads=Config.NHEADS,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(Config.DEVICE)
    model.eval()
    
    # Load and transform image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = get_transforms(train=False)
    transformed = transform(image=image_rgb, bboxes=[], class_labels=[])
    input_tensor = transformed['image'].unsqueeze(0).to(Config.DEVICE)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Post-process
    h, w, _ = image.shape
    target_sizes = torch.tensor([[h, w]], device=Config.DEVICE)
    results = post_process(outputs, target_sizes)[0]
    
    # Draw bounding boxes
    scores = results['scores']
    labels = results['labels']
    boxes = results['boxes']
    
    detections = 0
    for score, label, box in zip(scores, labels, boxes):
        if score > conf_threshold:
            detections += 1
            box = box.cpu().numpy().astype(int)
            x_min, y_min, x_max, y_max = box
            
            # Draw rectangle
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw label
            class_name = "litter" # Hardcoded for this dataset
            label_text = f"{class_name}: {score:.2f}"
            cv2.putText(image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    print(f"Found {detections} objects with confidence > {conf_threshold}")
    
    # Save output image
    cv2.imwrite(output_path, image)
    print(f"Inference result saved to: {output_path}")

    # Display image (optional, may not work in all environments)
    # plt.figure(figsize=(12, 12))
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()


# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================
def main():
    print("--- Advanced River Litter Detector ---")
    print(f"Using device: {Config.DEVICE}")

    # --- Setup Datasets and Dataloaders ---
    print("\nSetting up datasets...")
    train_dataset = TUDGVDataset(Config.DATASET_PATH, split='train', transform=get_transforms(train=True))
    val_dataset = TUDGVDataset(Config.DATASET_PATH, split='val', transform=get_transforms(train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, collate_fn=collate_fn)
    
    print(f"Training images: {len(train_dataset)}, Validation images: {len(val_dataset)}")
    
    # --- Setup Model, Criterion, and Optimizer ---
    print("\nInitializing model...")
    model = HybridDetector(
        num_classes=Config.NUM_CLASSES,
        num_queries=Config.NUM_QUERIES,
        hidden_dim=Config.HIDDEN_DIM,
        nheads=Config.NHEADS,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS
    )
    model.to(Config.DEVICE)
    
    matcher = HungarianMatcher(cost_class=Config.CLS_LOSS_COEF, cost_bbox=Config.BBOX_LOSS_COEF, cost_giou=Config.GIOU_LOSS_COEF)
    weight_dict = {'loss_ce': Config.CLS_LOSS_COEF, 'loss_bbox': Config.BBOX_LOSS_COEF, 'loss_giou': Config.GIOU_LOSS_COEF}
    criterion = SetCriterion(Config.NUM_CLASSES, matcher=matcher, weight_dict=weight_dict, eos_coef=Config.EOS_COEF)
    criterion.to(Config.DEVICE)
    
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": Config.LR_BACKBONE,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    # --- Training Loop ---
    print("\nStarting training...")
    best_map = 0
    logs = {'train_loss': [], 'map': [], 'map_50': []}
    
    # Prepare COCO ground truth for evaluation
    coco_gt = create_coco_gt(Config.DATASET_PATH, split='val')

    start_time = time.time()
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, Config.DEVICE, epoch)
        logs['train_loss'].append(train_loss)
        
        coco_eval = evaluate(model, criterion, val_loader, Config.DEVICE, coco_gt)
        
        if coco_eval:
            current_map = coco_eval.stats[0] # mAP @ 0.5:0.95
            current_map_50 = coco_eval.stats[1] # mAP @ 0.5
            logs['map'].append(current_map)
            logs['map_50'].append(current_map_50)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val mAP: {current_map:.4f} | Val mAP@.50: {current_map_50:.4f}")

            if current_map > best_map:
                best_map = current_map
                best_model_path = os.path.join(Config.OUTPUT_DIR, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"*** New best model saved to {best_model_path} with mAP: {best_map:.4f} ***")
        else:
            logs['map'].append(0)
            logs['map_50'].append(0)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val mAP: N/A")

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/60:.2f} minutes.")
    
    # --- Post-Training ---
    print("\nGenerating final plots...")
    plot_logs(logs)
    
    # --- Inference Demo ---
    # Find a sample image from the validation set to run inference on
    val_img_dir = Path(Config.DATASET_PATH) / 'images' / 'val'
    sample_image_path = str(val_img_dir / sorted(os.listdir(val_img_dir))[0])
    inference_output_path = os.path.join(Config.OUTPUT_DIR, 'inference_result.jpg')
    best_model_path = os.path.join(Config.OUTPUT_DIR, 'best_model.pth')
    
    if os.path.exists(best_model_path):
        run_inference(
            model_path=best_model_path,
            image_path=sample_image_path,
            output_path=inference_output_path
        )
    else:
        print("\nCould not find a saved model to run inference.")

if __name__ == '__main__':
    # Check if dataset path exists
    if not os.path.exists(Config.DATASET_PATH) or not os.path.exists(os.path.join(Config.DATASET_PATH, 'data.yaml')):
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print("!!! ERROR: Dataset not found or 'data.yaml' is missing.!!!")
         print(f"!!! Please change `Config.DATASET_PATH` to the correct path. It is currently: '{Config.DATASET_PATH}'")
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        main()