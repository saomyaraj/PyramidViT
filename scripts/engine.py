# scripts/engine.py

import math
import sys
import torch
from tqdm import tqdm

import sys
sys.path.append('e:/version_2/vit_detector')
from scripts.loss import SetCriterion
from scripts.utils import collate_fn

def train_one_epoch(model: torch.nn.Module, criterion: SetCriterion,
                    data_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    print(f"Starting epoch {epoch} [train]...")
    
    total_loss = 0
    num_batches = 0
    
    # Using tqdm for a progress bar
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [train]")
    
    for images, targets in pbar:
        # Move data to the correct device
        # Note: images is a list of tensors, targets is a list of dicts
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # This is a placeholder for proper image batching with padding
        # For now, we assume all images are resized to the same size
        # A more robust solution would pad images to the max size in the batch
        try:
            image_batch = torch.stack(images, 0)
        except RuntimeError:
            print("\nError: Images in batch have different sizes. Implement padding for robust training.")
            # Simple padding to max size in batch (crude)
            max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
            batch_shape = (len(images),) + max_size
            padded_images = images[0].new_full(batch_shape, 0)
            for img, pad_img in zip(images, padded_images):
                pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
            image_batch = padded_images


        outputs = model(image_batch)
        
        loss_dict = criterion(outputs, targets)
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2} # Example weights
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        total_loss += loss_value
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix(loss=loss_value, ce=loss_dict_reduced.get('loss_ce', 0), bbox=loss_dict_reduced.get('loss_bbox', 0))

    avg_loss = total_loss / max(num_batches, 1)
    print(f"Epoch {epoch} [train] finished. Average loss: {avg_loss:.4f}")
    
    return {'loss': avg_loss}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    total_loss = 0
    num_batches = 0
    pbar = tqdm(data_loader, desc="[eval]")

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Placeholder for padding
        try:
            image_batch = torch.stack(images, 0)
        except RuntimeError:
            max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
            batch_shape = (len(images),) + max_size
            padded_images = images[0].new_full(batch_shape, 0)
            for img, pad_img in zip(images, padded_images):
                pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
            image_batch = padded_images

        outputs = model(image_batch)
        loss_dict = criterion(outputs, targets)
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        total_loss += losses.item()
        num_batches += 1
        pbar.set_postfix(loss=losses.item())

    avg_loss = total_loss / max(num_batches, 1)
    print(f"[eval] finished. Average loss: {avg_loss:.4f}")
    
    # In a real scenario, you would also compute mAP here
    # This requires post-processing the model outputs and comparing with ground truth
    
    return {'loss': avg_loss}
