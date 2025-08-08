# scripts/utils.py

import torch

def collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    Since each image can have a different number of objects, we can't stack them
    in a single tensor. This function simply returns a list of images and a list of targets.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # The images are already tensors, but we can stack them if they have the same size
    # However, with data augmentation, they might not.
    # For now, we'll keep them as a list. The training script will handle padding if necessary.
    # images = torch.stack(images, 0) # This would fail if images have different sizes
    
    return images, targets

def box_cxcywh_to_xyxy(x):
    """
    Convert boxes from (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max).
    Both formats are normalized.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    """
    Convert boxes from (x_min, y_min, x_max, y_max) to (center_x, center_y, width, height).
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


if __name__ == '__main__':
    # --- Test utility functions ---
    print("Testing utility functions...")

    # Test box conversion
    cxcywh = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    xyxy = box_cxcywh_to_xyxy(cxcywh)
    print(f"\nInput (cxcywh): {cxcywh}")
    print(f"Converted (xyxy): {xyxy}")
    # Expected output: tensor([[0.4000, 0.4000, 0.6000, 0.6000]])
    
    # Test back-conversion
    cxcywh_recon = box_xyxy_to_cxcywh(xyxy)
    print(f"Re-converted (cxcywh): {cxcywh_recon}")
    
    assert torch.allclose(cxcywh, cxcywh_recon), "Box conversion functions are not inverses!"
    print("Box conversion test passed.")

    # Test collate_fn (dummy example)
    print("\nTesting collate_fn...")
    dummy_batch = [
        (torch.randn(3, 224, 224), {'boxes': torch.rand(2, 4), 'labels': torch.randint(0, 5, (2,))}),
        (torch.randn(3, 224, 224), {'boxes': torch.rand(5, 4), 'labels': torch.randint(0, 5, (5,))})
    ]
    
    images, targets = collate_fn(dummy_batch)
    
    print(f"Collate function output:")
    print(f"  - Type of images: {type(images)}")
    print(f"  - Length of images list: {len(images)}")
    print(f"  - Type of targets: {type(targets)}")
    print(f"  - Length of targets list: {len(targets)}")
    print(f"  - Number of boxes in first target: {len(targets[0]['boxes'])}")
    print(f"  - Number of boxes in second target: {len(targets[1]['boxes'])}")
    print("Collate function test passed.")
