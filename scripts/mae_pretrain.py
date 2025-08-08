# scripts/mae_pretrain.py

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import os

import sys
sys.path.append('e:/version_2/vit_detector')
import config

# --- MAE Model ---
class MaskedAutoencoderViT(nn.Module):
    """
    Minimalist Masked Autoencoder for ViT pre-training.
    """
    def __init__(self, vit_model='vit_base_patch16_224', mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.vit = timm.create_model(vit_model, pretrained=False)
        
        # Encoder: Use the ViT's patch_embed and blocks
        self.patch_embed = self.vit.patch_embed
        self.blocks = self.vit.blocks
        self.norm = self.vit.norm
        
        # Decoder: A simple linear layer to reconstruct pixel values
        self.decoder_embed = nn.Linear(self.vit.embed_dim, self.patch_embed.patch_size[0]**2 * 3, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.vit.patch_embed.num_patches + 1, self.vit.embed_dim), requires_grad=False)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by shuffling indices and splitting.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.vit.pos_embed[:, 1:, :]

        # masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.vit.cls_token + self.vit.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # remove cls token
        x = x[:, 1:, :]
        
        # append mask tokens to sequence
        mask_tokens = self.decoder_pos_embed.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) # unshuffle

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 * 3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs, self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

# --- Main Pre-training Script ---
def main():
    print("Starting MAE Pre-training...")
    device = torch.device(config.DEVICE)

    # --- Dataset and DataLoader ---
    # Simple transform for pre-training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Use ImageFolder to grab all images, ignoring labels
    # Combine labeled and unlabeled data for pre-training
    datasets = []
    try:
        # Use the labeled training data for pre-training
        dataset_labeled = ImageFolder(root=os.path.join(config.DATA_PATH, '/images/train'), transform=transform)
        print(f"Found {len(dataset_labeled)} labeled images for pre-training.")
        datasets.append(dataset_labeled)
    except FileNotFoundError:
        print(f"Warning: Labeled training data not found at {os.path.join(config.DATA_PATH, '/images/train')}")

    # Optionally add unlabeled data if the path is provided
    if config.UNLABELED_DATA_PATH:
        try:
            dataset_unlabeled = ImageFolder(root=config.UNLABELED_DATA_PATH, transform=transform)
            print(f"Found {len(dataset_unlabeled)} unlabeled images.")
            datasets.append(dataset_unlabeled)
        except FileNotFoundError:
            print(f"Warning: Unlabeled dataset not found at {config.UNLABELED_DATA_PATH}")
    else:
        print("Info: No unlabeled data path provided. Pre-training will proceed using only the labeled training set.")

    if not datasets:
        print("Error: No data found for pre-training. Please check `config.DATA_PATH`.")
        return

    full_dataset = torch.utils.data.ConcatDataset(datasets)
    print(f"Total images for pre-training: {len(full_dataset)}")

    dataloader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    # --- Model and Optimizer ---
    model = MaskedAutoencoderViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
    
    # --- Training Loop ---
    for epoch in range(config.EPOCHS):
        total_loss = 0
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            
            optimizer.zero_grad()
            
            loss, _, _ = model(imgs)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % config.LOG_INTERVAL == 0:
                print(f"Epoch {epoch+1}/{config.EPOCHS} | Iteration {i} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Finished --- Avg Loss: {avg_loss:.4f} ---")

    # --- Save Backbone Weights ---
    if not os.path.exists(config.SAVE_DIR):
        os.makedirs(config.SAVE_DIR)
        
    save_path = os.path.join(config.SAVE_DIR, 'mae_vit_backbone.pth')
    # Save only the ViT backbone weights
    torch.save(model.vit.state_dict(), save_path)
    print(f"Pre-trained ViT backbone saved to {save_path}")


if __name__ == '__main__':
    main()
