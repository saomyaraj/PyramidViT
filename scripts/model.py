# scripts/model.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models._utils import IntermediateLayerGetter

import sys
sys.path.append('e:/version_2/vit_detector')
import config

# --- Helper Modules ---
class ConvStem(nn.Module):
    """
    A simple 2-layer Conv stem.
    """
    def __init__(self, in_channels=3, out_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class FPN(nn.Module):
    """
    Lightweight Feature Pyramid Network.
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        # features can be either a list or dict from backbone
        # Convert to list if it's a dict
        if isinstance(features, dict):
            features = [features[str(i)] for i in range(len(features))]
        
        laterals = [
            lateral_conv(features[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            # Get the target size for interpolation
            target_size = laterals[i-1].shape[-2:]
            # Interpolate to match the size of the lower level feature
            upsampled = F.interpolate(laterals[i], size=target_size, mode='nearest')
            laterals[i-1] += upsampled

        # Output convolutions
        outs = [
            self.output_convs[i](laterals[i]) for i in range(len(laterals))
        ]
        
        return outs


class DeformableDETRHead(nn.Module):
    """
    DETR-style detection head with deformable attention.
    This is a simplified placeholder. A real implementation would use MSDeformAttn.
    """
    def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1) # +1 for no-object
        self.bbox_embed = nn.Linear(hidden_dim, 4) # cxcywh

    def forward(self, features):
        # features is a list of feature maps from FPN
        # For simplicity, we'll use the smallest feature map for the transformer
        src = features[-1] # Use the highest-level feature map
        bs, c, h, w = src.shape
        
        src = src.flatten(2).permute(0, 2, 1) # (bs, h*w, c)
        
        # The object queries are the same for all images in the batch
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        
        # Use transformer encoder and decoder
        # Encoder processes the image features
        memory = self.transformer.encoder(src)
        
        # Decoder generates object predictions from queries
        hs = self.transformer.decoder(query_embed, memory)
        
        # Class and bbox predictions
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        return outputs_class, outputs_coord

# --- Main Model ---
class ViTDetector(nn.Module):
    def __init__(self, backbone_name=None, num_classes=None, num_queries=None, 
                 hidden_dim=None, nheads=None, num_encoder_layers=None, 
                 num_decoder_layers=None):
        super().__init__()
        
        # Use config defaults if parameters not provided
        self.backbone_name = backbone_name or config.BACKBONE
        self.num_classes = num_classes or config.NUM_CLASSES
        self.num_queries = num_queries or config.NUM_QUERIES
        self.hidden_dim = hidden_dim or config.HIDDEN_DIM
        self.nheads = nheads or config.NHEADS
        self.num_encoder_layers = num_encoder_layers or config.NUM_ENCODER_LAYERS
        self.num_decoder_layers = num_decoder_layers or config.NUM_DECODER_LAYERS
        
        # --- Backbone ---
        # Using timm's Vision Transformer backbone
        # PVT-v2-B2 is generally better than Swin for detection tasks
        # We need features at multiple stages
        try:
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=True, # Start with ImageNet pre-training
                features_only=True,
                out_indices=(1, 2, 3) # Get features from intermediate stages
            )
        except Exception as e:
            print(f"Warning: Could not create {self.backbone_name}, falling back to swin_tiny")
            self.backbone = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=True,
                features_only=True,
                out_indices=(1, 2, 3)
            )
        
        # --- FPN ---
        backbone_channels = self.backbone.feature_info.channels()
        self.fpn = FPN(in_channels_list=backbone_channels, out_channels=config.FPN_CHANNELS)

        # --- Detection Head ---
        self.head = DeformableDETRHead(
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            nheads=self.nheads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            num_queries=self.num_queries
        )
        
        # Note: Iterative Refinement and Conv-Stem are not included in this simplified first pass
        # to keep the core logic clear. They can be added as modular components.

    def load_mae_backbone(self, path):
        """Loads weights from a self-supervised MAE pre-trained model."""
        print(f"Loading MAE backbone from {path}")
        state_dict = torch.load(path, map_location='cpu')
        # We need to be careful about mismatched keys (e.g., if MAE model has a 'head')
        self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, images):
        # Get multi-scale features from the backbone
        features = self.backbone(images)
        
        # Features is already a list from timm models with features_only=True
        # Get FPN features
        fpn_features = self.fpn(features)
        
        # Get predictions from the head
        # The head will internally select the feature maps it needs
        pred_logits, pred_boxes = self.head(fpn_features)
        
        output = {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}
        
        # Return directly as dict for compatibility with DETR loss function
        return output

if __name__ == '__main__':
    # Example usage
    model = ViTDetector(
        backbone_name=config.BACKBONE,
        num_classes=config.NUM_CLASSES,
        num_queries=config.NUM_QUERIES,
        hidden_dim=config.HIDDEN_DIM,
        nheads=config.NHEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS
    ).to(config.DEVICE)
    
    # Load self-supervised weights if available
    if hasattr(config, 'PRETRAINED_BACKBONE_PATH') and os.path.exists(config.PRETRAINED_BACKBONE_PATH):
        model.load_mae_backbone(config.PRETRAINED_BACKBONE_PATH)

    # Create a dummy input
    dummy_images = torch.randn(2, 3, 800, 800).to(config.DEVICE)
    
    # Forward pass
    outputs = model(dummy_images)
    
    print("Model created successfully!")
    print("Output shapes:")
    print(f"  - Logits: {outputs[0]['pred_logits'].shape}") # (batch_size, num_queries, num_classes + 1)
    print(f"  - Boxes: {outputs[0]['pred_boxes'].shape}")   # (batch_size, num_queries, 4)
