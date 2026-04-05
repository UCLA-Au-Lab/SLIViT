import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import MaskedAutoEncoderViT


class ConvNeXtMAE2D(nn.Module):
    """
    Stage 1: 2D feature-level MAE for pretraining ConvNeXt.

    ConvNeXt processes individual 2D slices (3, 256, 256) into feature maps
    (feat_channels, 8, 8). MONAI MAE operates on these feature maps:
    64 patches of feat_channels values each, masks 75%, encodes visible,
    decodes, and reconstructs the masked patches.

    After training, save just the ConvNeXt weights with save_feature_extractor().
    """

    def __init__(
        self,
        feature_extractor,
        feat_channels=1024,
        hidden_size=512,
        mlp_dim=2048,
        num_layers=6,
        num_heads=8,
        masking_ratio=0.75,
        decoder_hidden_size=256,
        decoder_mlp_dim=1024,
        decoder_num_layers=4,
        decoder_num_heads=4,
        dropout_rate=0.0,
        pos_embed_type="sincos",
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feat_channels = feat_channels

        self.mae = MaskedAutoEncoderViT(
            in_channels=feat_channels,
            img_size=(8, 8),
            patch_size=(1, 1),  # 64 patches, one per spatial position
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            masking_ratio=masking_ratio,
            decoder_hidden_size=decoder_hidden_size,
            decoder_mlp_dim=decoder_mlp_dim,
            decoder_num_layers=decoder_num_layers,
            decoder_num_heads=decoder_num_heads,
            proj_type="conv",
            pos_embed_type=pos_embed_type,
            decoder_pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=2,
        )

    def _extract_features(self, x):
        """Run ConvNeXt feature extractor."""
        return self.feature_extractor(x)

    def forward(self, x):
        """
        2D MAE forward pass on individual slices.

        Args:
            x: (B, 3, 256, 256) single B-scan slices

        Returns:
            loss: scalar MSE loss on masked feature patches
        """
        # Feature extraction
        feats = self._extract_features(x)  # (B, feat_channels, 8, 8)
        B, C, H, W = feats.shape

        # Reconstruction targets (stop gradient)
        # Patchify: (B, C, 8, 8) with patch (1,1) → (B, 64, C)
        targets = feats.detach().reshape(B, C, H * W).permute(0, 2, 1)  # (B, 64, feat_channels)

        # MONAI MAE forward
        with torch.amp.autocast(device_type="cuda", enabled=False):
            pred, mask = self.mae(feats.float())  # pred: (B, 64, feat_channels), mask: (B, 64)

        # MSE loss on masked patches only
        loss = F.mse_loss(pred[mask == 1], targets[mask == 1])
        return loss

    def save_feature_extractor(self, path):
        """Save just the ConvNeXt feature extractor state dict for Stage 2."""
        torch.save(self.feature_extractor.state_dict(), path)
