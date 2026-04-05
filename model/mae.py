import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import MaskedAutoEncoderViT


class SLIViTMAE(nn.Module):
    """
    SLIViT MAE: ConvNeXt feature extractor + MONAI MaskedAutoEncoderViT.

    The ConvNeXt processes concatenated volume slices into a feature map
    of shape (B, 768, 8, N*8). This is treated as a 2D "image" with 768
    channels and fed to MONAI's MAE, which patches it into N patches of
    (768, 8, 8) = 49,152 values each — one patch per B-scan slice.

    During training, forward() returns the MSE reconstruction loss.
    For embedding extraction, use encode() which returns the CLS token.
    """

    def __init__(
        self,
        feature_extractor,
        num_patches=28,
        hidden_size=256,
        mlp_dim=1024,
        num_layers=5,
        num_heads=16,
        masking_ratio=0.75,
        decoder_hidden_size=128,
        decoder_mlp_dim=512,
        decoder_num_layers=2,
        decoder_num_heads=4,
        dropout_rate=0.0,
        pos_embed_type="sincos",
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_patches = num_patches
        self.feat_channels = 768
        self.patch_h = 8
        self.patch_w = 8
        self.patch_dim = self.feat_channels * self.patch_h * self.patch_w  # 49152

        self.mae = MaskedAutoEncoderViT(
            in_channels=self.feat_channels,
            img_size=(self.patch_h, num_patches * self.patch_w),
            patch_size=(self.patch_h, self.patch_w),
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

    def _patchify(self, feats):
        """
        Reshape feature map into per-slice patch vectors.

        Args:
            feats: (B, 768, 8, N*8)

        Returns:
            patches: (B, N, 49152)
        """
        B, C, H, W = feats.shape
        nH = H // self.patch_h
        nW = W // self.patch_w
        patches = feats.reshape(B, C, nH, self.patch_h, nW, self.patch_w)
        patches = patches.permute(0, 2, 4, 1, 3, 5)  # (B, nH, nW, C, pH, pW)
        patches = patches.reshape(B, nH * nW, -1)  # (B, N, C*pH*pW)
        return patches

    def forward(self, x):
        """
        MAE forward pass: extract features, mask, encode, decode, compute loss.

        Args:
            x: (B, 3, 256, 256*N) concatenated volume slices

        Returns:
            loss: scalar MSE loss on masked patch features
        """
        # Feature extraction (end-to-end)
        feats = self.feature_extractor(x).last_hidden_state  # (B, 768, 8, N*8)

        # Reconstruction targets (stop gradient)
        targets = self._patchify(feats.detach())  # (B, N, 49152)

        # MONAI MAE: patch embed → mask → encode → decode → predict
        pred, mask = self.mae(feats)  # pred: (B, N, 49152), mask: (B, N) 1=masked

        # MSE loss on masked patches only
        loss = F.mse_loss(pred[mask == 1], targets[mask == 1])
        return loss

    def encode(self, x):
        """
        Extract volume embeddings (CLS token) — no masking.

        Args:
            x: (B, 3, 256, 256*N) concatenated volume slices

        Returns:
            embeddings: (B, hidden_size)
        """
        feats = self.feature_extractor(x).last_hidden_state  # (B, 768, 8, N*8)

        # Patch embedding (conv projection + positional embedding)
        tokens = self.mae.patch_embedding(feats)  # (B, N, hidden_size)

        # Prepend CLS token
        cls_tokens = self.mae.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)  # (B, 1+N, hidden_size)

        # Encoder
        tokens = self.mae.blocks(tokens)  # (B, 1+N, hidden_size)

        return tokens[:, 0]  # CLS token embedding
