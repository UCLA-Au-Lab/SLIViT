import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SLIViTMAE(nn.Module):
    """
    Masked Autoencoder for SLIViT.

    Architecture:
      ConvNeXt feature extractor → ViT encoder (with masking) → lightweight decoder

    During training, a fraction of slice-patches are masked. The encoder processes
    only visible patches; the decoder reconstructs masked patch features.

    For embedding extraction, use the `encode()` method which returns the CLS token.
    """

    def __init__(
        self,
        feature_extractor,
        num_patches,
        encoder_dim=256,
        encoder_depth=5,
        encoder_heads=20,
        decoder_dim=128,
        decoder_depth=2,
        decoder_heads=4,
        mask_ratio=0.75,
        patch_feat_dim=768 * 64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.patch_feat_dim = patch_feat_dim

        # --- Encoder ---
        self.patch_embed = nn.Linear(patch_feat_dim, encoder_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, encoder_dim) * 0.02)
        self.emb_dropout = nn.Dropout(emb_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        # --- Decoder ---
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)
        self.decoder_pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, decoder_dim) * 0.02
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_feat_dim)

        self._init_weights()

    def _init_weights(self):
        # Xavier uniform for linear layers, normal for embeddings
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.decoder_pos_embedding, std=0.02)

    def random_masking(self, B, N, device):
        """
        Per-sample random masking via argsort of noise.

        Returns:
            ids_keep: (B, num_visible) indices of visible patches
            ids_mask: (B, num_masked) indices of masked patches
            ids_restore: (B, N) indices to unshuffle back to original order
        """
        num_keep = max(1, int(N * (1 - self.mask_ratio)))

        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :num_keep]
        ids_mask = ids_shuffle[:, num_keep:]

        return ids_keep, ids_mask, ids_restore

    def forward_encoder(self, patches, ids_keep):
        """
        Encode only visible patches.

        Args:
            patches: (B, N, patch_feat_dim) all patch features
            ids_keep: (B, num_visible) indices of visible patches

        Returns:
            encoded: (B, 1 + num_visible, encoder_dim)
        """
        B = patches.shape[0]

        # Gather visible patches
        visible = torch.gather(
            patches, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.patch_feat_dim)
        )  # (B, num_visible, patch_feat_dim)

        # Embed
        visible = self.patch_embed(visible)  # (B, num_visible, encoder_dim)

        # Add positional embeddings (skip pos 0 which is for CLS)
        vis_pos = torch.gather(
            self.pos_embedding[:, 1:].expand(B, -1, -1),
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, visible.shape[-1]),
        )
        visible = visible + vis_pos

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.pos_embedding[:, :1]
        visible = torch.cat([cls_tokens, visible], dim=1)

        visible = self.emb_dropout(visible)

        # Encode
        encoded = self.encoder(visible)
        encoded = self.encoder_norm(encoded)

        return encoded

    def forward_decoder(self, encoded, ids_restore):
        """
        Decode: project encoder output, add mask tokens, predict patch features.

        Args:
            encoded: (B, 1 + num_visible, encoder_dim)
            ids_restore: (B, N) indices to unshuffle

        Returns:
            pred: (B, N, patch_feat_dim) predicted features for all positions
        """
        B, _, _ = encoded.shape

        # Project to decoder dim
        tokens = self.decoder_embed(encoded)  # (B, 1 + num_visible, decoder_dim)

        # Separate CLS and patch tokens
        cls_dec = tokens[:, :1]
        patch_tokens = tokens[:, 1:]  # (B, num_visible, decoder_dim)

        # Append mask tokens for masked positions
        num_masked = self.num_patches - patch_tokens.shape[1]
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        all_patch_tokens = torch.cat([patch_tokens, mask_tokens], dim=1)  # (B, N, decoder_dim)

        # Unshuffle to original order
        all_patch_tokens = torch.gather(
            all_patch_tokens,
            1,
            ids_restore.unsqueeze(-1).expand(-1, -1, all_patch_tokens.shape[-1]),
        )

        # Add decoder positional embeddings
        all_patch_tokens = all_patch_tokens + self.decoder_pos_embedding[:, 1:]

        # Prepend CLS
        cls_dec = cls_dec + self.decoder_pos_embedding[:, :1]
        full_seq = torch.cat([cls_dec, all_patch_tokens], dim=1)  # (B, 1+N, decoder_dim)

        # Decode
        decoded = self.decoder(full_seq)
        decoded = self.decoder_norm(decoded)

        # Predict patch features (skip CLS)
        pred = self.decoder_pred(decoded[:, 1:])  # (B, N, patch_feat_dim)

        return pred

    def forward(self, x):
        """
        MAE forward pass: extract features, mask, encode, decode, compute loss.

        Args:
            x: (B, C, H, W) concatenated volume slices

        Returns:
            loss: scalar MSE loss on masked patch features
        """
        B = x.shape[0]

        # 1. Feature extraction (full volume, end-to-end)
        feats = self.feature_extractor(x).last_hidden_state  # (B, 768, H, W)
        feats = feats.reshape(B, self.num_patches, 768, 64)  # (B, N, 768, 64)

        # Reconstruction targets (stop gradient)
        targets = feats.detach().reshape(B, self.num_patches, -1)  # (B, N, 768*64)

        # Flatten patches for encoder input
        patches = feats.reshape(B, self.num_patches, -1)  # (B, N, 768*64)

        # 2. Random masking
        ids_keep, ids_mask, ids_restore = self.random_masking(B, self.num_patches, x.device)

        # 3. Encode visible patches
        encoded = self.forward_encoder(patches, ids_keep)

        # 4. Decode all patches
        pred = self.forward_decoder(encoded, ids_restore)

        # 5. Loss: MSE on masked patches only
        target_masked = torch.gather(
            targets, 1, ids_mask.unsqueeze(-1).expand(-1, -1, self.patch_feat_dim)
        )
        pred_masked = torch.gather(
            pred, 1, ids_mask.unsqueeze(-1).expand(-1, -1, self.patch_feat_dim)
        )
        loss = F.mse_loss(pred_masked, target_masked)

        return loss

    def encode(self, x):
        """
        Extract volume embeddings (CLS token) — no masking.

        Args:
            x: (B, C, H, W) concatenated volume slices

        Returns:
            embeddings: (B, encoder_dim)
        """
        B = x.shape[0]

        feats = self.feature_extractor(x).last_hidden_state
        feats = feats.reshape(B, self.num_patches, 768, 64)
        patches = feats.reshape(B, self.num_patches, -1)

        # Embed all patches
        tokens = self.patch_embed(patches) + self.pos_embedding[:, 1:]

        # Prepend CLS
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.pos_embedding[:, :1]
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Encode
        encoded = self.encoder(tokens)
        encoded = self.encoder_norm(encoded)

        return encoded[:, 0]  # CLS token embedding
