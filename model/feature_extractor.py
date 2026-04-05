import torch
from torch import nn
from transformers import AutoModel


CONVNEXT_CONFIGS = {
    "tiny":  {"model": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",  "channels": 768},
    "small": {"model": "facebook/dinov3-convnext-small-pretrain-lvd1689m", "channels": 768},
    "base":  {"model": "facebook/dinov3-convnext-base-pretrain-lvd1689m",  "channels": 1024},
    "large": {"model": "facebook/dinov3-convnext-large-pretrain-lvd1689m", "channels": 1536},
}


class ConvNeXtFeatureExtractor(nn.Module):
    """
    Wraps the DINOv3 ConvNeXt encoder (the stages ModuleList) into a
    module that takes pixel input and returns a plain feature tensor.

    DINOv3 model structure:
        model.model = DINOv3ConvNextEncoder (contains model.model.stages)
        model.layer_norm = LayerNorm (dropped)
        model.pool = AdaptiveAvgPool2d (dropped)
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        out = self.encoder(x)
        # Handle BaseModelOutput, tuple, or plain tensor
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        elif isinstance(out, tuple):
            return out[0]
        return out


def get_feature_extractor(num_labels=4, pretrained_weights='', variant='base'):
    """
    Build a ConvNeXt feature extractor.

    Args:
        num_labels: unused (kept for backward compat with old call sites).
        pretrained_weights: path to a .pth file with feature extractor weights.
        variant: ConvNeXt size — 'tiny', 'small', 'base', or 'large'.

    Returns:
        (feature_extractor, channels): the nn.Sequential feature extractor and its output channel count.
    """
    cfg = CONVNEXT_CONFIGS[variant]
    model = AutoModel.from_pretrained(cfg["model"])

    # DINOv3 structure: model.model = DINOv3ConvNextEncoder, model.layer_norm, model.pool
    # We keep just the encoder (which contains the stages), drop LayerNorm and pool
    encoder = list(model.children())[0]  # DINOv3ConvNextEncoder
    fe = ConvNeXtFeatureExtractor(encoder)

    if pretrained_weights:
        fe.load_state_dict(torch.load(pretrained_weights, map_location=torch.device("cuda")))

    return fe, cfg["channels"]


def load_pretrained_feature_extractor(fe_path, variant='base'):
    """
    Load a feature extractor from a Stage 1 MAE2D checkpoint.

    The checkpoint contains just the feature extractor state_dict
    (not wrapped in CustomHuggingFaceModel).

    Args:
        fe_path: path to convnext_mae2d.pth
        variant: ConvNeXt size — must match what was used in Stage 1.

    Returns:
        (feature_extractor, channels)
    """
    fe, channels = get_feature_extractor(4, variant=variant)
    fe.load_state_dict(torch.load(fe_path, map_location="cuda"))
    return fe, channels
