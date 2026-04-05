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
    Wraps ConvNeXt embeddings + encoder into a single module that
    returns a plain tensor (not a BaseModelOutput).
    """

    def __init__(self, embeddings, encoder):
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder

    def forward(self, x):
        x = self.embeddings(x)
        out = self.encoder(x)
        # Handle both BaseModelOutput and tuple returns
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


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

    # Extract embeddings + encoder, drop final LayerNorm
    children = list(model.children())
    fe = ConvNeXtFeatureExtractor(children[0], children[1])

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
