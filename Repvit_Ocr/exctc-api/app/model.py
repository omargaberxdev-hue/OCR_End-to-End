"""Architecture (kept exactly as trained so checkpoints / pickled models load) + CTC decoding."""
import torch
import torch.nn.functional as F
import timm
from torch import nn


class RepViTBackbone(nn.Module):
    def __init__(self, model_name: str = "repvit_m1_0", pretrained: bool = True, keep_width_stages: int = 1):
        super().__init__()
        self.model_name = model_name
        self.network = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")

        downs = [s.downsample for s in list(self.network.stages)[1:]]
        assert all(hasattr(d, "spatial_downsample") for d in downs), "unexpected timm RepViT layout"
        assert 0 <= keep_width_stages <= len(downs), f"keep_width_stages must be 0..{len(downs)}"
        for d in downs[len(downs) - keep_width_stages:]:
            d.spatial_downsample.c.stride = (2, 1)
        self.stride_x, self.stride_y = 32 // 2 ** keep_width_stages, 32

        was_training = self.network.training
        self.network.eval()
        try:
            with torch.no_grad():
                features = self._forward_features(torch.zeros(1, 3, 80, 256))
                if features.ndim != 4:
                    raise RuntimeError(f"{model_name}.forward_features returned {tuple(features.shape)}; expected [B, C, H, W].")
                self.out_channels = int(features.shape[1])
        finally:
            self.network.train(was_training)

    def _forward_features(self, images):
        features = self.network.forward_features(images)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        return features

    def forward(self, images):
        return self._forward_features(images)


class SqueezeNet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.f2map = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, features, temperature: float = 1.0):
        squeeze_map = F.relu(self.bn(self.f2map(features)))
        concentrate_loss = 1.0 / squeeze_map.var(dim=2).mean()
        squeeze_map = F.softmax(squeeze_map / temperature, dim=2)
        sequence = torch.sum(features * squeeze_map, dim=2)
        return sequence, squeeze_map, concentrate_loss


class RepViTExCTC(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str = "repvit_m1_0", pretrained: bool = True,
                 keep_width_stages: int = 1):
        super().__init__()
        self.backbone = RepViTBackbone(backbone_name, pretrained, keep_width_stages)
        channels = self.backbone.out_channels
        self.S = SqueezeNet(channels)
        self.C = nn.Linear(channels, num_classes)
        self.num_classes = num_classes

    def forward(self, images, temperature: float = 1.0, dense_only: bool = False):
        features = self.backbone(images)
        features = F.dropout(features, p=0.5, training=self.training)

        if dense_only:
            logits_2d = self.C(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return {"logits_2d": logits_2d, "feature_map_2d": features}

        features_1d, squeeze_map, concentrate_loss = self.S(features, temperature)
        ctc_logits = self.C(features_1d.permute(2, 0, 1))

        B, C, H, W = features.shape
        dense = features.reshape(B, C, H * W).permute(2, 0, 1)
        dense_logits = self.C(dense)

        return {"logits": ctc_logits, "dense_logits": dense_logits, "squeeze_map": squeeze_map,
                "concentrate_loss": concentrate_loss, "feature_map_2d": features}


def ctc_input_lengths(input_widths, padded_input_width: int, output_width: int):
    ratio = input_widths.float() / float(padded_input_width)
    return torch.round(ratio * output_width).long().clamp(min=1, max=output_width)


# ---- vocabulary / decoding (no HF tokenizers needed at inference) ----
def build_id2ch(alphabet: str) -> dict[int, str]:
    """id 0 = <blank>, 1 = <unk>, then characters in alphabet order (same as the training tokenizer)."""
    vocab = {"<blank>": 0, "<unk>": 1}
    for ch in alphabet:
        vocab.setdefault(ch, len(vocab))
    return {i: c for c, i in vocab.items()}


def ctc_collapse(ids, id2ch, gap=None):
    """Greedy CTC collapse; `gap` inserts a space between characters emitted > gap columns apart."""
    out, prev, last_x = [], 0, None
    for x, i in enumerate(ids):
        i = int(i)
        if i != prev and i > 1:
            if gap is not None and last_x is not None and x - last_x > gap:
                out.append(" ")
            out.append(id2ch[i])
            last_x = x
        prev = i
    return "".join(out)
