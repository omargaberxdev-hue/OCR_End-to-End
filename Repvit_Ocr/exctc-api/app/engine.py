"""Model loading + inference. Only the 1D CTC branch is used."""
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import Settings
from .model import RepViTExCTC, build_id2ch, ctc_collapse, ctc_input_lengths

log = logging.getLogger("exctc")
PAD_VALUE = 127  # ~ the pixel value that normalises to 0 (what collate() pads training batches with)


def _build_from_checkpoint(path: str, cfg: Settings, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    alphabet = ckpt.get("alphabet", cfg.alphabet)
    id2ch = build_id2ch(alphabet)
    model = RepViTExCTC(num_classes=len(id2ch), backbone_name=cfg.backbone, pretrained=False,
                        keep_width_stages=ckpt.get("keep_width_stages", cfg.keep_width_stages)).to(device)
    model.load_state_dict(ckpt["model"])
    return model, id2ch, f"checkpoint {path} (epoch {ckpt.get('epoch', '?')})"


def _load_from_mlflow(cfg: Settings, device):
    import mlflow, mlflow.pytorch
    mlflow.set_tracking_uri(f"https://dagshub.com/{cfg.repo_owner}/{cfg.repo_name}.mlflow")
    # DagsHub auth comes from MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD env vars.
    try:
        # the pickled model references __main__.<Class>; expose the classes there so unpickling works
        import __main__
        from . import model as m
        for n in ("RepViTBackbone", "SqueezeNet", "RepViTExCTC"):
            setattr(__main__, n, getattr(m, n))
        pickled = mlflow.pytorch.load_model(cfg.registry_uri, map_location=device)
        # The pickle carries the training classes' bytecode by value (built on another Python version),
        # so running its forward() can crash. Keep only the weights and load them into OUR classes.
        stride_x = int(pickled.backbone.stride_x)                  # 32 // 2**keep_width_stages
        keep = {32: 0, 16: 1, 8: 2, 4: 3}.get(stride_x, cfg.keep_width_stages)
        model = RepViTExCTC(num_classes=int(pickled.num_classes), backbone_name=cfg.backbone,
                            pretrained=False, keep_width_stages=keep).to(device)
        model.load_state_dict(pickled.state_dict())
        del pickled
        return model, build_id2ch(cfg.alphabet), f"registry {cfg.registry_uri} (weights)"
    except Exception as e:
        log.warning("registry load failed (%s: %s); falling back to run checkpoint", type(e).__name__, e)
    path = mlflow.artifacts.download_artifacts(run_id=cfg.run_id, artifact_path=f"checkpoint/{Path(cfg.checkpoint).name}")
    return _build_from_checkpoint(path, cfg, device)


class OCREngine:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(int(os.getenv("TORCH_THREADS", max(1, (os.cpu_count() or 2) // 2))))
        if Path(cfg.model_path).is_file():
            model, self.id2ch, self.info = _build_from_checkpoint(cfg.model_path, cfg, self.device)
        else:
            model, self.id2ch, self.info = _load_from_mlflow(cfg, self.device)
        self.model = model.eval()
        assert len(self.id2ch) == self.model.num_classes, \
            f"vocab has {len(self.id2ch)} classes, model head has {self.model.num_classes}"
        log.info("loaded %s | classes %d | device %s", self.info, len(self.id2ch), self.device)

    @torch.no_grad()
    def read(self, img_rgb: np.ndarray, spaces: bool) -> str:
        """One line / word crop (RGB uint8) -> text."""
        cfg = self.cfg
        H, W = img_rgb.shape[:2]
        new_w = min(cfg.max_width, max(1, round(W * cfg.height / H * cfg.width_scale)))
        im = cv2.resize(img_rgb, (new_w, cfg.height))
        im = cv2.copyMakeBorder(im, 0, 0, 0, -new_w % 32, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
        x = (im.astype(np.float32) / 255.0 - 0.5) / 0.5            # == Normalize(mean=.5, std=.5)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
        logits = self.model(x)["logits"]                            # [T,1,K]
        n = ctc_input_lengths(torch.tensor([new_w]), x.shape[-1], logits.shape[0]).item()
        ids = logits[:n, 0].argmax(-1).cpu().numpy()
        return ctc_collapse(ids, self.id2ch, gap=cfg.space_gap if spaces else None)
