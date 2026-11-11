import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    alphabet: str = os.getenv("ALPHABET", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,-!?")
    height: int = int(os.getenv("HEIGHT", 64))
    max_width: int = int(os.getenv("MAX_WIDTH", 2048))
    width_scale: float = float(os.getenv("WIDTH_SCALE", 2.0))
    space_gap: int = int(os.getenv("SPACE_GAP", 6))
    backbone: str = os.getenv("BACKBONE", "repvit_m1_0")
    keep_width_stages: int = int(os.getenv("KEEP_WIDTH_STAGES", 1))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", 10))
    max_lines: int = int(os.getenv("MAX_LINES", 100))

    # model source: a local checkpoint wins; otherwise MLflow on DagsHub (registry -> run checkpoint)
    model_path: str = os.getenv("MODEL_PATH", "/models/best_exctc_model.pt")
    registry_uri: str = os.getenv("REGISTRY_URI", "models:/RepViT-ExCTC-OCR/1")
    repo_owner: str = os.getenv("REPO_OWNER", "omargaberxdev-hue")
    repo_name: str = os.getenv("REPO_NAME", "exctc")
    run_id: str = os.getenv("RUN_ID", "a8dc0e1a0651441a83a4d3ace4e34d10")
    checkpoint: str = os.getenv("CHECKPOINT", "best_exctc_model.pt")


settings = Settings()
