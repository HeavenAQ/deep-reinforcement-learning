from datetime import datetime
from logging import getLogger
from pathlib import Path

import torch

logger = getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def save_model(model: torch.nn.Module, model_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"{model_name}_{timestamp}.pth"
    checkpoint_path = PROJECT_ROOT / "models" / checkpoint_name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"{checkpoint_name} has been saved to {checkpoint_path}.")
