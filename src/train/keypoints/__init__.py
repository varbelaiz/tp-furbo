"""Training module for keypoint detection models."""

from src.train.keypoints.config import (
    get_default_config,
    create_custom_config,
    TrainingConfig,
)
from src.train.keypoints.trainer import YOLOKeypointTrainer

__all__ = [
    'get_default_config',
    'create_custom_config',
    'TrainingConfig',
    'YOLOKeypointTrainer',
]
