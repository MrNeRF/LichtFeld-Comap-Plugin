# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Feature extraction using pycolmap."""

from pathlib import Path
from typing import Callable, Optional

import pycolmap

from .utils import ColmapConfig


def extract_features(
    config: ColmapConfig,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> int:
    """Extract SIFT features from images.

    Args:
        config: COLMAP configuration
        progress: Optional callback (current, total, message)

    Returns:
        Number of images processed
    """
    images = (
        list(config.image_path.glob("*.jpg"))
        + list(config.image_path.glob("*.JPG"))
        + list(config.image_path.glob("*.jpeg"))
        + list(config.image_path.glob("*.JPEG"))
        + list(config.image_path.glob("*.png"))
        + list(config.image_path.glob("*.PNG"))
    )

    if not images:
        raise ValueError(f"No images found in {config.image_path}")

    if progress:
        progress(0, len(images), "Starting feature extraction...")

    camera_mode_map = {
        "AUTO": pycolmap.CameraMode.AUTO,
        "SINGLE": pycolmap.CameraMode.SINGLE,
        "PER_FOLDER": pycolmap.CameraMode.PER_FOLDER,
        "PER_IMAGE": pycolmap.CameraMode.PER_IMAGE,
    }

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.max_image_size = config.max_image_size
    extraction_options.sift.max_num_features = config.max_num_features

    pycolmap.extract_features(
        database_path=str(config.database_path),
        image_path=str(config.image_path),
        camera_mode=camera_mode_map.get(config.camera_mode, pycolmap.CameraMode.AUTO),
        camera_model=config.camera_model,
        extraction_options=extraction_options,
    )

    if progress:
        progress(len(images), len(images), "Feature extraction complete")

    return len(images)
