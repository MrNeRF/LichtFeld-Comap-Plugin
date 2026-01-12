# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Image undistortion using pycolmap."""

from typing import Callable, Optional

import pycolmap

from .utils import ColmapConfig


def undistort_images(
    config: ColmapConfig,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    """Undistort images using estimated camera parameters.

    Args:
        config: COLMAP configuration
        progress: Optional callback (current, total, message)
    """
    if progress:
        progress(0, 100, "Undistorting images...")

    config.undistorted_path.mkdir(parents=True, exist_ok=True)

    input_path = config.sparse_path / "0"

    pycolmap.undistort_images(
        output_path=str(config.undistorted_path),
        input_path=str(input_path),
        image_path=str(config.image_path),
        output_type="COLMAP",
    )

    if progress:
        progress(100, 100, "Undistortion complete")
