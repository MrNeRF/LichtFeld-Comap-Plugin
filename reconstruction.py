# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bundle adjustment / reconstruction using pycolmap."""

from typing import Callable, Optional

import pycolmap

from .utils import ColmapConfig, ReconstructionResult


def reconstruct(
    config: ColmapConfig,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> ReconstructionResult:
    """Run incremental Structure-from-Motion.

    Args:
        config: COLMAP configuration
        progress: Optional callback (current, total, message)

    Returns:
        ReconstructionResult with reconstruction statistics
    """
    if progress:
        progress(0, 100, "Starting incremental mapping...")

    config.sparse_path.mkdir(parents=True, exist_ok=True)

    pipeline_options = pycolmap.IncrementalPipelineOptions()

    reconstructions = pycolmap.incremental_mapping(
        database_path=str(config.database_path),
        image_path=str(config.image_path),
        output_path=str(config.sparse_path),
        options=pipeline_options,
    )

    if not reconstructions:
        return ReconstructionResult(
            success=False,
            error="No valid reconstructions found",
        )

    recon = max(reconstructions.values(), key=lambda r: r.num_images())

    output_model = config.sparse_path / "0"
    output_model.mkdir(exist_ok=True)
    recon.write(str(output_model))

    if progress:
        progress(100, 100, f"Reconstruction complete: {recon.num_images()} images")

    return ReconstructionResult(
        success=True,
        sparse_path=output_model,
        num_cameras=recon.num_cameras(),
        num_images=recon.num_images(),
        num_points=recon.num_points3D(),
        mean_reproj_error=recon.compute_mean_reprojection_error(),
    )
