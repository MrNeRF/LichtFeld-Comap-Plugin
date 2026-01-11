# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Full COLMAP pipeline."""

from pathlib import Path
from typing import Callable, Optional, Union

from .features import extract_features
from .matching import match_features
from .reconstruction import reconstruct
from .runner import ColmapJob
from .utils import ColmapConfig, ReconstructionResult


def run_pipeline(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    camera_model: str = "OPENCV",
    match_type: str = "exhaustive",
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> ReconstructionResult:
    """Run full COLMAP pipeline synchronously.

    Args:
        image_path: Path to directory containing images
        output_path: Optional output directory (default: temp dir)
        camera_model: Camera model (OPENCV, PINHOLE, SIMPLE_RADIAL, etc.)
        match_type: Matching strategy (exhaustive, sequential, vocab_tree, spatial)
        progress: Optional callback (current, total, message)

    Returns:
        ReconstructionResult with reconstruction statistics
    """
    config = ColmapConfig(
        image_path=image_path,
        output_path=output_path,
        camera_model=camera_model,
        match_type=match_type,
    )

    try:
        extract_features(config, progress)
        match_features(config, progress)
        return reconstruct(config, progress)

    except Exception as e:
        return ReconstructionResult(success=False, error=str(e))


def run_pipeline_async(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    camera_model: str = "OPENCV",
    match_type: str = "exhaustive",
    on_progress: Optional[Callable[[str, float, str], None]] = None,
    on_complete: Optional[Callable[[ReconstructionResult], None]] = None,
    on_error: Optional[Callable[[Exception], None]] = None,
) -> ColmapJob:
    """Run COLMAP pipeline in background thread.

    Args:
        image_path: Path to directory containing images
        output_path: Optional output directory (default: temp dir)
        camera_model: Camera model (OPENCV, PINHOLE, SIMPLE_RADIAL, etc.)
        match_type: Matching strategy (exhaustive, sequential, vocab_tree, spatial)
        on_progress: Callback (stage, progress, message)
        on_complete: Callback when done (result)
        on_error: Callback on error (exception)

    Returns:
        ColmapJob that can be cancelled or waited on
    """
    config = ColmapConfig(
        image_path=image_path,
        output_path=output_path,
        camera_model=camera_model,
        match_type=match_type,
    )

    job = ColmapJob(
        config=config,
        on_progress=on_progress,
        on_complete=on_complete,
        on_error=on_error,
    )
    job.start()
    return job
