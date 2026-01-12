# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Feature matching using pycolmap."""

from typing import Callable, Optional

import pycolmap

from .utils import ColmapConfig


def match_features(
    config: ColmapConfig,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> int:
    """Match features between image pairs.

    Args:
        config: COLMAP configuration
        progress: Optional callback (current, total, message)

    Returns:
        Number of matches
    """
    if progress:
        progress(0, 100, f"Starting {config.match_type} matching...")

    matching_options = pycolmap.FeatureMatchingOptions()

    if config.match_type == "exhaustive":
        pycolmap.match_exhaustive(
            database_path=str(config.database_path),
            matching_options=matching_options,
        )
    elif config.match_type == "sequential":
        pycolmap.match_sequential(
            database_path=str(config.database_path),
            matching_options=matching_options,
        )
    elif config.match_type == "vocab_tree":
        pycolmap.match_vocab_tree(
            database_path=str(config.database_path),
            matching_options=matching_options,
        )
    elif config.match_type == "spatial":
        pycolmap.match_spatial(
            database_path=str(config.database_path),
            matching_options=matching_options,
        )
    else:
        raise ValueError(f"Unknown match type: {config.match_type}")

    if progress:
        progress(100, 100, "Matching complete")

    db = pycolmap.Database.open(str(config.database_path))
    match_count = db.num_matches
    db.close()

    return match_count
