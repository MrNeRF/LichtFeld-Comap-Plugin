# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Utility types and helpers for COLMAP plugin."""

import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ColmapConfig:
    """Configuration for COLMAP pipeline."""

    image_path: Path
    output_path: Optional[Path] = None
    camera_mode: str = "AUTO"
    camera_model: str = "OPENCV"
    match_type: str = "exhaustive"
    max_image_size: int = 3200
    max_num_features: int = 8192

    def __post_init__(self):
        self.image_path = Path(self.image_path)
        if self.output_path is None:
            self.output_path = Path(tempfile.mkdtemp(prefix="lfs_colmap_"))
        else:
            self.output_path = Path(self.output_path)
            self.output_path.mkdir(parents=True, exist_ok=True)

    @property
    def database_path(self) -> Path:
        return self.output_path / "database.db"

    @property
    def sparse_path(self) -> Path:
        return self.output_path / "sparse"


@dataclass
class ReconstructionResult:
    """Result of COLMAP reconstruction."""

    success: bool
    sparse_path: Optional[Path] = None
    num_cameras: int = 0
    num_images: int = 0
    num_points: int = 0
    mean_reproj_error: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
