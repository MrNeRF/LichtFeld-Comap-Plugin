# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""COLMAP Reconstruction Plugin for LichtFeld Studio."""

import lichtfeld as lf

from .pipeline import run_pipeline, run_pipeline_async
from .utils import ColmapConfig, ReconstructionResult
from .runner import ColmapJob, ColmapStage

_panel_class = None


def on_load():
    """Called when plugin loads."""
    global _panel_class

    from .panels.reconstruction import ColmapPanel

    _panel_class = ColmapPanel
    lf.ui.register_panel(ColmapPanel)


def on_unload():
    """Called when plugin unloads."""
    global _panel_class

    if _panel_class:
        lf.ui.unregister_panel(_panel_class)
        _panel_class = None


__all__ = [
    "run_pipeline",
    "run_pipeline_async",
    "ColmapConfig",
    "ReconstructionResult",
    "ColmapJob",
    "ColmapStage",
]
