# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""COLMAP Reconstruction Panel."""

import lichtfeld as lf

from ..pipeline import run_pipeline_async
from ..utils import ReconstructionResult


class ColmapPanel:
    """GUI panel for COLMAP reconstruction workflow."""

    panel_label = "COLMAP Reconstruction"
    panel_space = "SIDE_PANEL"
    panel_order = 5

    def __init__(self):
        self.job = None
        self.image_path = ""
        self.camera_model_idx = 1
        self.match_type_idx = 0
        self.last_result = None

        self.camera_models = ["PINHOLE", "OPENCV", "SIMPLE_RADIAL", "RADIAL"]
        self.match_types = ["exhaustive", "sequential", "vocab_tree", "spatial"]

    def draw(self, layout):
        layout.label("Image Folder:")
        changed, self.image_path = layout.input_text("##imgpath", self.image_path)

        layout.separator()

        if layout.collapsing_header("Settings", default_open=False):
            _, self.camera_model_idx = layout.combo(
                "Camera Model", self.camera_model_idx, self.camera_models
            )
            _, self.match_type_idx = layout.combo(
                "Matching", self.match_type_idx, self.match_types
            )

        layout.separator()

        if self.job and self.job.is_running():
            stage = self.job.stage.value
            progress = self.job.progress

            layout.label(f"Stage: {stage}")
            layout.progress_bar(progress / 100.0, self.job.status)

            if layout.button("Cancel"):
                self.job.cancel()
        else:
            if layout.button("Start Reconstruction", (0, 36)):
                self._start()

        if self.last_result and self.last_result.success:
            layout.separator()
            layout.heading("Results")
            layout.label(f"Images: {self.last_result.num_images}")
            layout.label(f"Points: {self.last_result.num_points}")
            layout.label(f"Error: {self.last_result.mean_reproj_error:.3f}")

            if layout.button("Import to Scene"):
                self._import_scene()

        if self.last_result and not self.last_result.success:
            layout.separator()
            layout.text_colored(f"Error: {self.last_result.error}", (1.0, 0.3, 0.3, 1.0))

    def _start(self):
        if not self.image_path:
            return

        self.last_result = None
        self.job = run_pipeline_async(
            image_path=self.image_path,
            camera_model=self.camera_models[self.camera_model_idx],
            match_type=self.match_types[self.match_type_idx],
            on_complete=self._on_complete,
            on_error=self._on_error,
        )

    def _on_complete(self, result):
        self.last_result = result

    def _on_error(self, error):
        self.last_result = ReconstructionResult(success=False, error=str(error))

    def _import_scene(self):
        if self.last_result and self.last_result.sparse_path:
            lf.io.load(str(self.last_result.sparse_path.parent))
