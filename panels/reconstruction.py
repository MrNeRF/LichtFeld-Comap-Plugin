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
        self.output_path = ""
        self.camera_model_idx = 1
        self.match_type_idx = 0
        self.last_result = None
        self._pending_import = None  # Path to import on main thread

        self.camera_models = ["PINHOLE", "OPENCV", "SIMPLE_RADIAL", "RADIAL"]
        self.match_types = ["exhaustive", "sequential", "vocab_tree", "spatial"]

    def draw(self, layout):
        # Check for pending import (must happen on main thread)
        if self._pending_import:
            path = self._pending_import
            self._pending_import = None
            lf.log.info(f"Executing deferred import on main thread: {path}")
            lf.app.open(path)

        layout.label("Image Folder:")
        _, self.image_path = layout.path_input("##imgpath", self.image_path, True, "Select Image Folder")

        layout.label("Output Folder:")
        _, self.output_path = layout.path_input("##outpath", self.output_path, True, "Select Output Folder")

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
            layout.label(f"Reproj Error: {self.last_result.mean_reproj_error:.3f}")

            if layout.button("Import to Scene##colmap_import", (0, 36)):
                lf.log.info("Import to Scene button CLICKED!")
                self._import_scene()

        if self.last_result and not self.last_result.success:
            layout.separator()
            layout.label("Error:")
            layout.text_selectable(self.last_result.error or "Unknown error", 60)

    def _start(self):
        if not self.image_path:
            return

        self.last_result = None
        self.job = run_pipeline_async(
            image_path=self.image_path,
            output_path=self.output_path if self.output_path else None,
            camera_model=self.camera_models[self.camera_model_idx],
            match_type=self.match_types[self.match_type_idx],
            on_complete=self._on_complete,
            on_error=self._on_error,
        )

    def _on_complete(self, result):
        import lichtfeld as lf
        lf.log.info(f"on_complete called! result.success={result.success}, undist={result.undistorted_path}")
        self.last_result = result

        # Queue import for main thread (via draw())
        if result.undistorted_path:
            lf.log.info(f"Queueing import: {result.undistorted_path}")
            self._pending_import = str(result.undistorted_path)
        elif result.sparse_path:
            lf.log.info(f"Queueing import sparse: {result.sparse_path.parent}")
            self._pending_import = str(result.sparse_path.parent)

    def _on_error(self, error):
        self.last_result = ReconstructionResult(success=False, error=str(error))

    def _import_scene(self):
        lf.log.info(f"_import_scene called, last_result={self.last_result}")
        if self.last_result:
            lf.log.info(f"undistorted_path={self.last_result.undistorted_path}")
            lf.log.info(f"sparse_path={self.last_result.sparse_path}")

        if self.last_result and self.last_result.undistorted_path:
            path = str(self.last_result.undistorted_path)
            lf.log.info(f"Opening undistorted path: {path}")
            lf.app.open(path)
        elif self.last_result and self.last_result.sparse_path:
            path = str(self.last_result.sparse_path.parent)
            lf.log.info(f"Opening sparse path parent: {path}")
            lf.app.open(path)
        else:
            lf.log.warn("No path available to import")
