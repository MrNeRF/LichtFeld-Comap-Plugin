# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Background job runner for COLMAP pipeline."""

import threading
from enum import Enum
from typing import Callable, Optional

from .utils import ColmapConfig, ReconstructionResult
from .features import extract_features
from .matching import match_features
from .reconstruction import reconstruct


class ColmapStage(Enum):
    """Pipeline execution stage."""

    IDLE = "idle"
    EXTRACTING = "extracting"
    MATCHING = "matching"
    RECONSTRUCTING = "reconstructing"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class ColmapJob:
    """Background COLMAP job with progress tracking."""

    def __init__(
        self,
        config: ColmapConfig,
        on_progress: Optional[Callable[[str, float, str], None]] = None,
        on_complete: Optional[Callable[[ReconstructionResult], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self.config = config
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error

        self._stage = ColmapStage.IDLE
        self._progress = 0.0
        self._status = ""
        self._cancelled = False
        self._result: Optional[ReconstructionResult] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def stage(self) -> ColmapStage:
        with self._lock:
            return self._stage

    @property
    def progress(self) -> float:
        with self._lock:
            return self._progress

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def result(self) -> Optional[ReconstructionResult]:
        with self._lock:
            return self._result

    def is_running(self) -> bool:
        return self.stage in (
            ColmapStage.EXTRACTING,
            ColmapStage.MATCHING,
            ColmapStage.RECONSTRUCTING,
        )

    def cancel(self):
        with self._lock:
            self._cancelled = True

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Job already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def wait(self, timeout: Optional[float] = None) -> Optional[ReconstructionResult]:
        if self._thread:
            self._thread.join(timeout)
        return self._result

    def _update(self, stage: ColmapStage, progress: float, status: str):
        with self._lock:
            self._stage = stage
            self._progress = progress
            self._status = status

        if self.on_progress:
            self.on_progress(stage.value, progress, status)

    def _run(self):
        try:
            def check_cancelled():
                with self._lock:
                    return self._cancelled

            self._update(ColmapStage.EXTRACTING, 0.0, "Extracting features...")
            if check_cancelled():
                self._update(ColmapStage.CANCELLED, 0.0, "Cancelled")
                return

            extract_features(self.config)
            self._update(ColmapStage.EXTRACTING, 33.0, "Features extracted")

            self._update(ColmapStage.MATCHING, 33.0, "Matching features...")
            if check_cancelled():
                self._update(ColmapStage.CANCELLED, 33.0, "Cancelled")
                return

            match_features(self.config)
            self._update(ColmapStage.MATCHING, 66.0, "Matching complete")

            self._update(ColmapStage.RECONSTRUCTING, 66.0, "Reconstructing...")
            if check_cancelled():
                self._update(ColmapStage.CANCELLED, 66.0, "Cancelled")
                return

            result = reconstruct(self.config)

            with self._lock:
                self._result = result

            self._update(ColmapStage.DONE, 100.0, "Complete")

            if self.on_complete:
                self.on_complete(result)

        except Exception as e:
            self._update(ColmapStage.ERROR, self._progress, str(e))
            with self._lock:
                self._result = ReconstructionResult(success=False, error=str(e))

            if self.on_error:
                self.on_error(e)
