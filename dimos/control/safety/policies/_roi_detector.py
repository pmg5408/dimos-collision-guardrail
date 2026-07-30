# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared base for detectors that score a forward region of interest.

Such a detector reduces the ROI pair to a single score; the same caution/stop
bands turn that score into a RiskLevel. Grayscale conversion, downsampling, ROI
extraction, and quality gating are identical across them and only the measurement
differs, so only `_measure` is left to subclasses.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Self, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from dimos.control.safety.guardrail_hysteresis import RiskLevel
from dimos.control.safety.guardrail_policy import (
    GuardrailPolicy,
    PolicyConfig,
    RiskAssessment,
    RiskResult,
    RiskUnavailable,
)
from dimos.msgs.sensor_msgs.Image import Image

GrayImage = NDArray[np.uint8]


class PolicyRoiConfig(PolicyConfig):
    """Settings shared by every forward-ROI detector.

    Subclasses add a `kind`, a `build`, and their own defaults for the two score
    thresholds. The score's units differ per detector, the mapping does not.
    """

    # Preprocessing
    downsample_width_px: int = Field(default=160, ge=32)

    # Forward ROI geometry
    forward_roi_top_fraction: float = Field(default=0.45, ge=0.0, le=1.0)
    forward_roi_bottom_fraction: float = Field(default=0.95, ge=0.0, le=1.0)
    forward_roi_width_fraction: float = Field(default=0.5, gt=0.0, le=1.0)

    # Whether the ROI is measurable at all
    low_texture_variance_threshold: float = Field(default=150.0, ge=0.0)
    occlusion_dark_pixel_threshold: int = Field(default=20, ge=0, le=255)
    occlusion_bright_pixel_threshold: int = Field(default=235, ge=0, le=255)
    occlusion_extreme_fraction_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    # Score to risk band. The score is the detector's raw measurement
    # (RiskAssessment.score); its scale is detector-specific, so subclasses set the
    # defaults.
    caution_score_threshold: float = Field(ge=0.0)
    stop_score_threshold: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        if self.forward_roi_top_fraction >= self.forward_roi_bottom_fraction:
            raise ValueError(
                "forward_roi_top_fraction must be less than forward_roi_bottom_fraction"
            )

        if self.occlusion_dark_pixel_threshold >= self.occlusion_bright_pixel_threshold:
            raise ValueError(
                "occlusion_dark_pixel_threshold must be less than occlusion_bright_pixel_threshold"
            )

        if self.caution_score_threshold > self.stop_score_threshold:
            raise ValueError(
                "caution_score_threshold must be less than or equal to stop_score_threshold"
            )

        return self


class RoiDetector(GuardrailPolicy):
    """Extracts a forward ROI pair, gates its quality, and scores it.

    Subclasses implement `_measure`; everything else including preprocessing, quality
    gating, and turning the score into a risk band is shared.
    """

    def __init__(self, config: PolicyRoiConfig) -> None:
        self._config = config

    def evaluate(self, previous_image: Image, current_image: Image) -> RiskResult:
        previous_gray = self._to_resized_gray(previous_image)
        current_gray = self._to_resized_gray(current_image)

        previous_roi, current_roi = self._extract_forward_rois(previous_gray, current_gray)

        if previous_roi.size == 0 or current_roi.size == 0:
            return RiskUnavailable("invalid_forward_roi")

        quality_failure_reason = self._quality_failure_reason(previous_roi, current_roi)
        if quality_failure_reason is not None:
            return RiskUnavailable(quality_failure_reason)

        score = self._measure(previous_roi, current_roi)
        return RiskAssessment(level=self._risk_level(score), score=score)

    def reset(self) -> None:
        """No-op: forward-ROI detectors carry no state between frame pairs."""

    @abstractmethod
    def _measure(self, previous_roi: GrayImage, current_roi: GrayImage) -> float:
        """Reduce the ROI pair to a single collision-risk score."""

    def _risk_level(self, score: float) -> RiskLevel:
        if score >= self._config.stop_score_threshold:
            return RiskLevel.STOP

        if score >= self._config.caution_score_threshold:
            return RiskLevel.CAUTION

        return RiskLevel.CLEAR

    def _to_resized_gray(self, image: Image) -> GrayImage:
        gray = cast("GrayImage", image.to_grayscale().data)
        if gray.dtype != np.uint8:
            gray = cast("GrayImage", cv2.convertScaleAbs(gray))

        height, width = gray.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Image has invalid dimensions")

        target_width = min(width, self._config.downsample_width_px)
        if target_width == width:
            return cast("GrayImage", np.ascontiguousarray(gray))

        scale = target_width / float(width)
        target_height = max(round(height * scale), 2)
        resized = cv2.resize(
            gray,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        return cast("GrayImage", np.ascontiguousarray(resized))

    def _extract_forward_rois(
        self,
        previous_gray: GrayImage,
        current_gray: GrayImage,
    ) -> tuple[GrayImage, GrayImage]:
        height, width = current_gray.shape
        x0, x1, y0, y1 = self._forward_roi_bounds(width=width, height=height)

        return (
            np.ascontiguousarray(previous_gray[y0:y1, x0:x1]),
            np.ascontiguousarray(current_gray[y0:y1, x0:x1]),
        )

    def _quality_failure_reason(
        self,
        previous_roi: GrayImage,
        current_roi: GrayImage,
    ) -> str | None:
        if self._is_occluded(previous_roi):
            return "previous_roi_occluded"

        if self._is_occluded(current_roi):
            return "current_roi_occluded"

        if self._is_low_texture(previous_roi):
            return "previous_roi_low_texture"

        if self._is_low_texture(current_roi):
            return "current_roi_low_texture"

        return None

    def _forward_roi_bounds(self, *, width: int, height: int) -> tuple[int, int, int, int]:
        roi_width = max(round(width * self._config.forward_roi_width_fraction), 2)
        x0 = max((width - roi_width) // 2, 0)
        x1 = min(x0 + roi_width, width)

        y0 = min(max(round(height * self._config.forward_roi_top_fraction), 0), height - 1)
        y1 = min(max(round(height * self._config.forward_roi_bottom_fraction), y0 + 1), height)

        return x0, x1, y0, y1

    def _is_low_texture(self, roi: GrayImage) -> bool:
        return float(np.var(roi)) < self._config.low_texture_variance_threshold

    def _is_occluded(self, roi: GrayImage) -> bool:
        dark_fraction = float(np.mean(roi <= self._config.occlusion_dark_pixel_threshold))
        bright_fraction = float(np.mean(roi >= self._config.occlusion_bright_pixel_threshold))
        return (
            max(dark_fraction, bright_fraction) >= self._config.occlusion_extreme_fraction_threshold
        )
