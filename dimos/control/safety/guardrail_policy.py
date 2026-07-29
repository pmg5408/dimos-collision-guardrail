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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Self, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dimos.control.safety.guardrail_hysteresis import RiskLevel
from dimos.msgs.sensor_msgs.Image import Image

GrayImage = NDArray[np.uint8]


@dataclass(frozen=True)
class RiskAssessment:
    """A usable measurement: how alarmed the detector is, and its raw score."""

    level: RiskLevel
    score: float


@dataclass(frozen=True)
class RiskUnavailable:
    """The detector cannot measure this frame pair, with a named cause."""

    reason: str


RiskResult = RiskAssessment | RiskUnavailable


class OpticalFlowMagnitudePolicyConfig(BaseModel):
    """How the optical-flow detector reads a frame pair and scores it."""

    model_config = ConfigDict(extra="forbid")

    # Preprocessing
    flow_downsample_width_px: int = Field(default=160, ge=32)

    # Forward ROI geometry
    forward_roi_top_fraction: float = Field(default=0.45, ge=0.0, le=1.0)
    forward_roi_bottom_fraction: float = Field(default=0.95, ge=0.0, le=1.0)
    forward_roi_width_fraction: float = Field(default=0.5, gt=0.0, le=1.0)

    # Whether the ROI is measurable at all
    low_texture_variance_threshold: float = Field(default=150.0, ge=0.0)
    occlusion_dark_pixel_threshold: int = Field(default=20, ge=0, le=255)
    occlusion_bright_pixel_threshold: int = Field(default=235, ge=0, le=255)
    occlusion_extreme_fraction_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    # Flow magnitude to risk band
    caution_flow_magnitude_threshold: float = Field(default=0.8, ge=0.0)
    stop_flow_magnitude_threshold: float = Field(default=1.5, ge=0.0)

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

        if self.caution_flow_magnitude_threshold > self.stop_flow_magnitude_threshold:
            raise ValueError(
                "caution_flow_magnitude_threshold must be less than or equal to "
                "stop_flow_magnitude_threshold"
            )

        return self

    def build(self) -> GuardrailPolicy:
        return OpticalFlowMagnitudeGuardrailPolicy(self)


class GuardrailPolicy(Protocol):
    """Detector contract: measure risk from a frame pair, nothing more.

    The module validates inputs before calling evaluate, decides what the reported
    risk means for the robot's state, and builds the outgoing command. reset()
    exists for detectors that carry state across frames.
    """

    def evaluate(self, previous_image: Image, current_image: Image) -> RiskResult: ...

    def reset(self) -> None: ...


class OpticalFlowMagnitudeGuardrailPolicy(GuardrailPolicy):
    """Forward-motion RGB guardrail using flow magnitude in a central lower ROI."""

    # V1 keeps Farneback internals fixed to reduce tuning surface. Promote
    # these to config only after hardware tuning shows they need adjustment.
    _FARNEBACK_PYR_SCALE = 0.5
    _FARNEBACK_LEVELS = 3
    _FARNEBACK_WINDOW_SIZE = 15
    _FARNEBACK_ITERATIONS = 3
    _FARNEBACK_POLY_N = 5
    _FARNEBACK_POLY_SIGMA = 1.2
    _FARNEBACK_FLAGS = 0

    def __init__(self, config: OpticalFlowMagnitudePolicyConfig) -> None:
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

        mean_flow_magnitude = self._mean_flow_magnitude(previous_roi, current_roi)
        return RiskAssessment(
            level=self._risk_level(mean_flow_magnitude),
            score=mean_flow_magnitude,
        )

    def reset(self) -> None:
        """No-op: this detector carries no state between frame pairs."""

    def _to_resized_gray(self, image: Image) -> GrayImage:
        gray = cast("GrayImage", image.to_grayscale().data)
        if gray.dtype != np.uint8:
            gray = cast("GrayImage", cv2.convertScaleAbs(gray))

        height, width = gray.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Image has invalid dimensions")

        target_width = min(width, self._config.flow_downsample_width_px)
        if target_width == width:
            return cast("GrayImage", np.ascontiguousarray(gray))

        scale = target_width / float(width)
        target_height = max(round(height * scale), 2)
        resized = cv2.resize(  # type: ignore[call-overload]
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

    def _mean_flow_magnitude(self, previous_roi: GrayImage, current_roi: GrayImage) -> float:
        flow = cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]
            previous_roi,
            current_roi,
            cast("Any", None),
            self._FARNEBACK_PYR_SCALE,
            self._FARNEBACK_LEVELS,
            self._FARNEBACK_WINDOW_SIZE,
            self._FARNEBACK_ITERATIONS,
            self._FARNEBACK_POLY_N,
            self._FARNEBACK_POLY_SIGMA,
            self._FARNEBACK_FLAGS,
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(magnitude))

    def _risk_level(self, mean_flow_magnitude: float) -> RiskLevel:
        if mean_flow_magnitude >= self._config.stop_flow_magnitude_threshold:
            return RiskLevel.STOP

        if mean_flow_magnitude >= self._config.caution_flow_magnitude_threshold:
            return RiskLevel.CAUTION

        return RiskLevel.CLEAR

