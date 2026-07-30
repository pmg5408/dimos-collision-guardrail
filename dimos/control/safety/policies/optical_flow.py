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

"""Collision risk from optical flow magnitude in a forward region of interest."""

from __future__ import annotations

from typing import Any, Literal, cast

import cv2
import numpy as np
from pydantic import Field

from dimos.control.safety.guardrail_policy import GuardrailPolicy
from dimos.control.safety.policies._roi_detector import (
    GrayImage,
    PolicyRoiConfig,
    RoiDetector,
)


class OpticalFlowMagnitudePolicyConfig(PolicyRoiConfig):
    """Settings for the optical-flow detector. The score is mean flow magnitude."""

    kind: Literal["optical_flow"] = "optical_flow"

    caution_score_threshold: float = Field(default=0.8, ge=0.0)
    stop_score_threshold: float = Field(default=1.5, ge=0.0)

    def build(self) -> GuardrailPolicy:
        return OpticalFlowMagnitudeGuardrailPolicy(self)


class OpticalFlowMagnitudeGuardrailPolicy(RoiDetector):
    """Scores a frame pair by the mean Farneback flow magnitude in its ROI."""

    # V1 keeps Farneback internals fixed to reduce tuning surface. Promote
    # these to config only after hardware tuning shows they need adjustment.
    _FARNEBACK_PYR_SCALE = 0.5
    _FARNEBACK_LEVELS = 3
    _FARNEBACK_WINDOW_SIZE = 15
    _FARNEBACK_ITERATIONS = 3
    _FARNEBACK_POLY_N = 5
    _FARNEBACK_POLY_SIGMA = 1.2
    _FARNEBACK_FLAGS = 0

    def _measure(self, previous_roi: GrayImage, current_roi: GrayImage) -> float:
        flow = cv2.calcOpticalFlowFarneback(
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
