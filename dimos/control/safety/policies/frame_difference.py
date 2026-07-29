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

"""Collision risk from mean absolute intensity change in a forward region of interest."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from dimos.control.safety.guardrail_policy import GuardrailPolicy
from dimos.control.safety.policies._roi_detector import (
    GrayImage,
    RoiDetector,
    PolicyRoiConfig,
)


class FrameDifferencePolicyConfig(PolicyRoiConfig):
    """Settings for the frame-difference detector.

    The score is the mean absolute change in pixel intensity between the two ROIs,
    on the 0-255 scale -- so its thresholds are far larger than the flow detector's.
    """

    kind: Literal["frame_difference"] = "frame_difference"

    caution_score_threshold: float = Field(default=8.0, ge=0.0)
    stop_score_threshold: float = Field(default=20.0, ge=0.0)

    def build(self) -> GuardrailPolicy:
        return FrameDifferenceGuardrailPolicy(self)


class FrameDifferenceGuardrailPolicy(RoiDetector):
    """Scores a frame pair by the mean absolute intensity change in its ROI."""

    def _measure(self, previous_roi: GrayImage, current_roi: GrayImage) -> float:
        # int16 so the subtraction is signed; uint8 would wrap negative differences.
        difference = current_roi.astype(np.int16) - previous_roi.astype(np.int16)
        return float(np.mean(np.abs(difference)))
