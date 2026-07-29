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

from typing import Any

import numpy as np
import pytest

from dimos.control.safety.guardrail_hysteresis import RiskLevel
from dimos.control.safety.policies.frame_difference import (
    FrameDifferenceGuardrailPolicy,
    FrameDifferencePolicyConfig,
)
from dimos.control.safety.test_utils import _textured_gray_image
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat


def _policy(**overrides: Any) -> FrameDifferenceGuardrailPolicy:
    return FrameDifferenceGuardrailPolicy(FrameDifferencePolicyConfig(**overrides))


def test_measure_is_the_mean_absolute_intensity_change() -> None:
    previous = np.array([[10, 10], [10, 10]], dtype=np.uint8)
    current = np.array([[13, 10], [10, 6]], dtype=np.uint8)

    # |3| + |0| + |0| + |-4| over 4 pixels
    assert _policy()._measure(previous, current) == pytest.approx(7 / 4)


def test_signed_difference_does_not_wrap() -> None:
    # A darker current frame: uint8 subtraction would wrap 10 - 200 to a large
    # positive value; the score must reflect the true 190-per-pixel change.
    previous = np.full((4, 4), 200, dtype=np.uint8)
    current = np.full((4, 4), 10, dtype=np.uint8)

    assert _policy()._measure(previous, current) == pytest.approx(190.0)


def test_static_pair_scores_clear() -> None:
    frame = _textured_gray_image()

    result = _policy().evaluate(previous_image=frame, current_image=frame)

    assert result.level == RiskLevel.CLEAR
    assert result.score == pytest.approx(0.0)


def test_moving_pair_scores_positive() -> None:
    result = _policy().evaluate(
        previous_image=_textured_gray_image(),
        current_image=_textured_gray_image(shift_x=6),
    )

    assert result.score > 0.0


def test_quality_gating_is_inherited() -> None:
    black = Image.from_numpy(np.zeros((120, 160), dtype=np.uint8), format=ImageFormat.GRAY)

    result = _policy().evaluate(
        previous_image=black,
        current_image=_textured_gray_image(),
    )

    assert result.reason == "previous_roi_occluded"


def test_build_constructs_the_detector() -> None:
    assert isinstance(FrameDifferencePolicyConfig().build(), FrameDifferenceGuardrailPolicy)
