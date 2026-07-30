# Copyright 2025-2026 Dimensional Inc.
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
from pydantic import ValidationError

from dimos.control.safety.guardrail_hysteresis import RiskLevel
from dimos.control.safety.guardrail_policy import (
    RiskAssessment,
    RiskUnavailable,
)
from dimos.control.safety.policies import (
    OpticalFlowMagnitudeGuardrailPolicy,
    OpticalFlowMagnitudePolicyConfig,
)
from dimos.control.safety.test_utils import (
    _textured_gray_image,
)
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat


def _policy(**overrides: Any) -> OpticalFlowMagnitudeGuardrailPolicy:
    return OpticalFlowMagnitudeGuardrailPolicy(OpticalFlowMagnitudePolicyConfig(**overrides))


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        pytest.param(
            {"forward_roi_top_fraction": 0.9, "forward_roi_bottom_fraction": 0.5},
            "forward_roi_top_fraction",
            id="inverted_roi_bounds",
        ),
        pytest.param(
            {"occlusion_dark_pixel_threshold": 240, "occlusion_bright_pixel_threshold": 10},
            "occlusion_dark_pixel_threshold",
            id="inverted_occlusion_thresholds",
        ),
        pytest.param(
            {"caution_score_threshold": 2.0, "stop_score_threshold": 1.0},
            "caution_score_threshold",
            id="caution_above_stop",
        ),
    ],
)
def test_contradictory_settings_are_rejected(
    overrides: dict[str, Any],
    expected_message: str,
) -> None:
    with pytest.raises(ValidationError, match=expected_message):
        OpticalFlowMagnitudePolicyConfig(**overrides)


def test_unknown_setting_is_rejected() -> None:
    with pytest.raises(ValidationError, match="stop_flow_magnitude_threshhold"):
        OpticalFlowMagnitudePolicyConfig.model_validate(
            {"stop_flow_magnitude_threshhold": 1.5}
        )


def _uniform_gray_image(value: int, *, width: int = 160, height: int = 120) -> Image:
    return Image.from_numpy(
        np.full((height, width), value, dtype=np.uint8),
        format=ImageFormat.GRAY,
    )


@pytest.fixture
def image_pair() -> tuple[Image, Image]:
    return (_textured_gray_image(), _textured_gray_image(shift_x=3))


def test_usable_frame_pair_returns_an_assessment(image_pair: tuple[Image, Image]) -> None:
    result = _policy().evaluate(
        previous_image=image_pair[0],
        current_image=image_pair[1],
    )

    assert isinstance(result, RiskAssessment)
    assert result.score > 0.0


def test_static_pair_is_assessed_as_clear() -> None:
    frame = _textured_gray_image()

    result = _policy().evaluate(previous_image=frame, current_image=frame)

    assert isinstance(result, RiskAssessment)
    assert result.level == RiskLevel.CLEAR


@pytest.mark.parametrize(
    ("bad_frame_position", "bad_frame_kind", "expected_reason"),
    [
        pytest.param("previous", "black", "previous_roi_occluded", id="previous_black_occluded"),
        pytest.param("current", "black", "current_roi_occluded", id="current_black_occluded"),
        pytest.param("previous", "white", "previous_roi_occluded", id="previous_white_occluded"),
        pytest.param("current", "white", "current_roi_occluded", id="current_white_occluded"),
        pytest.param(
            "previous",
            "uniform_gray",
            "previous_roi_low_texture",
            id="previous_low_texture",
        ),
        pytest.param(
            "current",
            "uniform_gray",
            "current_roi_low_texture",
            id="current_low_texture",
        ),
    ],
)
def test_unusable_roi_reports_risk_unavailable(
    image_pair: tuple[Image, Image],
    bad_frame_position: str,
    bad_frame_kind: str,
    expected_reason: str,
) -> None:
    if bad_frame_kind == "black":
        bad_image = _uniform_gray_image(0)
    elif bad_frame_kind == "white":
        bad_image = _uniform_gray_image(255)
    else:
        bad_image = _uniform_gray_image(127)

    previous_image, current_image = image_pair
    if bad_frame_position == "previous":
        previous_image = bad_image
    else:
        current_image = bad_image

    result = _policy().evaluate(
        previous_image=previous_image,
        current_image=current_image,
    )

    assert isinstance(result, RiskUnavailable)
    assert result.reason == expected_reason


@pytest.mark.parametrize(
    ("mean_flow_magnitude", "expected_level"),
    [
        pytest.param(0.0, RiskLevel.CLEAR, id="zero_flow"),
        pytest.param(0.79, RiskLevel.CLEAR, id="just_below_caution"),
        pytest.param(0.8, RiskLevel.CAUTION, id="at_caution_threshold"),
        pytest.param(1.49, RiskLevel.CAUTION, id="just_below_stop"),
        pytest.param(1.5, RiskLevel.STOP, id="at_stop_threshold"),
        pytest.param(3.0, RiskLevel.STOP, id="well_above_stop"),
    ],
)
def test_flow_magnitude_maps_to_risk_level(
    mean_flow_magnitude: float,
    expected_level: RiskLevel,
) -> None:
    assert _policy()._risk_level(mean_flow_magnitude) == expected_level


def test_assessment_score_is_the_measured_flow_magnitude(
    image_pair: tuple[Image, Image],
    mocker,
) -> None:
    policy = _policy()
    mocker.patch.object(policy, "_measure", return_value=1.75)

    result = policy.evaluate(
        previous_image=image_pair[0],
        current_image=image_pair[1],
    )

    assert isinstance(result, RiskAssessment)
    assert result.score == pytest.approx(1.75)
    assert result.level == RiskLevel.STOP
