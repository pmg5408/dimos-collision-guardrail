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

from dimos.control.safety.guardrail_hysteresis import (
    HysteresisConfig,
    RiskHysteresis,
    RiskLevel,
)
from dimos.control.safety.guardrail_types import GuardrailState

CLEAR = RiskLevel.CLEAR
CAUTION = RiskLevel.CAUTION
STOP = RiskLevel.STOP


def _hysteresis(
    *,
    caution_frame_count: int = 2,
    stop_frame_count: int = 2,
    clear_frame_count: int = 3,
    stop_release_frame_count: int = 2,
) -> RiskHysteresis:
    return RiskHysteresis(
        HysteresisConfig(
            caution_frame_count=caution_frame_count,
            stop_frame_count=stop_frame_count,
            clear_frame_count=clear_frame_count,
            stop_release_frame_count=stop_release_frame_count,
        )
    )


def _observe_all(hysteresis: RiskHysteresis, levels: list[RiskLevel]) -> list[GuardrailState]:
    return [hysteresis.observe(level) for level in levels]


def test_clear_observations_stay_in_pass() -> None:
    assert _observe_all(_hysteresis(), [CLEAR, CLEAR, CLEAR]) == [
        GuardrailState.PASS,
        GuardrailState.PASS,
        GuardrailState.PASS,
    ]


def test_consecutive_caution_observations_reach_clamp() -> None:
    assert _observe_all(_hysteresis(caution_frame_count=2), [CAUTION, CAUTION]) == [
        GuardrailState.PASS,
        GuardrailState.CLAMP,
    ]


def test_first_stop_observation_clamps_immediately() -> None:
    assert _observe_all(_hysteresis(stop_frame_count=2), [STOP]) == [GuardrailState.CLAMP]


def test_consecutive_stop_observations_latch() -> None:
    assert _observe_all(_hysteresis(stop_frame_count=2), [STOP, STOP]) == [
        GuardrailState.CLAMP,
        GuardrailState.STOP_LATCHED,
    ]


def test_latched_stop_holds_through_first_clear_observation() -> None:
    assert _observe_all(_hysteresis(), [STOP, STOP, CLEAR]) == [
        GuardrailState.CLAMP,
        GuardrailState.STOP_LATCHED,
        GuardrailState.STOP_LATCHED,
    ]


def test_latched_stop_release_requires_a_run_of_clear_observations() -> None:
    states = _observe_all(
        _hysteresis(clear_frame_count=3),
        [STOP, STOP, CLEAR, CLEAR, CLEAR],
    )

    assert states[0] == GuardrailState.CLAMP
    assert states[1] == GuardrailState.STOP_LATCHED
    assert states[2] != GuardrailState.PASS
    assert states[3] != GuardrailState.PASS
    assert states[4] == GuardrailState.PASS


def test_clamp_returns_to_pass_after_a_run_of_clear_observations() -> None:
    assert _observe_all(_hysteresis(), [CAUTION, CAUTION, CLEAR, CLEAR, CLEAR]) == [
        GuardrailState.PASS,
        GuardrailState.CLAMP,
        GuardrailState.CLAMP,
        GuardrailState.CLAMP,
        GuardrailState.PASS,
    ]


def test_caution_observation_interrupts_a_clear_run() -> None:
    states = _observe_all(
        _hysteresis(clear_frame_count=3),
        [STOP, STOP, CLEAR, CLEAR, CAUTION, CLEAR, CLEAR],
    )

    assert states[1] == GuardrailState.STOP_LATCHED
    assert states[4] == GuardrailState.CLAMP
    assert states[5] == GuardrailState.CLAMP
    assert states[6] == GuardrailState.CLAMP


def test_reset_clears_the_latch_and_the_counters() -> None:
    hysteresis = _hysteresis(stop_frame_count=2)
    _observe_all(hysteresis, [STOP, STOP])

    hysteresis.reset()

    assert hysteresis.observe(STOP) == GuardrailState.CLAMP
