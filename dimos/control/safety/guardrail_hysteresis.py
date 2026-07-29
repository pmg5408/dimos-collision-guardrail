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

"""Turns a sequence of risk observations into a guardrail state.

A policy measures collision risk in whatever units its detector produces and
reduces that measurement to an ordered RiskLevel;
the thresholds for that mapping are detector-specific and
belong to the policy. Deciding the state based on the risk is not detector-specific.
So this module consumes levels only, and never sees a threshold or a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field

from dimos.control.safety.guardrail_types import GuardrailState


class RiskLevel(IntEnum):
    """Ordered risk bands. Comparison order is meaningful."""

    CLEAR = 0
    CAUTION = 1
    STOP = 2


class HysteresisConfig(BaseModel):
    """How many consecutive observations each transition requires."""

    model_config = ConfigDict(extra="forbid")

    caution_frame_count: int = Field(default=2, ge=1)
    stop_frame_count: int = Field(default=2, ge=1)
    clear_frame_count: int = Field(default=3, ge=1)
    stop_release_frame_count: int = Field(default=2, ge=1)


@dataclass
class _ObservationCounts:
    caution: int = 0
    stop: int = 0
    clear: int = 0
    below_stop: int = 0


class RiskHysteresis:
    """Maps a run of risk observations onto PASS, CLAMP, or STOP_LATCHED."""

    def __init__(self, config: HysteresisConfig) -> None:
        self._config = config
        self._state = GuardrailState.PASS
        self._counts = _ObservationCounts()

    def observe(self, level: RiskLevel) -> GuardrailState:
        self._count(level)
        self._state = self._next_state()
        return self._state

    def reset(self) -> None:
        self._state = GuardrailState.PASS
        self._counts = _ObservationCounts()

    def _count(self, level: RiskLevel) -> None:
        counts = self._counts
        if level >= RiskLevel.STOP:
            counts.stop += 1
            counts.caution += 1
            counts.below_stop = 0
            counts.clear = 0
        elif level >= RiskLevel.CAUTION:
            counts.stop = 0
            counts.caution += 1
            counts.below_stop += 1
            counts.clear = 0
        else:
            counts.stop = 0
            counts.caution = 0
            counts.below_stop += 1
            counts.clear += 1

    def _next_state(self) -> GuardrailState:
        counts = self._counts
        config = self._config

        if self._state == GuardrailState.STOP_LATCHED:
            if counts.stop >= config.stop_frame_count:
                return GuardrailState.STOP_LATCHED

            if counts.below_stop < config.stop_release_frame_count:
                return GuardrailState.STOP_LATCHED

            if counts.clear >= config.clear_frame_count:
                return GuardrailState.PASS

            return GuardrailState.CLAMP

        if self._state == GuardrailState.CLAMP:
            if counts.stop >= config.stop_frame_count:
                return GuardrailState.STOP_LATCHED

            if counts.clear >= config.clear_frame_count:
                return GuardrailState.PASS

            return GuardrailState.CLAMP

        if counts.stop >= config.stop_frame_count:
            return GuardrailState.STOP_LATCHED

        if counts.stop > 0:
            return GuardrailState.CLAMP

        if counts.caution >= config.caution_frame_count:
            return GuardrailState.CLAMP

        return GuardrailState.PASS
