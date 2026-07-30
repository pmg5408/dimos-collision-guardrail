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

"""What a collision detector must provide, and what it may report.

Implementations live in dimos.control.safety.policies. Nothing here knows how any
particular detector measures an image.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from dimos.control.safety.guardrail_hysteresis import RiskLevel
from dimos.msgs.sensor_msgs.Image import Image


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


class GuardrailPolicy(Protocol):
    """Detector contract: measure risk from a frame pair and return a Risk.

    The module validates inputs before calling evaluate, decides what the reported
    risk means for the robot's state, and builds the outgoing command. reset()
    exists for detectors that carry state across frames.
    """

    def evaluate(self, previous_image: Image, current_image: Image) -> RiskResult: ...

    def reset(self) -> None: ...


class PolicyConfig(BaseModel, ABC):
    """Settings for one detector, and the means of constructing the detector.

    A detector is chosen by configuration. Building the detector belongs here
    rather than in the module because only the config knows which detector
    its own values describe.

    Subclasses declare a `kind` Literal, which is the tag the config is selected by.
    """

    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def build(self) -> GuardrailPolicy:
        """Construct the detector these settings describe."""
