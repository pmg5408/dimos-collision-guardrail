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

"""Vocabulary shared by the guardrail module, its policies, and the hysteresis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from dimos.msgs.geometry_msgs.Twist import Twist


class GuardrailState(str, Enum):
    INIT = "init"
    PASS = "pass"
    CLAMP = "clamp"
    STOP_LATCHED = "stop_latched"
    SENSOR_DEGRADED = "sensor_degraded"


@dataclass
class GuardrailDecision:
    state: GuardrailState
    cmd_vel: Twist
    reason: str
    risk_score: float = 0.0
    publish_immediately: bool = False
