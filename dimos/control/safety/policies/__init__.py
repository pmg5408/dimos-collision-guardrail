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

"""Collision detectors implementing the guardrail policy contract.

A detector is any class satisfying the GuardrailPolicy protocol, paired with a
PolicyConfig subclass that carries a `kind` tag and builds it. Detectors that score
a forward ROI may extend RoiDetector / PolicyRoiConfig to inherit that pipeline;
one that works differently implements the protocol directly.

To add one: give it a PolicyConfig subclass with a new `kind` and a build(), then
add that config to AnyPolicyConfig below. Nothing outside this package changes.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from dimos.control.safety.policies.frame_difference import (
    FrameDifferenceGuardrailPolicy,
    FrameDifferencePolicyConfig,
)
from dimos.control.safety.policies.optical_flow import (
    OpticalFlowMagnitudeGuardrailPolicy,
    OpticalFlowMagnitudePolicyConfig,
)

# Every detector config, tagged by `kind` so pydantic can build the right one from
# data. This is the single edit point when a detector is added.
AnyPolicyConfig = Annotated[
    OpticalFlowMagnitudePolicyConfig | FrameDifferencePolicyConfig,
    Field(discriminator="kind"),
]

__all__ = [
    "AnyPolicyConfig",
    "FrameDifferenceGuardrailPolicy",
    "FrameDifferencePolicyConfig",
    "OpticalFlowMagnitudeGuardrailPolicy",
    "OpticalFlowMagnitudePolicyConfig",
]
