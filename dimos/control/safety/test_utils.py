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

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from dimos.control.safety.guardrail_hysteresis import RiskLevel
from dimos.control.safety.guardrail_policy import RiskAssessment, RiskResult
from dimos.core.stream import Transport, _Stream
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat

T = TypeVar("T")


class FakeTransport(Transport[T]):
    def __init__(self) -> None:
        self._subscribers: list[Callable[[T], Any]] = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def broadcast(self, selfstream: _Stream[T] | None, value: T) -> None:
        for callback in list(self._subscribers):
            callback(value)

    def subscribe(
        self,
        callback: Callable[[T], Any],
        selfstream: _Stream[T] | None = None,
    ) -> Callable[[], None]:
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            self._subscribers.remove(callback)

        return unsubscribe


class SequencePolicy:
    """Replays risk readings, repeating the last one once exhausted."""

    def __init__(self, results: list[RiskResult]) -> None:
        self._results = results
        self._index = 0

    def evaluate(self, previous_image: Image, current_image: Image) -> RiskResult:
        if self._index < len(self._results):
            result = self._results[self._index]
            self._index += 1
            return result
        return self._results[-1]

    def reset(self) -> None:
        self._index = 0


def _textured_gray_image(*, width: int = 160, height: int = 120, shift_x: int = 0) -> Image:
    yy, xx = np.indices((height, width))
    pattern = ((xx * 5 + yy * 9 + shift_x * 17) % 256).astype(np.uint8)
    return Image.from_numpy(pattern, format=ImageFormat.GRAY)


def _cmd(
    x: float = 0.35,
    *,
    linear_y: float = 0.0,
    angular_z: float = 0.25,
) -> Twist:
    return Twist(
        linear=[x, linear_y, 0.0],
        angular=[0.0, 0.0, angular_z],
    )


def _assessment(level: RiskLevel, score: float = 0.0) -> RiskAssessment:
    return RiskAssessment(level=level, score=score)


