"""Local stub of dimOS's ``Twist`` message so the guardrail runs standalone."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class _Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


def _to_vec3(value: _Vec3 | Iterable[float] | None) -> _Vec3:
    if value is None:
        return _Vec3()
    if isinstance(value, _Vec3):
        return _Vec3(value.x, value.y, value.z)
    x, y, z = value
    return _Vec3(float(x), float(y), float(z))


class Twist:
    """Linear/angular velocity command with mutable xyz components.

    The constructor copies its inputs into fresh ``_Vec3`` instances, so a
    decision derived from an incoming command never aliases that command.
    """

    def __init__(
        self,
        linear: _Vec3 | Iterable[float] | None = None,
        angular: _Vec3 | Iterable[float] | None = None,
    ) -> None:
        self.linear = _to_vec3(linear)
        self.angular = _to_vec3(angular)

    @classmethod
    def zero(cls) -> Twist:
        return cls()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Twist):
            return NotImplemented
        return self.linear == other.linear and self.angular == other.angular

    def __repr__(self) -> str:
        return f"Twist(linear={self.linear}, angular={self.angular})"
