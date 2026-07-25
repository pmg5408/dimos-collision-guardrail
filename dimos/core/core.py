"""Local stub of dimOS's ``rpc`` decorator so the guardrail runs standalone."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable[..., object])


def rpc(func: F) -> F:
    """Identity decorator. The real dimOS ``rpc`` exposes the method to the
    runtime's RPC layer; standalone, calling the method directly is enough."""
    return func
