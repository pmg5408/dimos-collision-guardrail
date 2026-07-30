"""Local stub of dimOS's stream primitives so the guardrail runs standalone.

Provides the ``In`` / ``Out`` typed streams the module declares as class
annotations, plus a synchronous in-process ``Transport``. This mirrors how the
test suite's ``FakeTransport`` behaves, so the same module code runs unchanged
against either.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")

Subscriber = Callable[[T], object]
Unsubscribe = Callable[[], None]


class Transport(Generic[T]):
    """Interface a stream delegates delivery to. Subclassed by ``LocalTransport``
    here and by ``FakeTransport`` in the tests."""

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def broadcast(self, selfstream: _Stream[T] | None, value: T) -> None:
        raise NotImplementedError

    def subscribe(
        self,
        callback: Subscriber[T],
        selfstream: _Stream[T] | None = None,
    ) -> Unsubscribe:
        raise NotImplementedError

    def publish(self, value: T) -> None:
        """Send a value into this transport from outside any stream.

        Concrete on the base so every transport inherits it: subclasses supply
        delivery via ``broadcast`` and get ``publish`` for free. This is how a
        producer (or a test) injects a value without owning an ``Out`` stream.
        """
        self.broadcast(None, value)


class LocalTransport(Transport[T]):
    """In-process transport: delivers synchronously to every subscriber."""

    def __init__(self) -> None:
        self._subscribers: list[Subscriber[T]] = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def broadcast(self, selfstream: _Stream[T] | None, value: T) -> None:
        for callback in list(self._subscribers):
            callback(value)

    def subscribe(
        self,
        callback: Subscriber[T],
        selfstream: _Stream[T] | None = None,
    ) -> Unsubscribe:
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe


class _Stream(Generic[T]):
    def __init__(self) -> None:
        self.transport: Transport[T] = LocalTransport()

    def subscribe(self, callback: Subscriber[T]) -> Unsubscribe:
        return self.transport.subscribe(callback, self)

    def publish(self, value: T) -> None:
        self.transport.broadcast(self, value)


class In(_Stream[T]):
    """Inbound stream the module subscribes to."""


class Out(_Stream[T]):
    """Outbound stream the module publishes on."""
