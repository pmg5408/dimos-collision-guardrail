"""Local stub of dimOS's ``Module`` base and ``ModuleConfig`` so the guardrail
runs standalone.

The one non-trivial piece is stream discovery: the module declares its streams
as bare class annotations (``color_image: In[Image]``), so the base has to find
those annotations and bind a live ``In`` / ``Out`` instance for each. Because the
module uses ``from __future__ import annotations``, the annotations are strings
at runtime and must be resolved with ``typing.get_type_hints`` before their
origin can be inspected.
"""

from __future__ import annotations

import typing
from typing import Any, ClassVar, Generic, TypeVar, get_origin

from pydantic import BaseModel
from reactivex.disposable import CompositeDisposable

from dimos.core.stream import In, Out, _Stream


class ModuleConfig(BaseModel):
    """Base config. Subclasses add pydantic fields and cross-field validators."""


ConfigT = TypeVar("ConfigT", bound=ModuleConfig)


class Module(Generic[ConfigT]):
    default_config: ClassVar[type[ModuleConfig]]

    def __init__(self, **kwargs: Any) -> None:
        self.config: ConfigT = type(self).default_config(**kwargs)  # type: ignore[assignment]
        self._disposables = CompositeDisposable()
        self._stream_names: list[str] = []
        self._bind_streams()

    def _bind_streams(self) -> None:
        for name, hint in typing.get_type_hints(type(self)).items():
            origin = get_origin(hint)
            if origin in (In, Out):
                setattr(self, name, origin())
                self._stream_names.append(name)

    def _streams(self) -> list[_Stream[Any]]:
        return [getattr(self, name) for name in self._stream_names]

    def start(self) -> None:
        for stream in self._streams():
            stream.transport.start()

    def stop(self) -> None:
        for stream in self._streams():
            stream.transport.stop()

    def _close_module(self) -> None:
        self._disposables.dispose()
        for stream in self._streams():
            stream.transport.stop()

    @classmethod
    def blueprint(cls) -> type[Module[ConfigT]]:
        """No-op stand-in. The real dimOS ``blueprint`` builds a wiring spec for
        the module graph; nothing here needs that to run or be tested."""
        return cls
