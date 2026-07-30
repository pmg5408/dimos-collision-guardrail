"""Local stub of dimOS's ``Image`` message so the guardrail runs standalone."""

from __future__ import annotations

from enum import Enum
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


class ImageFormat(str, Enum):
    GRAY = "gray"
    RGB = "rgb"
    BGR = "bgr"


class Image:
    """Thin wrapper over a numpy pixel buffer plus its color format.
    """

    def __init__(self, data: NDArray[Any], format: ImageFormat) -> None:
        self._data = np.array(data)
        self._format = format

    @classmethod
    def from_numpy(cls, data: NDArray[Any], format: ImageFormat = ImageFormat.RGB) -> Image:
        return cls(data, format)

    @property
    def data(self) -> NDArray[Any]:
        return self._data

    @property
    def format(self) -> ImageFormat:
        return self._format

    def to_grayscale(self) -> Image:
        if self._format == ImageFormat.GRAY or self._data.ndim == 2:
            return Image(self._data, ImageFormat.GRAY)
        code = cv2.COLOR_RGB2GRAY if self._format == ImageFormat.RGB else cv2.COLOR_BGR2GRAY
        return Image(cv2.cvtColor(self._data, code), ImageFormat.GRAY)
