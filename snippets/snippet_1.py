# autodoc_core/ports.py
from typing import Protocol
import numpy as np

class CentroidStore(Protocol):
    def load(self, layer: str, date: str) -> np.ndarray: ...
    def put(self, layer: str, date: str, vec: np.ndarray) -> None: ...