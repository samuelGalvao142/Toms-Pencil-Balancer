import numpy as np
from src.shared import CameraObservation


class DVSLineAlgorithm:
    """Base class for DVS line estimation algorithms."""

    def update(self, events_np)-> CameraObservation | tuple[None, None]:
        raise NotImplementedError

    def reset(self):
        pass
