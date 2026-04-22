from dataclasses import dataclass

from src.shared import (
    ControlInput,
    State,
)

from .base import BaseController


@dataclass
class SimpleControllerParams:
    g_p: float
    g_a: float
    g_d: float


SIMPLE_CONTROLLER_PRESETS = {
    "default": {
        "g_p": 1.25,
        "g_a": 0.8,
        "g_d": 0.0,
    },
    "pos": {
        "base": "default",
        "g_p": 1.0,
        "g_a": 0.0,
        "g_d": 0.0,
    },
    "tilt": {
        "base": "default",
        "g_p": 0.0,
        "g_a": 0.5,
        "g_d": 0.0,
    },
    "conradt": {
        "g_p": 1.5,
        "g_a": 2.50,
        "g_d": 0.0,
    },
}


class SimpleController(BaseController):
    """Direct stateless controller for independent x/y axes."""

    def __init__(self, params: SimpleControllerParams):
        self._params = params

    def _axis_command(self, pos: float, alpha: float, vel: float) -> float:
        return float(
            self._params.g_p * pos
            + self._params.g_a * alpha
            + self._params.g_d * vel
        )

    def compute(self, state: State) -> ControlInput:
        x_des = self._axis_command(state.px, state.ax, state.vx)
        y_des = self._axis_command(state.py, state.ay, state.vy)
        return ControlInput(x_des, y_des)

    def reference_command(self) -> ControlInput:
        return ControlInput(0.0, 0.0)

    def set_applied_command(self, u: ControlInput, state: State) -> None:
        del u, state

    def reset(self, x_hat: State | None = None):
        del x_hat
