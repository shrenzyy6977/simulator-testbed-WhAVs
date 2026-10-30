"""Path-tracking controllers. All share the Controller interface and the SpeedPID."""
from .base import Controller, SpeedPID
from .geometric import PurePursuit, Stanley
from .optimal import LQR, LinearMPC

CONTROLLERS = {c.name: c for c in (PurePursuit, Stanley, LQR, LinearMPC)}


def make_controller(name: str, speed: dict | None = None, **kw) -> Controller:
    try:
        cls = CONTROLLERS[name]
    except KeyError:
        raise ValueError(f"Unknown controller '{name}'. Options: {list(CONTROLLERS)}") from None
    return cls(speed=SpeedPID(**(speed or {})), **kw)


__all__ = ["Controller", "SpeedPID", "PurePursuit", "Stanley", "LQR", "LinearMPC", "CONTROLLERS", "make_controller"]
