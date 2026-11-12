"""Opt-in seams for third-party supervisor and robot-adapter plugins.

No provider, network client, credential, or robot SDK is bundled. Plugins are
loaded only when an explicit environment variable names a Python factory.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Protocol


class Supervisor(Protocol):
    """Optional research-only reviewer; implementations may use a local or remote service."""

    def review(self, lateral_error_mm: float) -> float | None: ...


class RobotAdapter(Protocol):
    """Research-only interface contract; no hardware adapter is implemented."""

    def observe(self) -> dict[str, float]: ...

    def apply_correction(self, correction: dict[str, float]) -> None: ...


def _load_factory(spec: str, setting_name: str) -> Any:
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError(f"{setting_name} must be written as 'module:factory'")
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise TypeError(f"{setting_name} factory {spec!r} is not callable")
    return factory()


def load_supervisor() -> Supervisor | None:
    """Load a user-supplied supervisor; the default is no supervisor."""

    spec = os.environ.get("ROBO_WELD_SUPERVISOR_PROVIDER", "").strip()
    if not spec:
        return None
    supervisor = _load_factory(spec, "ROBO_WELD_SUPERVISOR_PROVIDER")
    if not callable(getattr(supervisor, "review", None)):
        raise TypeError("configured supervisor must implement review(lateral_error_mm)")
    return supervisor


def load_robot_adapter() -> RobotAdapter | None:
    """Load an adapter only after a separate, explicit research opt-in."""

    if os.environ.get("ROBO_WELD_ENABLE_REAL_ROBOT") != "1":
        return None
    spec = os.environ.get("ROBO_WELD_ROBOT_ADAPTER_PROVIDER", "").strip()
    if not spec:
        raise RuntimeError(
            "real-robot integration is research-only; set "
            "ROBO_WELD_ROBOT_ADAPTER_PROVIDER=module:factory after an independent safety review"
        )
    return _load_factory(spec, "ROBO_WELD_ROBOT_ADAPTER_PROVIDER")
