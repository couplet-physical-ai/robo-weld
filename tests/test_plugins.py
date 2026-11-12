import os
import sys
import types
import unittest
from unittest.mock import patch

from robo_weld.cpu_demo import run_demo
from robo_weld.plugins import load_robot_adapter, load_supervisor


class PluginTests(unittest.TestCase):
    def test_optional_supervisor_is_disabled_without_provider(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(load_supervisor())

    def test_real_robot_adapter_is_disabled_by_default(self):
        with patch.dict(os.environ, {"ROBO_WELD_ROBOT_ADAPTER_PROVIDER": "missing:factory"}, clear=True):
            self.assertIsNone(load_robot_adapter())

    def test_real_robot_requires_explicit_provider_after_opt_in(self):
        with patch.dict(os.environ, {"ROBO_WELD_ENABLE_REAL_ROBOT": "1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "research-only"):
                load_robot_adapter()

    def test_explicit_provider_can_review_the_dagger_loop(self):
        class DemoSupervisor:
            def review(self, lateral_error_mm):
                return lateral_error_mm if abs(lateral_error_mm) >= 0.45 else None

        module = types.ModuleType("demo_supervisor_plugin")
        module.make = DemoSupervisor
        with patch.dict(sys.modules, {module.__name__: module}):
            with patch.dict(
                os.environ,
                {"ROBO_WELD_SUPERVISOR_PROVIDER": "demo_supervisor_plugin:make"},
                clear=True,
            ):
                provider = load_supervisor()
                result = run_demo(seed=7, rounds=2, episodes_per_round=3, steps=24, supervisor=provider)

        self.assertEqual(result["supervisor_mode"], "configured-provider")
        self.assertEqual(result["synthetic_critic_interventions"], 0)
        self.assertGreater(result["configured_provider_interventions"], 0)


if __name__ == "__main__":
    unittest.main()
