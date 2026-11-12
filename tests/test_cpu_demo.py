import unittest

from robo_weld.cpu_demo import run_demo


class CpuDemoTests(unittest.TestCase):
    def test_default_demo_is_deterministic_and_reports_synthetic_interventions(self):
        first = run_demo(seed=7, rounds=4, episodes_per_round=5, steps=40)
        second = run_demo(seed=7, rounds=4, episodes_per_round=5, steps=40)

        self.assertEqual(first, second)
        self.assertGreater(first["synthetic_critic_interventions"], 0)
        self.assertEqual(first["synthetic_critic_interventions"], first["dagger_labels"])
        self.assertIn("not weld quality", first["interpretation"])

    def test_dagger_refinement_improves_synthetic_lateral_tracking_metric(self):
        result = run_demo(seed=11, rounds=5, episodes_per_round=8, steps=48)
        self.assertGreater(result["learned_gain"], 0)
        self.assertLess(
            result["synthetic_lateral_tracking_mean_abs_mm_after"],
            result["synthetic_lateral_tracking_mean_abs_mm_before"],
        )


if __name__ == "__main__":
    unittest.main()
