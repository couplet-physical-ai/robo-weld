"""Launch a static, procedural cell scene in a user-installed Isaac Lab.

The scene is authored here from primitive shapes; it fetches no robot, weld,
SimReady, or Omniverse content. It is a visual integration smoke demo, not a
physics-validated robot model or a training run.
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120, help="simulation frames to step")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Isaac Lab is an optional, user-installed prerequisite. All geometry
    # below is created procedurally by this script from generic primitives.
    import isaaclab.sim as sim_utils

    simulation = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args.device)
    )
    simulation.set_camera_view(eye=(1.7, -1.8, 1.3), target=(0.55, 0.0, 0.3))

    def cuboid(path: str, size: tuple[float, float, float], position: tuple[float, float, float], color: tuple[float, float, float], angle_y: float = 0.0) -> None:
        cfg = sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        quaternion = (math.cos(angle_y / 2.0), 0.0, math.sin(angle_y / 2.0), 0.0)
        cfg.func(path, cfg, translation=position, orientation=quaternion)

    def cylinder(path: str, radius: float, height: float, position: tuple[float, float, float], color: tuple[float, float, float]) -> None:
        cfg = sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        cfg.func(path, cfg, translation=position)

    # Build the ground and lighting locally; no default vendor environment or
    # asset-server content is requested by this scene.
    cuboid("/World/Ground", (3.0, 3.0, 0.05), (0.0, 0.0, -0.025), (0.18, 0.20, 0.23))
    light = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)

    # Stylized arm links, torch body, coupon, and a bright synthetic seam.
    # These are static visual primitives, not a manufacturer robot model.
    cylinder("/World/ArmBase", 0.075, 0.34, (0.15, 0.0, 0.18), (0.28, 0.34, 0.42))
    cuboid("/World/UpperLink", (0.34, 0.075, 0.075), (0.40, 0.0, 0.43), (0.25, 0.40, 0.55), angle_y=-0.30)
    cuboid("/World/ForeLink", (0.28, 0.06, 0.06), (0.68, 0.0, 0.39), (0.25, 0.40, 0.55), angle_y=0.12)
    cylinder("/World/Torch", 0.018, 0.16, (0.88, 0.0, 0.30), (0.85, 0.36, 0.10))
    cuboid("/World/Coupon", (0.52, 0.30, 0.035), (0.68, 0.0, 0.055), (0.38, 0.41, 0.44))
    cuboid("/World/SyntheticSeam", (0.48, 0.006, 0.004), (0.68, 0.0, 0.075), (0.95, 0.62, 0.12))

    simulation.reset()
    for _ in range(args.steps):
        simulation.step(render=not args.headless)
    print(
        "Procedural Isaac Lab scene launched. Static primitives only; "
        "no robot policy, weld process, or physical validation was run."
    )
    simulation_app.close()


if __name__ == "__main__":
    main()
