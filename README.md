# robo-weld

A small research-oriented project demonstrating a synthetic seam-tracking correction loop, a procedural simulator scene, and opt-in adapter seams.

![Status: experimental](https://img.shields.io/badge/status-experimental-yellow) ![License: Apache--2.0](https://img.shields.io/badge/license-Apache--2.0-blue)

Created at [Exowatt](https://exowatt.com). Published under couplet-physical-ai, this project is governed independently and remains vendor-neutral.

## Status and scope

**Experimental.** This fresh-history export is deliberately a small, synthetic demonstration, not the original internal welding system or a reproduction of any reported training campaign. It is intended to be useful for learning about a deterministic correction loop and experimenting with provider interfaces. The CPU demo is the default quickstart and the required CI path.

This repository does not contain private CAD, site or part designs, real weld data, proprietary weights, training outputs, robot SDK binaries, or simulator/vendor media. No result here validates a robot, a welding process, or a safety system.

## Install

Requires Python 3.11+; the CPU demo has no runtime dependencies and uses only the standard library. From a checkout, the direct quickstart needs no install step:

```sh
PYTHONPATH=src python3 -m robo_weld.cpu_demo --seed 7
```

For an installed command, use [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked
uv run --locked robo-weld-cpu --seed 7
```

## Demo A — deterministic CPU synthetic seam / correction / DAgger loop

The direct command above is the default laptop quickstart and the public CI smoke test (no GPU, simulator, robot, network, or external service).

The script procedurally generates mathematical seam curves, applies a simple lateral correction policy, collects labels from a deterministic threshold-based **synthetic critic**, and refits a one-parameter policy over DAgger-style rounds. Output includes the seed, intervention/label counts, a no-correction baseline, the learned policy result and a synthetic expert reference. Re-running with the same inputs is deterministic. The baseline is deliberately a zero-gain controller; the expert reference uses the same synthetic action limit and is not a physical benchmark.

Those default interventions are synthetic algorithmic labels, **not reviews by a person, Jev, or an LLM supervisor**. A separately configured plugin can be explicitly selected with `--supervisor provider`; results identify that mode separately. The reported lateral-tracking metric measures only how closely the toy controller follows its generated curve; it is **not weld quality**, bead geometry, penetration, strength, or a validated physical measurement. This is not an autonomous welding policy. The single-gain toy learner converges to its synthetic expert reference; this illustrates the loop mechanics, not a research benchmark.

Run the CPU-only test suite:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Demo B — optional Isaac Lab procedural scene

`demos/isaac_lab/procedural_cell.py` creates a small scene from generic primitives: a stylized arm/torch, a coupon, and a highlighted synthetic seam. The geometry is authored in the script; it does not retrieve or bundle a robot model or scene asset. This optional visual smoke demo has **no trained policy**, no physics/articulation model, no validated robot dynamics, and is not a replay of the CPU experiment. It has not been run in this environment; public CI checks Python syntax only. It is a scene-generation example, not a full reproducible GPU training campaign.

Install a compatible Isaac Sim / Isaac Lab environment separately and accept the applicable NVIDIA terms. Nothing from NVIDIA, Isaac, Omniverse, or a robot vendor is distributed here; the demo uses only primitives defined in this repository and a locally created ground plane/light. From an Isaac Lab installation, for example:

```sh
./isaaclab.sh -p /path/to/robo-weld/demos/isaac_lab/procedural_cell.py --headless
```

The exact supported simulator, driver, and CUDA combination is controlled by the user-managed Isaac Lab installation. The optional scene is not run in public CI because it requires that licensed simulator and GPU-capable environment.

## Demo C — opt-in supervisor and robot adapter seams

The default installation has no supervisor, LLM client, network connection, robot driver, or robot SDK. An optional **research-only** supervisor plugin may implement `review(lateral_error_mm) -> correction_mm | None`; select it explicitly in the CPU loop:

```sh
ROBO_WELD_SUPERVISOR_PROVIDER='my_provider:create_supervisor' uv run --locked robo-weld-cpu --supervisor provider --seed 7
```

The factory name is `module:factory`; the plugin and its endpoint/authentication are entirely user-managed. Jev or other LLM-backed supervisors, if supplied by a user, are optional providers configured this way—there is no default gateway, endpoint, credential, or Exowatt service binding. Provider interventions are reported separately from the built-in synthetic critic; neither is a human review unless the plugin actually connects a human review workflow.

Real-robot adapter loading is separate, research-only, and disabled unless `ROBO_WELD_ENABLE_REAL_ROBOT=1` **and** `ROBO_WELD_ROBOT_ADAPTER_PROVIDER='module:factory'` are both explicitly set. This repository supplies only an interface contract, no hardware adapter. Do not connect a robot based on this sample; any real integration requires independent engineering, risk assessment, safety controls, and hardware-specific validation.

## Configuration and limits

- CPU demo options: `--seed`, `--rounds`, `--episodes-per-round`, and `--steps`.
- Optional providers: `ROBO_WELD_SUPERVISOR_PROVIDER`, `ROBO_WELD_ENABLE_REAL_ROBOT`, and `ROBO_WELD_ROBOT_ADAPTER_PROVIDER` as described above.
- The public quickstart and CI require no Exowatt services, credentials, or proprietary assets.
- The learned and expert-reference values are generated from this toy task's same synthetic labels; they are not an independent research benchmark.
- Simulation and synthetic labels cannot establish weld quality, process qualification, production readiness, or functional safety.

## License

Original code is licensed under [Apache-2.0](LICENSE). See [NOTICE](NOTICE).

## Contributing

See [CONTRIBUTING](https://github.com/couplet-physical-ai/.github/blob/main/CONTRIBUTING.md). Contributions require DCO sign-off (`git commit -s`); no CLA is required.

## Security

Report vulnerabilities using [GitHub private vulnerability reporting](https://github.com/couplet-physical-ai/robo-weld/security/advisories/new). See the [security policy](https://github.com/couplet-physical-ai/.github/blob/main/SECURITY.md); do not open a public issue.
