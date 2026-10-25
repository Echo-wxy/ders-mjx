# DERS: Dual-Execution Robotics Simulation

Implementation of **dual-execution simulation** for contact-rich robot manipulation, accompanying the article *Dual-execution simulation preserves exact forward dynamics while providing differentiable gradients for contact-rich robot manipulation* (Gao and Li, Scientific Reports, under revision).

Every transition is executed twice. A hard engine produces the state that the controller experiences; a relaxed engine supplies the derivative; a straight-through coupling returns the hard state bit for bit while passing the relaxed derivative. Each discrete mechanism of manipulation receives its own relaxation and temperature, and a residual network shapes the backward model without touching the executed trajectory.

## What is in the repository

| Path | Content |
|---|---|
| `ders/coupling.py` | Straight-through coupling. The executed value is the hard value itself; the derivative is attached through a custom tangent rule, so the forward state is bit-for-bit that of the hard engine (Eqs. 4--6). |
| `ders/relaxations.py` | The mechanism-specific relaxations of the backward model: triangular-kernel spatial read (Eq. 7), softmax skill router (Eq. 8), signed-distance contact gate (Eq. 9), gated counterfactual contact impulse (Eq. 10), soft depth composite for rendering (Eq. 11), each with its hard counterpart. |
| `ders/residual.py` | Residual network and its training target (Eqs. 3 and 12): a correction to the rate of change of the state, applied in the backward model. |
| `ders/soft_contact.py` | The contact gate and the counterfactual impulse applied inside the MJX constraint solver; the bitwise-exact coupling of hard and relaxed substeps; the conventional compliant-contact relaxation and the Contacts-From-Distance relaxation used as baselines. |
| `ders/engine.py` | Physics backend on MuJoCo/MJX: scenes, the hard step, and the transitions `hard_ad`, `cfd`, `ders`, `generic`. |
| `ders/metaworld_patch.py` | Meta-World v3 Sawyer scenes made MJX-compatible (collision geometry only; visual geometry unchanged), including the round-geometry variants. |
| `ders/sampling.py`, `ders/problems.py` | Scripted controllers, evaluation states and trajectory problems. |
| `scripts/` | Experiments (below), the runner `run_all.py`, the self-check and the analysis. |
| `third_party/` | MJX from the `diffmjx` branch and softjax, Apache-2.0 (`third_party/NOTICE.md`). |
| `tests/` | Unit tests: bitwise forward identity of every coupled transition; sharp-limit and differentiability checks of every relaxation. |

## Installation (Windows or Linux, CPU)

```
conda create -y -n ders python=3.12
conda activate ders
pip install -r requirements-windows.txt
pip install --no-deps metaworld==3.1.1 third_party/mujoco_mjx_diffmjx third_party/softjax
python scripts/selfcheck.py
```

`install_windows.bat` and `run_all_windows.bat` wrap these commands for a default Anaconda installation. The self-check loads the five scenes, verifies the bitwise forward identity and times one gradient.

## Experiments

| Script / stage | Experiment | Article | Output |
|---|---|---|---|
| `scripts/forward.py` (stage `main`) | Forward consistency of the coupled transition with the hard engine: exact and left-to-right coupling, relaxed engine executed | Section 2.1, Fig. 3a | `results/g1/` |
| `scripts/grad_fidelity.py` (stage `main`) | Single-step Jacobians of every estimator against central finite differences, seven step sizes, contact strata (far, near, pre-contact, mode switch); `--alpha-scales` gives the temperature sweep | Section 2.2, Fig. 4 | `results/g3/`, `results/g4/` |
| `scripts/trajopt.py --mode tune` (stage `tune`) | Pre-registered parameter selection on the tuning seed 0 | Methods | `results/tune/`, `configs/tuned.json` |
| `scripts/trajopt.py --mode descent` (stage `main`) | One-step descent test and multi-step gradient cosines (horizons 5 and 30) | Sections 2.2--2.3 | `results/g6a/`, `results/g5/` |
| `scripts/trajopt.py --mode optimize` (stage `main`) | 50-iteration trajectory optimization with the same hard forward step for every estimator | Section 2.3 | `results/g6b/` |
| `scripts/cost.py` (stage `cost`) | Compile time, forward and forward+backward time, memory, horizon and batch scaling | Section 2.6, Fig. 6 | `results/g7/` |
| stage `round` | The same analyses on the scenes whose collision geometry reports separation distances (finger pads as capsules, objects as spheres or capsules) | Section 2.3 | `results/round/` |
| `scripts/analyze.py` | Summary tables, strata metrics and paired cluster-bootstrap intervals from `results/` | Tables 3--5 | `analysis/` |

`python scripts/run_all.py --jobs 4` runs stages `tune`, `warmup`, `main` and `cost`; `python scripts/run_all.py --stage round --jobs 3` runs the round-geometry scenes. Every stage is resumable: a job whose output exists is skipped.

Five gradient estimators share the identical hard forward step and differ only in the backward model: direct automatic differentiation of the hard step (`hard_ad`), the Contacts-From-Distance relaxation of Paulus et al. (`cfd`, run with its authors' code), a conventional compliant-contact straight-through relaxation (`generic`), the proposed estimator (`ders`), and zeroth-order randomized smoothing (`rs`). The relaxation parameters of every estimator are selected on seed 0 from a pre-registered grid (`GRID` in `scripts/trajopt.py`) and frozen before the evaluation seeds 1--5 are run.

## Protocol notes

* All estimators execute the identical hard forward step: MuJoCo 3.3.1 through MJX, semi-implicit Euler with a 2.5 ms step and five sub-steps per control step, soft-constraint contacts with pyramidal friction cones, Newton solver with a fixed six iterations and six line-search iterations. The adaptive integrator of DiffMJX is not used.
* Contact candidates are generated within 1 cm of touching and activated at zero signed distance; with this setting the executed trajectory is identical to that of a zero-margin model.
* Gradient comparisons run in float64; optimization runs in float32.
* The cross-simulator benchmark of the article (PyBullet, MuJoCo, Isaac Gym, Brax and Nimble on Meta-World MT10/MT50) depends on those third-party engines and their task ports and is not part of this repository.

## Raw results

The raw results used in the article are deposited together with this code (directory `results/` of the archived release). Floating-point results are reproducible bit for bit on the same CPU and may differ in the last digits across machines.

## License

Apache-2.0 (see `LICENSE`). Third-party components keep their own licenses (`third_party/NOTICE.md`).
