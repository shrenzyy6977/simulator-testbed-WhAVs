# Literature update: simulators and benchmarks for field robots (2025–26)

Answers the first gap in the progress review: our sources were from 2021–2024 and none covered
current simulators or benchmarks for field robots. Each entry was checked against the arXiv
version linked (abstract and full text where noted); check the published version before citing a
number in the report. Entries use the template from `literature-notes.md`.

Our own numbers quoted below come from `experiments/results/rq1/` and `rq3/` (Apple M4, CPU only).

## 1. Works named in the progress review

### 1.1 Verti-Bench (RSS 2025 per the review; arXiv lists no venue)
- **Citation:** T. Xu, C. Pan, M. B. Rao, A. Datar, A. Pokhrel, Y. Lu, X. Xiao, "Verti-Bench: A General
  and Scalable Off-Road Mobility Benchmark for Vertically Challenging Terrain", arXiv:2502.11426, 2025.
- **Link:** https://arxiv.org/abs/2502.11426
- **Problem addressed:** off-road mobility systems are hard to compare fairly because vehicles and
  terrains differ; dataset-based comparison does not capture closed-loop interaction.
- **Approach:** Project Chrono (PyChrono). 100 off-road worlds, 1,000 navigation tasks. 10 ground
  classes: 7 rigid (grass, wood, gravel, dirt, clay, rock, concrete), each with a friction coefficient
  sampled from a normal distribution, and 3 deformable (snow, mud, sand) simulated with the Soil
  Contact Model (SCM, Bekker–Wong) at soft / medium / hard settings. 9 vehicles (1/10, 1/6 and full
  scale; 4-, 6-, 8-wheeled and 2-tracked), written as Chrono C++ classes, not URDF. Gym-like interface
  and datasets of expert demonstrations, random exploration and failures.
- **Evidence:** 10 baselines in three groups: classical (PID, elevation heuristics, MPPI), end-to-end
  learning (RL, MCL, ACL) and hybrid (WMVCT, MPPI-6, TAL, TNT). Hybrid planners do best (TAL 58.4%,
  TNT 57.4% success); PID reaches 48.8% and RL only 10.0% (stuck in 75.7% of its failures).
  Performance drops with elevation, deformable ground and obstacle density. Real 1/10-scale tests on
  high elevation: TNT 5/5, PID 3/5, ACL 2/5.
- **Limitations / gap:** real-time factor "between 0.4 and 1.5" (Sec. VI), about 10 h to evaluate one
  system; no GPU acceleration; ground-truth perception (true state and elevation map); learned
  systems were not trained in the benchmark itself.
- **Observation:** Sec. III-A calls Chrono "slightly faster-than-real-time", which conflicts with the
  0.4–1.5 range in Sec. VI.
- **Relevance to our project:** the closest benchmark to ours and the source of our ground-class idea.
  We match its rigid-ground part (per-class friction, plus rolling resistance it does not model) and
  add perception noise and URDF loading, both listed as its gaps or future work. We do not have
  deformable soil (SCM); our sinkage is a soft-contact proxy. Its finding that classical PID beats
  RL is a useful contrast with our RQ3, where PPO beats go-to-goal only after randomised training.

### 1.2 URBAN-SIM (CVPR 2025, Highlight)
- **Citation:** W. Wu, H. He, C. Zhang, J. He, S. Z. Zhao, R. Gong, Q. Li, B. Zhou, "Towards Autonomous
  Micromobility through Scalable Urban Simulation", CVPR 2025, arXiv:2505.00690.
- **Link:** https://arxiv.org/abs/2505.00690 · project: https://metadriverse.github.io/urban-sim/
- **Problem addressed:** delivery robots, wheelchairs and scooters are still driven manually in busy
  cities; learning to drive them needs large, varied urban training scenes.
- **Approach:** built on NVIDIA Omniverse and PhysX 5. Hierarchical urban scene generation,
  interactive dynamics generation (pedestrians, vehicles) and asynchronous scene sampling. Robots load
  from URDF. Terrain types: flat, stairs, slopes, rough ground; the appendix lists 8 textures that
  "can be set with varied friction coefficients". URBAN-BENCH: 8 tasks over urban locomotion,
  navigation and traverse; 4 robots (COCO delivery robot, Unitree Go2, B2-W, G1); PPO via RL Games.
- **Evidence:** up to 256 parallel environments on one NVIDIA L40S (46 GB), 11.2 GB used, 1,800–2,600
  FPS with RGB-D sensors (the appendix says 2,300–2,600); FPS grows from 100 with 1 environment to
  2,620 with 256.
- **Limitations / gap:** simulation only, no real-robot experiments yet (Appendix G); scenes are
  generated, so they do not match real data distributions; urban scenes only; no rolling resistance
  or deformable ground is described.
- **Observation:** the review says URBAN-SIM has no ground properties per surface. The paper does
  allow a friction coefficient per texture; what it lacks is rolling resistance, sinkage and
  off-road ground classes. We should phrase our contribution that way.
- **Relevance to our project:** the reference design for Option B (URDF robots, parallel PPO
  training). It needs a datacentre GPU; ours runs on a laptop CPU.

### 1.3 Wheeled Lab (CoRL 2025)
- **Citation:** T. Han, P. Shah, S. Rajagopal, Y. Bao, S. Jung, S. Talia, G. Guo, B. Xu, B. Mehta,
  E. Romig, R. Scalise, B. Boots, "Wheeled Lab: Modern Sim2Real for Low-cost, Open-source Wheeled
  Robotics", CoRL 2025, arXiv:2502.07380.
- **Link:** https://arxiv.org/abs/2502.07380 · site: https://uwrobotlearning.github.io/WheeledLab/
- **Problem addressed:** modern sim-to-real RL relies on proprietary simulators and costly robots,
  which leaves out smaller labs and teaching.
- **Approach:** Isaac Lab with low-cost RC cars (HOUND, about USD 3,000 with sensors and a Jetson;
  MuSHR, about USD 930). Three zero-shot policies: drifting on carpet, elevation traversal over indoor
  ramps and walls (2.5 × 2.5 m local elevation map), and visual navigation on a figure-8 track.
  Randomised friction (centred on a measured 0.4; a wide 0.2–0.8 range worked "poorly and
  inconsistently") and actuator gains; Gaussian noise on elevation observations; colour jitter and blur
  for vision. 1,024 parallel environments; more than 2,000 agents on one RTX 3080.
- **Evidence:** the drifting policy completes full laps with slip angles up to 58° at about 1.6 m/s,
  while the baseline cannot hold a stable turn; visual navigation succeeds in 3/5 trials with
  augmentation and 0/5 without (MLP).
- **Limitations / gap:** simple indoor terrain; one friction value per task rather than ground classes;
  at about 3 m/s wheels reach 60 rad/s and "occasionally" penetrate the ground and jump (fixed with
  damped suspension joints); success rates are small-sample.
- **Relevance to our project:** shows that randomising friction and actuators transfers to real
  cars, and that a too-wide friction range hurts, which matches our RQ3 caution about randomisation
  ranges. Its wheel-contact glitch is a PhysX counterpart to our MuJoCo skid-steer turning problem.

### 1.4 Vid2Sim (arXiv 2025)
- **Citation:** Z. Xie, Z. Liu, Z. Peng, W. Wu, B. Zhou, "Vid2Sim: Realistic and Interactive Simulation
  from Video for Urban Navigation", arXiv:2501.06693, 2025.
- **Link:** https://arxiv.org/abs/2501.06693
- **Approach:** rebuilds photorealistic, interactive 3D scenes from a single monocular video and trains
  visual navigation agents with RL inside them.
- **Evidence:** urban navigation success is 31.2% higher in digital twins and 68.3% higher in the real
  world than with agents trained in earlier simulators (abstract).
- **Limitations / gap:** the contribution is the visuals; physics and ground properties are not the
  focus. Photorealism is out of scope for us.
- **Relevance to our project:** low. Cited to explain why we did not build on it.

### 1.5 NavBench-GS / S2E (arXiv 2025)
- **Citation:** H. He, Y. Ma, B. Squicciarini, W. Wu, B. Zhou, "From Seeing to Experiencing: Scaling
  Navigation Foundation Models with Reinforcement Learning", arXiv:2507.22028, 2025.
- **Link:** https://arxiv.org/abs/2507.22028
- **Approach:** pretrains a navigation model on videos, then fine-tunes it with RL in simulation using a
  residual-attention module. NavBench-GS is its benchmark of 3D Gaussian-Splatting reconstructions of
  real scenes with physical interaction.
- **Limitations / gap:** photorealistic urban scenes; the abstract gives no physics or ground model.
- **Relevance to our project:** low, same reason as Vid2Sim. Its "offline data, then RL" recipe is
  the template for the FrodoBots baseline the review asks for.

## 2. Simulation frameworks

### 2.1 MuJoCo Playground (RSS 2025, Outstanding Demo Paper Award)
- **Citation:** K. Zakka, B. Tabanpour, Q. Liao, et al., "MuJoCo Playground", arXiv:2502.08844, 2025.
- **Link:** https://arxiv.org/abs/2502.08844 · https://playground.mujoco.org/
- **Approach:** environments, batch renderer and training on MJX (MuJoCo in JAX) on GPU; quadrupeds,
  humanoids, hands and arms; policies train "in minutes on a single GPU" and transfer zero-shot from
  state and pixels.
- **Relevance to our project:** the GPU route for our MuJoCo model. It has no wheeled off-road or
  ground-class environments, which is where we add something.

### 2.2 mjlab and MuJoCo Warp (2026)
- **Citation:** K. Zakka, Q. Liao, B. Yi, L. Le Lay, K. Sreenath, P. Abbeel, "mjlab: A Lightweight
  Framework for GPU-Accelerated Robot Learning", arXiv:2601.22074, 2026.
- **Link:** https://arxiv.org/abs/2601.22074 · https://github.com/mujocolab/mjlab
- **Approach:** Isaac Lab's manager-based API (observation, reward and event managers) on top of
  MuJoCo Warp, the GPU version of MuJoCo by Google DeepMind and NVIDIA.
- **Relevance to our project:** if the TA confirms GPU access, porting `terrain3d` to MuJoCo Warp or
  mjlab keeps our models (heightfields, per-geom friction, URDF loading) and adds thousands of
  parallel environments. Heightfield and condim-6 contact support on the GPU must be checked first.

### 2.3 Isaac Lab (2025)
- **Citation:** M. Mittal et al. (NVIDIA), "Isaac Lab: A GPU-Accelerated Simulation Framework for
  Multi-Modal Robot Learning", arXiv:2511.04831, 2025.
- **Link:** https://arxiv.org/abs/2511.04831
- **Approach:** the successor to Isaac Gym: GPU physics, photorealistic rendering, actuator models,
  sensors at several rates, domain randomisation. Base of Wheeled Lab.
- **Relevance to our project:** the main alternative engine. It does not run on macOS, which is one
  reason we chose MuJoCo.

## 3. Terrain modelling

### 3.1 Chrono::CRM (submitted to Journal of Terramechanics, 2025)
- **Citation:** H. Unjhawala, L. Bakke, H. Zhang, M. Taylor, G. Arivoli, R. Serban, D. Negrut, "A
  Physics-Based Continuum Model for Versatile, Scalable, and Fast Terramechanics Simulation",
  arXiv:2507.05643, 2025.
- **Link:** https://arxiv.org/abs/2507.05643
- **Approach:** a continuum (SPH) soil model in Project Chrono instead of the semi-empirical
  Bekker–Wong / Janosi–Hanamoto models; runs on GPU, validated against three physical tests (one with
  NASA's MGRU3 rover) and a DEM digging simulation; "active domains" handle terrain up to 10 km long
  with 100 million particles.
- **Relevance to our project:** shows the high-fidelity end of terrain modelling. Granular soil is out
  of scope for us (per the review); it is a reference for calibrating our sand and mud classes.

### 3.2 ADEPT (arXiv 2025)
- **Citation:** Y. Yu, J. Xu, L. Liu, "ADEPT: Adaptive Diffusion Environment for Policy Transfer
  Sim-to-Real", arXiv:2506.01759, 2025.
- **Link:** https://arxiv.org/abs/2506.01759
- **Approach:** a diffusion model generates new off-road terrains that adapt to the current policy's
  performance; policies trained this way beat ones trained on procedural or natural terrains.
- **Relevance to our project:** our maps are random cosine hills. Adaptive terrain generation would
  be a natural next step for training, and real elevation maps (`terrain3d.terrain.from_file`) for
  evaluation.

## 4. Data for the ERC setting

### 4.1 FrodoBots-2K dataset (2024)
- **Link:** https://huggingface.co/datasets/frodobots/FrodoBots-2K (CC-BY-SA-4.0, about 1 TB)
- **Contents:** about 2,000 h of teleoperated sidewalk driving by game players in 10+ cities (about
  1,300 h released when the page was last updated): front camera at about 20 FPS (1024 × 576), rear
  camera at about 20 FPS, GPS at 1 Hz, accelerometer at 100 Hz, gyroscope and magnetometer at 1 Hz,
  control inputs at 10 Hz and RPM of all four wheels. No dedicated dataset paper was found; Zhang,
  Lin and Visser (arXiv:2407.05735, 2024) use it for visual SLAM and add recordings from the ERC at
  ICRA@40.
- **Relevance to our project:** the logs for the review's third baseline, and the data to calibrate
  skid-steer turning (wheel RPM against GPS / IMU motion). Our env first gave a GPS fix every
  0.1 s; it now uses the real 1 Hz rate (`NavConfig.gps_hz`), with the fix age in the observation.
  The control rate (10 Hz) already matched.

## 5. Comparison with our simulator

Numbers are from each paper on its own hardware and task, so they are not a like-for-like speed
comparison (URBAN-SIM renders RGB-D, we render nothing during training).

| | Verti-Bench | URBAN-SIM | Wheeled Lab | MVSim (base) | Ours (`terrain3d`) |
|---|---|---|---|---|---|
| Engine | Chrono (CPU) | PhysX 5 / Omniverse (GPU) | Isaac Lab (GPU) | Box2D + custom wheel model | MuJoCo 3 (CPU) |
| Terrain | 100 off-road worlds | generated city scenes | indoor ramps, carpet | 2.5D elevation maps | heightfield maps: hills, slopes, image / `.npy` |
| Ground properties | 7 rigid classes (friction), 3 deformable (SCM) | friction per texture (8) | one randomised friction | per-wheel friction model | 6 classes: friction, torsional, rolling resistance, softness |
| Deformable soil | yes (SCM) | no | no | no | no (soft-contact proxy) |
| Robots from URDF | no (C++ classes) | yes | Isaac Lab assets | no (XML) | yes (two rovers) |
| Perception noise | no (ground truth) | RGB-D rendered; noise not described | obs. noise, image augmentation | camera, RGB-D, LiDAR simulated | GPS noise and bias at 1 Hz, control delay |
| Speed | 0.4–1.5x real time | up to 2,620 FPS, 256 envs, L40S | not reported | slower than real time on our hardware (week 1) | 162x real time per process; 5,154 steps/s with 8 processes on an M4 |
| Baselines | 10 (classical, RL, hybrid) | PPO per robot | RL policy vs. baseline, 3 tasks | none | 4 path trackers, go-to-goal, PPO (2 training regimes) |

## 6. Where this leaves our contribution

- **The gap holds, with nuance.** No recent simulator combines ground classes beyond friction
  (rolling resistance, softness), URDF robots, perception noise and laptop-speed training.
  Verti-Bench has the richest ground model but is slow, uses true perception and has no URDF.
  URBAN-SIM has URDF and speed but only friction per texture and only urban scenes.
- **Our RQ1 result places us between them:** roughly 100–400x faster per process than Verti-Bench's 0.4–1.5x real time, with
  every feature on, but with a simpler (non-deformable) ground model and no camera.
- **What the literature says we still lack:** deformable soil (out of scope), camera observations
  (URBAN-SIM, Wheeled Lab), GPU parallelism (MuJoCo Warp / mjlab is the route), calibration against
  real logs (FrodoBots-2K).
- **Findings to connect in the report:** Verti-Bench's RL baseline fails badly (10% success), and
  Wheeled Lab finds that too-wide friction randomisation hurts. Our RQ3 sits in between: with the
  Earth Rover's 1 Hz GPS, both PPO policies beat a classical go-to-goal controller under heavy noise
  (49–55% vs 36%), but randomised training gives no significant extra robustness (+5.2 points on
  noise, p = 0.11; −4.6 on unseen ground). With a fresh fix every step it had been significant
  (+8.8 points, p = 0.019), so the GPS model alone changes the conclusion. RQ2 shows the same at the
  level of rankings: single upgrades barely reorder seven policies (Kendall's tau 0.71–1.00), but all
  upgrades together do (tau 0.52), and the best classical controller changes from Stanley to go-to-goal.
