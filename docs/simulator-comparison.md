# Simulator comparison

## Qualitative comparison reported in the MVSim paper (Table 1, SoftwareX 2023)

Copied from the paper for reference. Verify against current versions before relying on it.

| | Gazebo | Webots | MVSim |
|---|---|---|---|
| Creation | 2011 | 1996 | 2014 |
| Physics engine | DART | ODE | Hybrid (Box2D plus custom physics solvers) |
| 3D engine | OGRE | Custom | MRPT |
| Interfaces | C++, Python, ROS | C++, Python, ROS, MATLAB, Java | C++, Python, ROS |
| World files | SDF | VRML | XML |
| Robotics sensors | Yes (plugins) | Yes | Yes |
| Ground vehicles | Yes | Yes | Yes |
| UAVs | Yes | Yes | Not yet |
| Custom tire-ground force models | No | No | Yes |
| Approximated fast 2D lidar | No | No | Yes |
| License | Apache-2.0 | Apache-2.0 | BSD-3 |

## Our own observations

Fill in only what we have actually tried.

| Simulator | Install effort | What we ran | Notes |
|---|---|---|---|
| MVSim | TODO | TODO | TODO |
| Gazebo | TODO | TODO | TODO |
| Webots | TODO | TODO | TODO |

## Decision
TODO: what we are building on and why.
