# Literature notes

Drafted from the paper. Read the paper yourself and edit these notes so they reflect your understanding.

## 1. MVSim (base paper)
- **Citation:** J.-L. Blanco-Claraco, B. Tymchenko, F. J. Mañas-Alvarez, F. Cañadas-Aránega, Á. López-Gázquez, J. C. Moreno, "MultiVehicle Simulator (MVSim): Lightweight dynamics simulator for multiagents and mobile robotics research", SoftwareX 23 (2023) 101443.
- **Link:** https://doi.org/10.1016/j.softx.2023.101443 · code: https://github.com/MRPT/mvsim · docs: https://mvsimulator.readthedocs.io
- **Problem addressed:** closed-loop robotics/vehicle software needs realistic testing; field tests are costly. Needs a light simulator for many ground vehicles.
- **Approach:** 2D Box2D physics for bodies and collisions, with wheel-ground forces computed per wheel from wheel load, motor torque and local wheel velocity via a friction model (Section 2.2, Algorithm 1). Vehicles keep 3D poses so they can move on elevation maps.
- **Interfaces:** XML world files (no URDF), ZeroMQ + Protobuf pub/sub with C++ and Python clients, ROS 1/2 node, `mvsim-cli`, headless mode.
- **Sensors (paper):** pin-hole camera, RGB-D, 2D/3D LiDAR, GPU-accelerated via OpenGL.
- **Evidence:** CPU usage benchmark against Gazebo and Webots with up to 23 differential-drive robots with 2D LiDAR (Fig. 6).
- **Limitations / gaps:** no UAVs at publication time; all implemented friction models rely on one mechanical wheel model; 2D physics for body collisions.
- **Relevance to our project:** base paper and base code.

## 2. Other simulators cited by the paper (to read)
- Gerkey, Vaughan, Howard, "The Player/Stage project: Tools for multi-robot and distributed sensor systems", ICAR 2003.
- Koenig, Howard, "Design and use paradigms for Gazebo, an open-source multi-robot simulator", IROS 2004.
- Michel, "Webots: Professional mobile robot simulation", Int. J. Adv. Robotic Systems 1(1), 2004.

## 3. Supporting references from the paper (to read as needed)
- Catto, "Iterative dynamics with temporal coherence", GDC 2005 (Box2D).
- Ward, Iagnemma, "A dynamic-model-based wheel slip detector for mobile robots on outdoor terrain", IEEE T-RO 2008 (rolling friction model used by MVSim).
- Authors' benchmark scripts: https://github.com/ual-arm/robotic-simulators-benchmark

## Entry template
- **Citation:**
- **Link:**
- **Problem addressed:**
- **Approach:**
- **Relevance to our project:**
- **Limitations / gap:**
