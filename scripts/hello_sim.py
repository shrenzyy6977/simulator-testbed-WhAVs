"""Baseline environment check: spawn a wheeled vehicle in PyBullet, drive it,
log the trajectory to CSV, and save a plot + a camera snapshot.

Run from the repo root:
    python scripts/hello_sim.py          # headless (works anywhere)
    python scripts/hello_sim.py --gui    # opens the 3D viewer
"""
import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data

GUI = "--gui" in sys.argv
OUT = os.path.join("experiments", "results")
os.makedirs(OUT, exist_ok=True)

p.connect(p.GUI if GUI else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
car = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.1])

# Print the joint table so we know what the model exposes
joints = {}
for j in range(p.getNumJoints(car)):
    info = p.getJointInfo(car, j)
    name, jtype = info[1].decode(), info[2]
    joints[name] = (j, jtype)
    print(f"joint {j:2d}  {name:35s} type={jtype}")

wheels = [j for n, (j, t) in joints.items() if "wheel" in n and t != p.JOINT_FIXED]
steer = [j for n, (j, t) in joints.items() if "steer" in n and t != p.JOINT_FIXED]
print("wheel joints:", wheels, "| steering joints:", steer)

dt, duration = 1 / 240, 10.0
log = []
for step in range(int(duration / dt)):
    t = step * dt
    angle = 0.0 if t < 3 else 0.3          # drive straight, then turn
    for w in wheels:
        p.setJointMotorControl2(car, w, p.VELOCITY_CONTROL, targetVelocity=30, force=20)
    for s in steer:
        p.setJointMotorControl2(car, s, p.POSITION_CONTROL, targetPosition=angle)
    p.stepSimulation()
    pos, _ = p.getBasePositionAndOrientation(car)
    log.append((round(t, 4), *pos))
    if GUI:
        time.sleep(dt)

# Save trajectory
with open(os.path.join(OUT, "trajectory.csv"), "w", newline="") as f:
    csv.writer(f).writerows([("t", "x", "y", "z"), *log])

arr = np.array(log)
plt.figure(figsize=(5, 5))
plt.plot(arr[:, 1], arr[:, 2])
plt.scatter(arr[0, 1], arr[0, 2], c="g", label="start")
plt.scatter(arr[-1, 1], arr[-1, 2], c="r", label="end")
plt.axis("equal"); plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.title("Baseline vehicle trajectory"); plt.legend()
plt.savefig(os.path.join(OUT, "trajectory.png"), dpi=150, bbox_inches="tight")

# Camera snapshot of the final scene
w, h = 640, 480
view = p.computeViewMatrixFromYawPitchRoll(log[-1][1:], 5, 45, -30, 0, 2)
proj = p.computeProjectionMatrixFOV(60, w / h, 0.1, 100)
rgb = np.reshape(np.array(p.getCameraImage(w, h, view, proj)[2], dtype=np.uint8), (h, w, 4))
plt.imsave(os.path.join(OUT, "snapshot.png"), rgb[:, :, :3])

print(f"Final position: x={arr[-1,1]:.2f} y={arr[-1,2]:.2f}. Saved outputs to {OUT}/")
p.disconnect()
