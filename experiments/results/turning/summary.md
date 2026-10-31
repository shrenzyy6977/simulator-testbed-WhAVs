# Skid-steer turning calibration (Earth Rover Mini+)

Log: cagataydev/scout-earthrover-ecot (MIT), EarthRover Mini+ driven indoors. 41 spin-in-place bursts, 18 clean (turned the commanded way by >= 20 deg, no other turn command within 0.8 s after).

| | median | quartiles |
|---|---|---|
| real yaw rate per unit spin command [rad/s] | 1.17 | 0.80-2.54 |
| pure-rolling yaw rate per unit command (firmware) [rad/s] | 3.33 | |
| skid factor chi = rolling / real | 2.8 | 1.3-4.2 |
| model sideways / forward tyre friction that reproduces it (asphalt) | **0.41** | 0.63 to < 0.25 |

With isotropic friction (1.0) the model turns at 0.44 rad/s per unit command, 2.7x slower than the real robot.
