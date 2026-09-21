  # Iron Dome

A real-time computer vision turret that detects, tracks, and aims a laser at balloons.

A Jetson Orin Nano runs a fine-tuned YOLO detector through TensorRT. A Kalman filter predicts target motion to compensate for latency, and an Arduino Nano drives the pan/tilt motors and laser over serial protocol.

<img width="756" height="1008" alt="IMG_4330" src="https://github.com/user-attachments/assets/20c8d75b-bfd3-4136-bbdc-0f741d2756db" />

*First housing iteration*

**Status:** Work in progress. The system is functional but not yet optimized. A refined housing and demo video are coming soon.
