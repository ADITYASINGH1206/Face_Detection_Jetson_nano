# Edge AI Face Recognition Attendance System (Jetson Nano)

This project is a highly optimized, dual-stream facial recognition pipeline designed specifically for the NVIDIA Jetson Nano. It uses YOLOv8 for face detection, ArcFace for feature extraction, and a local SQLite database for blazing-fast, offline attendance logging.

## Hardware & Software Requirements
- **Hardware:** NVIDIA Jetson Nano (4GB) with a 32GB microSD card (minimum) and a cooling fan.
- **Cameras:** 2x USB Webcams or 2x IP Cameras (RTSP).
- **OS:** JetPack 4.6.1 (Ubuntu 18.04). **Do not use JetPack 5+**.
- **SDK:** DeepStream 6.0.1.

## Phase 1: System Setup
1. Flash your SD Card with the official NVIDIA JetPack 4.6.1 image.
2. Update the system and install dependencies:
   ```bash
   sudo apt update
   sudo apt install -y libssl1.0.0 libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstrtspserver-1.0-0 libjansson4 libyaml-cpp-dev python3-pip python3-gi python3-dev python3-gst-1.0 sqlite3