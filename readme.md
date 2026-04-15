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

Install DeepStream 6.0.1 via the NVIDIA SDK Manager or APT repository.

Install Python Bindings (pyds):

Bash
wget [https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.1/pyds-1.1.1-py3-none-linux_aarch64.whl](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.1/pyds-1.1.1-py3-none-linux_aarch64.whl)
pip3 install ./pyds-1.1.1-py3-none-linux_aarch64.whl
Phase 2: Model Preparation
Because DeepStream requires TensorRT engines, you must provide the base ONNX models. The Jetson Nano will compile these into .engine files on the first run.

Place yolov8n-face.onnx in the /models/ directory.

Place arcface.onnx (MobileFaceNet recommended for Nano) in the /models/ directory.

Phase 3: Database Initialization
We use SQLite to avoid the RAM overhead of a full PostgreSQL server.

Run the database setup script to create the tables:

Bash
python3 init_db.py
Register a Face: You must manually insert at least one known face embedding into the known_faces table to test recognition. (You can write a helper script to extract your face's embedding and insert it).

Phase 4: Execution
Run the main dual-stream pipeline.
Note: The very first execution will take 5-15 minutes as the Jetson Nano builds the TensorRT .engine files. Subsequent runs will start instantly.

Bash
python3 nano_dual_stream.py --input1 /dev/video0 --input2 /dev/video1
Testing & Monitoring
Verify Database: Open a new terminal and query the SQLite database to see live attendance logs:

Bash
sqlite3 database/attendance.db "SELECT * FROM attendance_logs ORDER BY timestamp DESC LIMIT 5;"
Monitor Hardware: Run jtop (install via sudo -H pip3 install jetson-stats) in a separate terminal to monitor your RAM usage, GPU load, and temperature. If RAM maxes out, reduce the camera resolution in the DeepStream capsfilter.