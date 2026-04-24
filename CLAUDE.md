# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TeleImager is a camera image streaming library for Unitree robotics teleoperation. It captures video from Intel RealSense cameras and publishes frames over ZeroMQ (PUB-SUB).

This is a simplified fork — only RealSense + ZMQ publishing is supported. UVC/OpenCV/IsaacSim backends and WebRTC have been removed.

## Setup & Install

```bash
# Client only
pip install -e .
# Client + server (adds pyrealsense2)
pip install -e ".[server]"
# System deps (Linux, for server)
sudo apt install -y libusb-1.0-0-dev libturbojpeg-dev
```

Python >=3.8, <3.12. Uses setuptools build system. No test suite exists.

## Running

```bash
# Server — captures from RealSense and publishes over ZMQ
teleimager-server                          # uses cam_config_server.yaml
teleimager-server --cf                     # discover RealSense serial numbers
teleimager-server --affinity               # pin to CPU cores 0,1,2

# Client — subscribes to ZMQ and displays frames
teleimager-client --host 192.168.123.164 --request_port 60000
```

## Architecture

Two modules in `src/teleimager/`:

- **image_server.py** — `ImageServer` class. Opens a RealSense pipeline, captures RGB+depth frames in a dedicated thread, writes to a `TripleRingBuffer`, and publishes via `ZMQ_PublisherManager`. Responds to config queries via `ZMQ_Responser` (REQ-REP). Entry point: `main()`.

- **image_client.py** — `ImageClient` class. Connects to the server's ZMQ REQ-REP port to fetch camera config (intrinsics, image shape, depth scale), then subscribes to PUB topics for each camera. Frames are received in background threads and stored in `TripleRingBuffer`s. Callers retrieve frames via `get_head_frame()` which returns a `TeleImage` dataclass (bgr, depth, fps fields).

Shared utilities live in `image_client.py` and are imported by `image_server.py` (the server depends on the client module, not vice versa):
- `TripleRingBuffer` — lock-based 3-slot ring buffer ensuring readers always get the latest complete frame without tearing
- `ZMQ_PublisherManager` / `ZMQ_Responser` — thin wrappers around pyzmq PUB and REP sockets
- `SimpleFPSMonitor` — rolling-window FPS tracker
- `TeleImage` — dataclass holding bgr, depth, and fps for a single frame

## Configuration

- `cam_config_server.yaml` — server-side camera config (resolution, fps, ZMQ ports, serial number, depth enable)
- `cam_config_client.yaml` — generated/saved by client after fetching config from server (gitignored)

## Key Conventions

- All user-callable APIs are located under `# public api` comments in the source
- Logging uses `logging_mp` (multiprocessing-safe logging) throughout
- Frames are JPEG-compressed for ZMQ transport; depth is raw uint16 bytes
- The server force-kills its process group on exit (`os.killpg`) to handle USB disconnect edge cases
