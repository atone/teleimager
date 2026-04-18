# Copyright 2025 YuShu TECHNOLOGY CO.,LTD ("Unitree Robotics")
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TeleImager image server — RealSense head camera, ZMQ PUB only.

This is a simplified variant of the upstream teleimager: it supports a single
RealSense RGB(+D) head camera and publishes multipart PUB frames over ZMQ.
UVC / OpenCV / IsaacSim backends and the WebRTC publisher have been removed.
"""
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import argparse
import functools
import itertools
import json
import os
import platform
import signal
import threading
import time
import zlib
from typing import Any, Dict, Optional

import cv2
import numpy as np
import yaml

from .image_client import TripleRingBuffer, ZMQ_PublisherManager, ZMQ_Responser

# ========================================================
# cam_config_server.yaml path
# ========================================================
CONFIG_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "cam_config_server.yaml",
    )
)


# ========================================================
# RealSense helpers
# ========================================================
def _import_pyrealsense2():
    try:
        import pyrealsense2 as rs
        return rs
    except ImportError:
        arch = platform.machine()
        system = platform.system()
        if system == "Linux" and arch.startswith("aarch64"):
            # Jetson NX / arm64
            msg = (
                "[RealSense] pyrealsense2 not installed. Please build from source:\n"
                "    cd ~\n"
                "    git clone https://github.com/IntelRealSense/librealsense.git\n"
                "    cd librealsense\n"
                "    git checkout v2.50.0\n"
                "    mkdir build && cd build\n"
                "    cmake .. -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=$(which python3)\n"
                "    make -j$(nproc)\n"
                "    sudo make install\n"
            )
        else:
            # x86/x64
            msg = (
                "[RealSense] pyrealsense2 not installed. You can try:\n"
                "    pip install pyrealsense2\n"
            )
        raise RuntimeError(msg)


def list_realsense_serial_numbers():
    """Return serial numbers of all connected RealSense devices."""
    rs = _import_pyrealsense2()
    serials = []
    for dev in rs.context().query_devices():
        try:
            serials.append(dev.get_info(rs.camera_info.serial_number))
        except Exception:
            continue
    return serials


# ========================================================
# Camera
# ========================================================
class RealSenseCamera:
    def __init__(self, cam_topic, serial_number, img_shape, fps,
                 zmq_port=55555, enable_depth=False):
        rs = _import_pyrealsense2()
        self._cam_topic = cam_topic
        self._img_shape = img_shape  # (H, W)
        self._fps = fps
        self._zmq_port = zmq_port
        self._zmq_buffer = TripleRingBuffer()
        self._ready = threading.Event()

        self._serial_number = serial_number
        self._enable_depth = enable_depth

        self.pipeline = None
        try:
            self.align = rs.align(rs.stream.color)
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self._serial_number)
            config.enable_stream(
                rs.stream.color, self._img_shape[1], self._img_shape[0],
                rs.format.bgr8, self._fps,
            )
            if self._enable_depth:
                config.enable_stream(
                    rs.stream.depth, self._img_shape[1], self._img_shape[0],
                    rs.format.z16, self._fps,
                )

            profile = self.pipeline.start(config)
            self._device = profile.get_device()
            if self._device is None:
                raise RuntimeError("pipeline profile has no device")

            if self._enable_depth:
                depth_sensor = self._device.first_depth_sensor()
                self.g_depth_scale = depth_sensor.get_depth_scale()
            else:
                self.g_depth_scale = 0.0

            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.intrinsics = intr
            self.intrinsics_dict = {
                "fx": intr.fx, "fy": intr.fy,
                "ppx": intr.ppx, "ppy": intr.ppy,
                "width": intr.width, "height": intr.height,
                "coeffs": list(intr.coeffs),
                "model": str(intr.model),
            }
            logger_mp.info(str(self))
        except Exception as e:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
            raise RuntimeError(
                f"[RealSenseCamera] Failed to initialize RealSense camera "
                f"{self._serial_number}: {e}"
            )

    def __str__(self):
        return (
            f"[RealSenseCamera: {self._cam_topic}] initialized with "
            f"{self._img_shape[0]}x{self._img_shape[1]} @ {self._fps} FPS, "
            f"depth={'on' if self._enable_depth else 'off'}, "
            f"zmq_port={self._zmq_port}"
        )

    def __repr__(self):
        return self.__str__()

    # ----- frame production -----
    def _update_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            return
        bgr_numpy = np.asanyarray(color_frame.get_data())
        ok, buf = cv2.imencode(".jpg", bgr_numpy)
        if not ok:
            return
        bgr_bytes = buf.tobytes()

        depth_bytes = None
        if self._enable_depth:
            depth_frame = aligned_frames.get_depth_frame()
            if depth_frame:
                depth_arr = np.asanyarray(depth_frame.get_data())
                # zlib level 1: ~3-5x compression on depth, ~1-2ms on Orin NX
                depth_bytes = zlib.compress(depth_arr.tobytes(), 1)

        self._zmq_buffer.write((bgr_bytes, depth_bytes))

        if not self._ready.is_set():
            self._ready.set()

    # ----- accessors -----
    def wait_until_ready(self, timeout=None):
        return self._ready.wait(timeout=timeout)

    def get_frame_parts(self):
        """Return (jpeg_bytes, depth_bytes_or_None, meta_dict) for multipart publish."""
        jpeg_bytes, depth_bytes = self._zmq_buffer.read()
        if jpeg_bytes is None:
            return None
        meta: Dict[str, Any] = {}
        if self._enable_depth and depth_bytes is not None:
            meta = {
                "depth_shape": list(self._img_shape),  # [H, W]
                "depth_scale": float(self.g_depth_scale),
                "depth_compression": "zlib",
            }
        return jpeg_bytes, depth_bytes, meta

    def get_zmq_port(self):
        return self._zmq_port

    def get_fps(self):
        return self._fps

    def release(self):
        try:
            if self.pipeline is not None and getattr(self.pipeline, "_running", False):
                try:
                    self.pipeline.stop()
                except Exception as e:
                    logger_mp.warning(f"[RealSenseCamera] pipeline.stop() failed: {e}")
        except Exception:
            pass
        self.pipeline = None
        logger_mp.info(f"[RealSenseCamera] Released {self._cam_topic}")


# ========================================================
# Image server
# ========================================================
class ImageServer:
    """Hosts a single RealSense head camera and publishes frames via ZMQ PUB."""

    CAMERA_TOPIC = "head_camera"

    def __init__(self, cam_config):
        self._cam_config = cam_config
        self._stop_event = threading.Event()
        self._camera: Optional[RealSenseCamera] = None
        self._threads: list[threading.Thread] = []

        head_cfg = cam_config.get(self.CAMERA_TOPIC)
        if head_cfg is None:
            raise RuntimeError(f"[Image Server] No '{self.CAMERA_TOPIC}' entry in config.")

        self._responser = ZMQ_Responser(
            self._cam_config,
            port=head_cfg["zmq_request_port"],
        )
        self._zmq_publisher_manager = ZMQ_PublisherManager.get_instance()

        try:
            img_shape = head_cfg.get("image_shape")
            fps = head_cfg.get("fps", 30)
            zmq_port = head_cfg.get("zmq_port")
            enable_depth = head_cfg.get("enable_depth", False)
            serial_number = head_cfg.get("serial_number")
            serial_number = str(serial_number) if serial_number else None

            if not serial_number:
                serials = list_realsense_serial_numbers()
                if not serials:
                    raise RuntimeError("[Image Server] No RealSense cameras found.")
                serial_number = serials[0]
                logger_mp.info(f"[Image Server] Auto-selected RealSense serial {serial_number}")

            self._camera = RealSenseCamera(
                self.CAMERA_TOPIC, serial_number, img_shape, fps,
                zmq_port=zmq_port, enable_depth=enable_depth,
            )
            self._cam_config[self.CAMERA_TOPIC]["intrinsics"] = self._camera.intrinsics_dict
        except Exception:
            self._clean_up()
            raise

        logger_mp.info("[Image Server] Image server started, waiting for client connections...")

    # ----- thread workers -----
    def _update_frames(self):
        interval = 1.0 / self._camera.get_fps()
        next_frame_time = time.monotonic()
        try:
            while not self._stop_event.is_set():
                try:
                    self._camera._update_frame()
                except Exception as e:
                    logger_mp.error(f"[Image Server] Error updating frame: {e}")
                    self._stop_event.set()
                    break
                next_frame_time += interval
                sleep_time = next_frame_time - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_frame_time = time.monotonic()
        except Exception as e:
            logger_mp.error(f"[Image Server] Frame update loop died: {e}")
            self._stop_event.set()

    def _zmq_pub(self):
        interval = 1.0 / self._camera.get_fps()
        next_frame_time = time.monotonic()
        frame_id = itertools.count()
        try:
            while not self._stop_event.is_set():
                parts = self._camera.get_frame_parts()
                if parts is None:
                    logger_mp.warning("[Image Server] head_camera returned no frame.")
                    self._stop_event.set()
                    break

                jpeg_bytes, depth_bytes, meta = parts
                header = {
                    "v": 1,
                    "camera": self.CAMERA_TOPIC,
                    "ts_ns": time.monotonic_ns(),
                    "frame_id": next(frame_id),
                    "jpeg_size": len(jpeg_bytes),
                    "depth_size": len(depth_bytes) if depth_bytes else 0,
                }
                if meta:
                    header.update(meta)
                header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

                if depth_bytes is not None:
                    message = [header_bytes, jpeg_bytes, depth_bytes]
                else:
                    message = [header_bytes, jpeg_bytes]

                self._zmq_publisher_manager.publish(message, self._camera.get_zmq_port())

                next_frame_time += interval
                sleep_time = next_frame_time - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_frame_time = time.monotonic()
        except Exception as e:
            logger_mp.error(f"[Image Server] ZMQ publish loop died: {e}")
            self._stop_event.set()

    def _clean_up(self):
        try:
            self._responser.stop()
        except Exception:
            pass

        for t in self._threads:
            if t.is_alive():
                t.join(timeout=1.0)
        self._threads.clear()

        try:
            self._zmq_publisher_manager.close()
        except Exception:
            pass

        if self._camera is not None:
            try:
                self._camera.release()
            except Exception as e:
                logger_mp.error(f"[Image Server] Error releasing camera: {e}")

        logger_mp.info("[Image Server] Clean up completed. Server stopped.")

    # ----- public api -----
    def start(self):
        if self._camera is None:
            logger_mp.error("[Image Server] No camera initialized, cannot start.")
            self._stop_event.set()
            return

        t = threading.Thread(target=self._update_frames, daemon=True)
        t.start()
        self._threads.append(t)

        if not self._camera.wait_until_ready(timeout=10.0):
            logger_mp.error("[Image Server] head_camera ready timeout after 10s.")
            self._stop_event.set()
            self._clean_up()
            return
        logger_mp.info("[Image Server] head_camera is ready.")

        t = threading.Thread(target=self._zmq_pub, daemon=True)
        t.start()
        self._threads.append(t)

    def wait(self):
        self._stop_event.wait()
        self._clean_up()

    def stop(self):
        self._stop_event.set()


# ========================================================
# Entry point
# ========================================================
def signal_handler(server, signum, frame):
    logger_mp.info(f"[Image Server] Received signal {signum}, initiating graceful shutdown...")
    server.stop()


def set_performance_mode(cores=(0, 1, 2)):
    import psutil
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity(list(cores))
        logger_mp.info(f"[Performance] CPU Affinity locked to: {list(cores)}")
    except psutil.AccessDenied:
        logger_mp.warning("[Performance] Access Denied: Run as sudo for full optimization")
    except Exception as e:
        logger_mp.error(f"[Performance] Error: {e}")


def main():
    logger_mp.info(
        "\n====================== Image Server Startup Guide ======================\n"
        "This simplified teleimager hosts a single RealSense head camera and\n"
        "publishes multipart RGB(+D) frames over ZMQ PUB.\n"
        "\n"
        "Run:\n"
        "    teleimager-server\n"
        "\n"
        "Use '--cf' to list connected RealSense devices.\n"
        "Configure the camera via 'cam_config_server.yaml'.\n"
        "=========================================================================="
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--cf", action="store_true",
                        help="List connected RealSense cameras and exit.")
    parser.add_argument("--no-affinity", action="store_false", dest="affinity",
                        help="Disable CPU affinity pinning.")
    args = parser.parse_args()

    if args.affinity:
        set_performance_mode(cores=(0, 1, 2))

    if args.cf:
        serials = list_realsense_serial_numbers()
        logger_mp.info("======================= RealSense Discovery =============================")
        if serials:
            for i, sn in enumerate(serials, 1):
                logger_mp.info(f"  [{i}] serial_number: {sn}")
        else:
            logger_mp.info("  (no RealSense devices detected)")
        logger_mp.info("=========================================================================")
        return

    try:
        with open(CONFIG_PATH, "r") as f:
            cam_config = yaml.safe_load(f)
    except Exception as e:
        logger_mp.error(f"Failed to load configuration file at {CONFIG_PATH}: {e}")
        exit(1)

    server = ImageServer(cam_config)
    server.start()

    signal.signal(signal.SIGINT, functools.partial(signal_handler, server))
    signal.signal(signal.SIGTERM, functools.partial(signal_handler, server))

    logger_mp.info("[Image Server] Running... Press Ctrl+C to exit.")
    server.wait()

    # usbhub plugout may cause block process exit, no better solution for now
    time.sleep(0.5)
    os.killpg(os.getpgrp(), 9)


if __name__ == "__main__":
    main()
