import systemd.daemon
import asyncio
import cv2
import os
import time
import threading
import numpy as np
from pyapriltags import Detector

try:
    from networktables import NetworkTables
    NT_AVAILABLE = True
except ImportError:
    NetworkTables = None
    NT_AVAILABLE = False


def load_camera_params(npz_path):
    """Load camera calibration from a .npz file.

    Returns (camera_matrix, dist_coeffs, camera_params_list)
    where camera_params_list is [fx, fy, cx, cy] suitable for pyapriltags.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        print("[calib] available keys:", data.files)
        # common key names for camera matrix
        cam_mat = None
        for k in ("camera_matrix", "mtx", "K"):
            if k in data:
                cam_mat = data[k]
                break
        if cam_mat is None:
            raise KeyError(f"No camera matrix found in {npz_path}; keys={data.files}")

        # common key names for distortion
        dist = None
        for k in ("dist_coeffs", "dist", "distortion"):
            if k in data:
                dist = data[k]
                break

        # derive fx,fy,cx,cy from camera matrix if possible
        try:
            fx = float(cam_mat[0, 0])
            fy = float(cam_mat[1, 1])
            cx = float(cam_mat[0, 2])
            cy = float(cam_mat[1, 2])
            cam_params = [fx, fy, cx, cy]
        except Exception:
            cam_params = None

        return cam_mat, dist, cam_params

class ELPCameraStream:
    def __init__(self, src=1, exposure_val=-7, max_reconnect_attempts=5, reconnect_delay=2.0):
        self.src = src
        self.exposure_val = exposure_val
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self.stream = None
        self.grabbed = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

        self.connection_state = False
        self.last_error = None
        self.reconnect_attempts = 0
        self.last_frame_time = None
        self.last_frame_interval = 0.0
        self.frame_count = 0

        self.open_camera()

    def open_camera(self):
        if self.stream is not None:
            self.stream.release()

        self.stream = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.stream.set(cv2.CAP_PROP_FPS, 230)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.95)

        if self.exposure_val is not None:
            self.stream.set(cv2.CAP_PROP_EXPOSURE, self.exposure_val)

        opened = self.stream.isOpened()
        self.grabbed, self.frame = self.stream.read()
        self.connection_state = bool(self.grabbed)
        self.last_error = None if self.grabbed else "Initial frame capture failed"
        print(f"[Camera] open_camera src={self.src} opened={opened} grabbed={self.grabbed} frame_shape={None if self.frame is None else self.frame.shape} last_error={self.last_error}")
        return self.grabbed

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def reconnect(self):
        attempt = 0
        while not self.stopped and attempt < self.max_reconnect_attempts:
            attempt += 1
            self.reconnect_attempts = attempt
            print(f"[Camera] Lost capture. Attempting reconnect {attempt}/{self.max_reconnect_attempts}...")
            time.sleep(self.reconnect_delay)
            if self.open_camera():
                print("[Camera] Reconnected successfully.")
                self.last_error = None
                return True

        self.last_error = "Failed to reconnect after maximum attempts"
        print("[Camera] Failed to reconnect after multiple attempts.")
        return False

    def update(self):
        while not self.stopped:
            if self.stream is None or not self.stream.isOpened():
                self.reconnect()
                time.sleep(0.5)
                continue

            grabbed, frame = self.stream.read()
            if not grabbed or frame is None:
                with self.lock:
                    self.grabbed = False
                    self.connection_state = False
                    self.last_error = "Frame grab failed"
                print("[Camera] Frame grab failed, trying to recover...")
                self.reconnect()
                continue

            now = time.time()
            with self.lock:
                self.grabbed = True
                self.frame = frame
                self.connection_state = True
                self.last_error = None
                self.reconnect_attempts = 0
                self.last_frame_interval = now - self.last_frame_time if self.last_frame_time is not None else 0.0
                self.last_frame_time = now
                self.frame_count += 1

            time.sleep(0.001)

    def read(self):
        with self.lock:
            return None if not self.grabbed else self.frame

    def is_connected(self):
        with self.lock:
            return self.connection_state and self.stream is not None and self.stream.isOpened()

    def get_status(self):
        with self.lock:
            return {
                "camera_connected": self.is_connected(),
                "camera_frame_count": self.frame_count,
                "camera_frame_rate": 1.0 / self.last_frame_interval if self.last_frame_interval > 0 else 0.0,
                "camera_frame_interval": self.last_frame_interval,
                "camera_last_frame_time": self.last_frame_time,
                "camera_reconnect_attempts": self.reconnect_attempts,
                "camera_last_error": self.last_error or "none"
            }
        
    def set_exposure(self, val):
        if self.stream is not None and self.stream.isOpened():
            self.stream.set(cv2.CAP_PROP_EXPOSURE, val)

    def stop(self):
        self.stopped = True
        if self.stream is not None:
            self.stream.release()

class NetworkTablesPublisher:
    def __init__(self, server="127.0.0.1", table_name="AprilTag"):
        self.enabled = NT_AVAILABLE
        self.table = None

        if not self.enabled:
            print("[NetworkTables] networktables package not installed. Publishing disabled.")
            return

        NetworkTables.initialize(server=server)
        self.table = NetworkTables.getTable(table_name)
        print(f"[NetworkTables] Initialized connection to {server} table '{table_name}'.")

    def publish_tag_pose(self, tag_data):
        if not self.enabled or self.table is None:
            return

        tag_id = tag_data["tag_id"]
        prefix = f"tag_{tag_id}"

        if tag_data["pose_translation"] is not None:
            self.table.putNumberArray(f"{prefix}_pose_translation", tag_data["pose_translation"])
        else:
            self.table.putString(f"{prefix}_pose_translation", "none")

        if tag_data["pose_rotation"] is not None:
            rotation_flat = [float(v) for row in tag_data["pose_rotation"] for v in row]
            self.table.putNumberArray(f"{prefix}_pose_rotation", rotation_flat)
        else:
            self.table.putString(f"{prefix}_pose_rotation", "none")

    def publish_camera_stats(self, camera_stats):
        if not self.enabled or self.table is None:
            return

        self.table.putBoolean("camera_connected", bool(camera_stats.get("camera_connected", False)))
        self.table.putNumber("camera_frame_rate", float(camera_stats.get("camera_frame_rate", 0.0)))
        self.table.putNumber("camera_frame_interval", float(camera_stats.get("camera_frame_interval", 0.0)))
        self.table.putNumber("camera_frame_count", int(camera_stats.get("camera_frame_count", 0)))

        last_frame_time = camera_stats.get("camera_last_frame_time")
        if last_frame_time is not None:
            self.table.putNumber("camera_last_frame_time", float(last_frame_time))
        else:
            self.table.putString("camera_last_frame_time", "none")

        self.table.putNumber("camera_reconnect_attempts", int(camera_stats.get("camera_reconnect_attempts", 0)))
        self.table.putString("camera_last_error", str(camera_stats.get("camera_last_error", "none")))

        self.table.putBoolean("camera_recording", bool(camera_stats.get("camera_recording", False)))
        self.table.putNumber("camera_recording_seconds_remaining", float(camera_stats.get("camera_recording_seconds_remaining", 0.0)))

class VideoRecorder:
    def __init__(self, output_dir="recordings", duration_seconds=180, codec="MJPG", fps=30):
        self.output_dir = output_dir
        self.duration_seconds = duration_seconds
        self.codec = codec
        self.fps = fps
        self.writer = None
        self.start_time = None
        self.recording = False
        self.frame_size = None
        os.makedirs(self.output_dir, exist_ok=True)

    def start(self, frame):
        if self.recording:
            return

        height, width = frame.shape[:2]
        self.frame_size = (width, height)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"match_start_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(filename, fourcc, self.fps, self.frame_size)
        self.start_time = time.time()
        self.recording = self.writer.isOpened()
        if self.recording:
            print(f"[VideoRecorder] Started recording to {filename}")
            self.write(frame)
        else:
            print(f"[VideoRecorder] Failed to open video writer for {filename}")

    def write(self, frame):
        if not self.recording or self.writer is None:
            return
        self.writer.write(frame)

    def stop(self):
        if self.writer is not None:
            self.writer.release()
        if self.recording:
            print("[VideoRecorder] Recording complete")
        self.writer = None
        self.recording = False
        self.start_time = None

    def update(self, frame):
        if not self.recording:
            return
        self.write(frame)
        if time.time() - self.start_time >= self.duration_seconds:
            self.stop()

    def get_status(self):
        if not self.recording:
            return {"camera_recording": False, "camera_recording_seconds_remaining": 0.0}
        elapsed = time.time() - self.start_time
        remaining = max(0.0, self.duration_seconds - elapsed)
        return {"camera_recording": True, "camera_recording_seconds_remaining": remaining}

class NetworkTablesTrigger:
    def __init__(self, server="127.0.0.1", table_name="RoboRIO", key_name="matchStart"):
        self.enabled = NT_AVAILABLE
        self.key_name = key_name
        self.table = None

        if not self.enabled:
            print("[NetworkTablesTrigger] networktables package not installed. Trigger disabled.")
            return

        NetworkTables.initialize(server=server)
        self.table = NetworkTables.getTable(table_name)
        print(f"[NetworkTablesTrigger] Listening for '{self.key_name}' on table '{table_name}'.")

    def should_start_recording(self):
        if not self.enabled or self.table is None:
            return False
        return self.table.getBoolean(self.key_name, False)

class AprilTagPipeline:
    def __init__(self, tag_family="tag16h5", estimate_pose=False, camera_params=None, tag_size=0.7):
        self.detector = Detector(families=tag_family)
        self.estimate_pose = estimate_pose
        self.camera_params = camera_params
        self.tag_size = tag_size
        
        self.latest_results = []
        self.frame_to_process = None
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.process_loop, args=(), daemon=True).start()
        return self

    def submit_frame(self, frame):
        # Thread-safe handoff of the latest raw frame
        with self.lock:
            self.frame_to_process = frame.copy()

    def get_results(self):
        with self.lock:
            return self.latest_results

    def process_loop(self):
        while not self.stopped:
            if self.frame_to_process is None:
                time.sleep(0.001)
                continue
                
            with self.lock:
                gray = cv2.cvtColor(self.frame_to_process, cv2.COLOR_BGR2GRAY)
                self.frame_to_process = None # Clear buffer slot

            if self.estimate_pose and self.camera_params is not None:
                detections = self.detector.detect(
                    gray,
                    estimate_tag_pose=True,
                    camera_params=self.camera_params,
                    tag_size=self.tag_size
                )
            else:
                # OPTIONAL: Pass your calibration parameters to get 3D translation/rotation coordinates
                # example: detections = self.detector.detect(gray, estimate_tag_pose=True, camera_params=[fx, fy, cx, cy], tag_size=0.16)
                detections = self.detector.detect(gray)
            
            with self.lock:
                self.latest_results = detections

    def stop(self):
        self.stopped = True

class RobotTracker:
    def __init__(self, camera_stream, nt_publisher):
        self.camera = camera_stream
        self.publisher = nt_publisher
        self.detector = Detector(families='tag36h11', nthreads=1) # Adjust family as needed

    def process_frame(self):
        """Processes a single frame. Returns True if progress was made."""
        frame = self.camera.read()
        if frame is None:
            return False
            
        # Convert to grayscale for AprilTag detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray)
        
        # Process and publish tag data
        for tag in tags:
            tag_data = {
                "tag_id": tag.tag_id,
                "pose_translation": tag.pose_t.flatten() if tag.pose_t is not None else None,
                "pose_rotation": tag.pose_R if tag.pose_R is not None else None
            }
            self.publisher.publish_tag_pose(tag_data)
            
        # Publish health metrics
        self.publisher.publish_camera_stats(self.camera.get_status())
        return True

def main():
    # 1. Initialize hardware and publishers
    # Attempt to load camera calibration (optional)
    try:
        K, dist, cam_params = load_camera_params("camera_calibration_20260801_184223.npz")
        print(f"[Main] Loaded camera calibration; cam_params={cam_params}")
    except Exception as e:
        K = None
        dist = None
        cam_params = None
        print(f"[Main] Camera calibration not loaded: {e}")

    camera = ELPCameraStream(src=1).start()
    publisher = NetworkTablesPublisher()
    tracker = RobotTracker(camera, publisher)
    
    # 2. Signal systemd that initialization is complete
    systemd.daemon.notify(systemd.daemon.Notification.READY)
    
    # 3. Track watchdog intervals
    WATCHDOG_INTERVAL = 4.0  # Seconds between pings
    last_ping_time = time.time()
    
    print("[Main] System initialized. Entering core tracking loop.")
    
    try:
        while True:
            # Execute processing step
            work_done = tracker.process_frame()
            
            # 4. Watchdog Logic: Ping ONLY if the loop is healthy and moving
            current_time = time.time()
            if current_time - last_ping_time >= WATCHDOG_INTERVAL:
                # OPTIONAL: Only ping if work_done is True (ensures camera is active)
                systemd.daemon.notify(systemd.daemon.Notification.WATCHDOG)
                last_ping_time = current_time
                
            # Tiny sleep to save CPU cycles if no frame was ready
            if not work_done:
                time.sleep(0.001)
                
    except KeyboardInterrupt:
        print("[Main] Terminating tracker application...")
    finally:
        camera.stop()

if __name__ == "__main__":
    main()