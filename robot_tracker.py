import cv2
import time
import threading
from pyapriltags import Detector

class ELPCameraStream:
    def __init__(self, src=1, exposure_val=-7):
        # Initialize video capture (Use cv2.CAP_V4L2 on Linux)
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW) 
        
        # Performance Settings
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.stream.set(cv2.CAP_PROP_FPS, 230)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Exposure Settings
        self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.95)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
                break
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame
        
    def set_exposure(self, val):
        self.stream.set(cv2.CAP_PROP_EXPOSURE, val)

    def stop(self):
        self.stopped = True
        self.stream.release()


class AprilTagPipeline:
    def __init__(self, tag_family="tag36h11"):
        self.detector = Detector(families=tag_family)
        
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
            if self.frame_to_process is str or self.frame_to_process is None:
                time.sleep(0.001)
                continue
                
            with self.lock:
                gray = cv2.cvtColor(self.frame_to_process, cv2.COLOR_BGR2GRAY)
                self.frame_to_process = None # Clear buffer slot

            # OPTIONAL: Pass your calibration parameters to get 3D translation/rotation coordinates
            # example: detections = self.detector.detect(gray, estimate_tag_pose=True, camera_params=[fx, fy, cx, cy], tag_size=0.16)
            detections = self.detector.detect(gray)
            
            with self.lock:
                self.latest_results = detections

    def stop(self):
        self.stopped = True


# --- Main Thread Execution ---
# Start hardware camera thread
cam = ELPCameraStream(src=1, exposure_val=-7).start()
time.sleep(1.0) 

# Start vision pipeline thread (default family: tag36h11)
pipeline = AprilTagPipeline(tag_family="tag36h11").start()

cv2.namedWindow("ELP High Speed AprilTag")

prev_time = time.time()
frame_count = 0

while True:
    frame = cam.read()
    if frame is None:
        continue
        
    frame_count += 1
    
    # 1. Send the frame to the background detector thread
    pipeline.submit_frame(frame)
    
    # 2. Get the latest available detections (won't block the loop)
    detections = pipeline.get_results()
    
    # Object collection to store coordinates generated this frame
    frame_coordinates = []

    # 3. Draw detections overlay and generate coordinate profiles
    for r in detections:
        # Extract corner points
        (ptA, ptB, ptC, ptD) = r.corners
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        
        # Center coordinates
        cX, cY = int(r.center[0]), int(r.center[1])

        # --- GENERATE COORDINATE DATA STRUCTURE ---
        tag_data = {
            "tag_id": r.tag_id,
            "center": (cX, cY),
            "corners": {
                "top_left": ptA,
                "top_right": ptB,
                "bottom_right": ptC,
                "bottom_left": ptD
            },
            # If estimate_tag_pose=True is used in the pipeline, these attributes become available:
            "pose_translation": r.pose_t.flatten().tolist() if hasattr(r, 'pose_t') and r.pose_t is not None else None,
            "pose_rotation": r.pose_R.tolist() if hasattr(r, 'pose_R') and r.pose_R is not None else None
        }
        frame_coordinates.append(tag_data)

        # Draw bounding box
        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2)
        
        # Draw center point and ID text
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"ID: {r.tag_id}", (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- ACTIONABLE STEP FOR COORDINATES ---
    # Example action: Print the active centers found in the current frame to terminal
    if frame_coordinates:
        for tag in frame_coordinates:
            print(f"[Coordinates] Tag {tag['tag_id']} Center -> X: {tag['center'][0]}, Y: {tag['center'][1]}")

    # Frame Rate Diagnostics
    curr_time = time.time()
    elapsed = curr_time - prev_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        print(f"Streaming Speed: {fps:.2f} FPS | Tracked Tags: {len(detections)}")
        frame_count = 0
        prev_time = curr_time

    cv2.imshow("ELP High Speed AprilTag", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cam.stop()
cv2.destroyAllWindows()