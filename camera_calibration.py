import cv2
import numpy as np
import time

# --- Configuration ---
CHECKERBOARD = (8, 6)       # Inner corners of your board (Width, Height)
TARGET_FRAMES = 50          # Total good frames needed to calibrate
COOLDOWN_TIME = 1.5         # Seconds to wait between accepting frames
SQUARE_SIZE = 25.0          # Physical size of a single square side in mm
CAMERA_SRC = 1              # The camera source ID for the camera plugged in. Can be -1, 0, 1, or even 2 depending on the computer configuration

# Stop criteria for refining corner detection accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
DETECTION_SCALE = 0.5
MOVEMENT_THRESHOLD = 2.5
COOLDOWN_TIME = 1.0         # Seconds to wait between accepting frames

# Prepare 3D real-world object point helper
COMMON_CHECKERBOARD_SIZES = [
    CHECKERBOARD,
    (8, 6),
    (7, 5),
    (10, 7),
    (11, 8)
]

def get_object_points(pattern):
    objp_local = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp_local[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp_local *= SQUARE_SIZE
    return objp_local


def find_checkerboard(gray, pattern):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    gray_small = cv2.resize(gray, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE, interpolation=cv2.INTER_LINEAR)

    found, corners = cv2.findChessboardCorners(gray_small, pattern, flags)
    if found:
        return True, corners / DETECTION_SCALE, "standard"

    gray_eq = cv2.equalizeHist(gray_small)
    found, corners = cv2.findChessboardCorners(gray_eq, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if found:
        return True, corners / DETECTION_SCALE, "equalized"

    if hasattr(cv2, "findChessboardCornersSB"):
        found_sb, corners_sb = cv2.findChessboardCornersSB(gray_small, pattern)
        if found_sb:
            return True, corners_sb / DETECTION_SCALE, "SB"

        found_sb, corners_sb = cv2.findChessboardCornersSB(gray_eq, pattern)
        if found_sb:
            return True, corners_sb / DETECTION_SCALE, "SB_eq"

    return False, None, ""

# In-memory arrays to store calibration data points
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# Initialize camera stream
cap = cv2.VideoCapture(CAMERA_SRC)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

img_count = 0
last_save_time = 0
prev_corners = None
gray_shape = None
prev_frame_time = None

print("--- Real-Time Stream Calibration ---")
print("Move the chessboard slowly across the screen.")
print("Cover the center, all 4 corners, and tilt the board slightly.")

while img_count < TARGET_FRAMES:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]

    # Try the configured pattern first, then a few common alternate patterns
    detection_pattern = CHECKERBOARD
    found, corners, detection_method = find_checkerboard(gray, detection_pattern)

    if not found:
        for alt_pattern in COMMON_CHECKERBOARD_SIZES:
            if alt_pattern == CHECKERBOARD:
                continue
            found, corners, detection_method = find_checkerboard(gray, alt_pattern)
            if found:
                detection_pattern = alt_pattern
                break

    current_time = time.time()
    frame_fps = 1.0 / (current_time - prev_frame_time) if prev_frame_time else 0.0
    prev_frame_time = current_time

    status_text = "Searching for chessboard..."
    detected_text = "Detected: NO"

    if found:
        detected_text = "Detected: YES"
        # Draw corners for real-time tracking visualization
        cv2.drawChessboardCorners(display_frame, detection_pattern, corners, found)
        status_text = f"Chessboard detected ({detection_pattern[0]}x{detection_pattern[1]}, {detection_method}). Hold still."
        
        # Check if cooldown has passed
        if (current_time - last_save_time) > COOLDOWN_TIME:
            
            # Motion check: ensure the frame isn't blurry
            if prev_corners is not None:
                movement = np.mean(np.abs(corners - prev_corners))
                
                # If movement is low, the frame is steady and ready
                if movement < MOVEMENT_THRESHOLD:
                    # Refine the pixel coordinates to sub-pixel accuracy
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    # Append directly to RAM instead of saving to disk
                    objpoints.append(get_object_points(detection_pattern))
                    imgpoints.append(corners2)
                    
                    img_count += 1
                    last_save_time = current_time
                    status_text = "FRAME CAPTURED!"
                    
                    # Flash visual feedback on screen
                    cv2.putText(display_frame, status_text, (50, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    status_text = f"Too blurry/moving ({movement:.2f}) — hold still"
            else:
                status_text = "Chessboard found. Move slowly and hold still."
            
            prev_corners = corners.copy()
        else:
            cooldown_remaining = COOLDOWN_TIME - (current_time - last_save_time)
            status_text = f"Captured. Cooling down {cooldown_remaining:.1f}s"
    else:
        prev_corners = None
        status_text = "No chessboard found. Center the board in frame."

    # UI Information Overlay
    cv2.rectangle(display_frame, (30, 410), (680, 500), (0, 0, 0), cv2.FILLED)
    cv2.putText(display_frame, detected_text, (50, 435),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display_frame, f"Status: {status_text}", (50, 462),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Collected: {img_count}/{TARGET_FRAMES}", (50, 488),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(display_frame, f"FPS: {frame_fps:.1f}", (420, 488),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.imshow("Live Calibration Processing", display_frame)

    # Allow user to quit early with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nProcess interrupted by user.")
        break

# Clean up video window immediately to signal data collection is finished
cap.release()
cv2.destroyAllWindows()

# --- Optimization and Final Calculation ---
if img_count >= TARGET_FRAMES:
    print("\n[INFO] Sufficient data collected! Running mathematical calibration matrix...")
    
    # Calculate parameters from memory
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
    
    print("\n--- Calibration Successful! ---")
    print(f"Reprojection Error: {ret:.4f} (Closer to 0 is better)")
    print("\nCamera Matrix (Intrinsic Parameters):\n", mtx)
    print("\nDistortion Coefficients:\n", dist)
    
    # NEW: Generate timestamped filename (Format: camera_calibration_20260801_100019.npz)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"camera_calibration_{timestamp}.npz"
    
    # Save the calibration parameters to disk
    np.savez(filename, mtx=mtx, dist=dist)
    print(f"\nParameters exported successfully to '{filename}'")
else:
    print(f"\n[ERROR] Calibration aborted. Only collected {img_count}/{TARGET_FRAMES} frames.") 