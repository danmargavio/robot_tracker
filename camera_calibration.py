import cv2
import numpy as np
import time

# --- Configuration ---
CHECKERBOARD = (9, 7)       # Inner corners of your board (Width, Height)
TARGET_FRAMES = 50          # Total good frames needed to calibrate
COOLDOWN_TIME = 1.5         # Seconds to wait between accepting frames
SQUARE_SIZE = 25.0          # Physical size of a single square side in mm
CAMERA_SRC = 1              # The camera source ID for the camera plugged in. Can be -1, 0, 1, or even 2 depending on the computer configuration

# Stop criteria for refining corner detection accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D real-world object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# In-memory arrays to store calibration data points
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# Initialize camera stream
cap = cv2.VideoCapture(CAMERA_SRC)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
#cap.set(cv2.CAP_PROP_FPS, 60)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

img_count = 0
last_save_time = 0
prev_corners = None
gray_shape = None

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

    # Find the chess board corners
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    current_time = time.time()

    if found:
        # Draw corners for real-time tracking visualization
        cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, found)
        
        # Check if cooldown has passed
        if (current_time - last_save_time) > COOLDOWN_TIME:
            
            # Motion check: ensure the frame isn't blurry
            if prev_corners is not None:
                movement = np.mean(np.abs(corners - prev_corners))
                
                # If movement is low, the frame is steady and ready
                if movement < 1.5:  
                    # Refine the pixel coordinates to sub-pixel accuracy
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    # Append directly to RAM instead of saving to disk
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    
                    img_count += 1
                    last_save_time = current_time
                    
                    # Flash visual feedback on screen
                    cv2.putText(display_frame, "FRAME CAPTURED!", (50, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
            prev_corners = corners.copy()
    else:
        prev_corners = None

    # UI Information Overlay
    cv2.putText(display_frame, f"Data Collected: {img_count}/{TARGET_FRAMES}", (50, 440), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
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