import cv2

cap = cv2.VideoCapture(1)  # Use correct device index
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 230)  # Set to target high frame rate
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.95)

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  cv2.imshow('High FPS Frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
