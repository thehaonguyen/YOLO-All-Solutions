import cv2
from ultralytics import solutions

# 1. Initialize video capture with your default data path
cap = cv2.VideoCapture("data/video.mp4")
assert cap.isOpened(), "Error reading video file"

# Get original video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# 2. Initialize video writer
video_writer = cv2.VideoWriter("security_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Email configurations for automated security alert dispatching
from_email = "abc@gmail.com"  # The sender email address
password = "---- ---- ---- ----"  # 16-digit App Password generated via Google App Passwords
to_email = "xyz@gmail.com"    # The recipient email address

# 3. Initialize Security Alarm object (Set show=False to prevent window conflicts)
securityalarm = solutions.SecurityAlarm(
    model="model/yolov8n.pt",  # Updated path to your default YOLO model directory
    show=False,                # Disable built-in window to allow full resizing control
    records=1,                 # Total target detection threshold required to trigger an email alert
)

# Authenticate with the secure SMTP email server
securityalarm.authenticate(from_email, password, to_email)

# --- WINDOW CONFIGURATION (ALLOW RESIZING VIA MOUSE) ---
win_name = "YOLO Real-Time Security Alarm Diagnostics"

# Enable WINDOW_NORMAL to allow dynamic resizing of the window using the mouse
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Set initial display window size to a balanced 640x640 resolution
cv2.resizeWindow(win_name, 640, 640)
# -------------------------------------------------------

# 4. Process video frames
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Process frame and run anomaly detection routines
    results = securityalarm(im0)

    # Display the processed frame in our custom interactive window
    cv2.imshow(win_name, results.plot_im)

    # Save the processed visualization frame to the output file
    video_writer.write(results.plot_im)

    # Break the loop early if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all hardware resources and clear windows
cap.release()
video_writer.release()
cv2.destroyAllWindows()