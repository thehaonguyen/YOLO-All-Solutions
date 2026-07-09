import cv2
from ultralytics import solutions

# 1. Initialize video capture with your default data directory
cap = cv2.VideoCapture("data/video.mp4")
assert cap.isOpened(), "Error reading video file"

# Define region points as a dictionary for multi-region tracking
region_points = {
    "region-01": [(50, 50), (250, 50), (250, 250), (50, 250)],
    "region-02": [(640, 640), (780, 640), (780, 720), (640, 720)],
}

# Get original video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# 2. Initialize video writer
video_writer = cv2.VideoWriter("region_counting.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 3. Initialize Region Counter object (Set show=False to prevent window conflicts)
regioncounter = solutions.RegionCounter(
    model="model/yolov8n.pt",  # Updated path to your default YOLO model directory
    region=region_points,     # Pass the multi-region configuration dictionary
    show=False,               # Disable built-in window to allow full resizing control
)

# --- WINDOW CONFIGURATION (ALLOW RESIZING VIA MOUSE) ---
win_name = "YOLO Multi-Region Counting Analytics"

# Enable WINDOW_NORMAL to allow dynamic resizing of the window using the mouse
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Set initial display window size to a balanced 640x640 resolution
cv2.resizeWindow(win_name, 640, 640)
# -------------------------------------------------------

# 4. Process video frames
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    # Process frame and handle object calculations per region
    results = regioncounter(im0)

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