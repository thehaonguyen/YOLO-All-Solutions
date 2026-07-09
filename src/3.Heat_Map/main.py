import cv2
from ultralytics import solutions

# 1. Initialize video capture
cap = cv2.VideoCapture("data/video.mp4")
assert cap.isOpened(), "Error reading video file"

# Get original video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# 2. Initialize video writer
video_writer = cv2.VideoWriter("heatmap_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Optional region points configurations for counting zones:
# region_points = [(20, 400), (1080, 400)]                                      # Line boundary
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]              # Rectangle zone
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # Polygon zone

# 3. Initialize Heatmap object (Set show=False to prevent display window conflicts)
heatmap = solutions.Heatmap(
    model="model/yolov8n.pt",  # Path to the YOLOv8 model file
    colormap=cv2.COLORMAP_PARULA,  # Color palette applied to the heatmap density
    show=False,                 # Disable built-in window to allow full resizing control
    # region=region_points,     # Uncomment to track counts within defined boundary zones
    classes=[0],               # Generate heatmap targeting specific classes (e.g., 0 for person)
    # imgsz=1280,               # Optional: Resize input frames for faster processing (default is 640)
    conf=0.25                   # Minimum confidence threshold for object detection
)

# --- WINDOW CONFIGURATION (ALLOW RESIZING VIA MOUSE) ---
win_name = "YOLO Real-Time Heatmap Analytics"

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

    # Process frame and generate heatmap layer
    results = heatmap(im0)

    # Display the mapped output frame in our custom interactive window
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