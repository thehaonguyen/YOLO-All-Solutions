import cv2
from ultralytics import solutions

# 1. Initialize video capture
cap = cv2.VideoCapture("data/video.mp4")
assert cap.isOpened(), "Error reading video file"

# Get original video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# 2. Initialize video writer (Note: Ultralytics Analytics defaults to 1280x720 output)
out = cv2.VideoWriter(
    "analytics_output.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1280, 720),
)

# 3. Initialize Analytics object
analytics = solutions.Analytics(
    model="model/yolov8n.pt",       
    analytics_type="pie",    
    show=False,               
    tracker="bytetrack.yaml", 
    conf=0.25,
    # classes=[0]  # Uncomment to filter specific classes (e.g., 0 for person)
)

# --- WINDOW CONFIGURATION (ALLOW RESIZING VIA MOUSE) ---
win_box = "1. YOLO Video Tracking"
win_chart = "2. Analytics Chart"

# Enable WINDOW_NORMAL to allow dynamic resizing of windows using the mouse
cv2.namedWindow(win_box, cv2.WINDOW_NORMAL)
cv2.namedWindow(win_chart, cv2.WINDOW_NORMAL)

# Set initial display size (640x640 aspect ratio for a balanced view)
cv2.resizeWindow(win_box, 640, 640)
cv2.resizeWindow(win_chart, 640, 640)
# -------------------------------------------------------

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if success:
        frame_count += 1
        
        # Create a clean copy of the frame to maintain high tracking accuracy
        im0_clean = im0.copy()
        analytics_results = analytics(im0_clean, frame_count)
        
        # Extract tracking results and plot bounding boxes onto the original frame
        if hasattr(analytics.model, 'predictor') and analytics.model.predictor.results:
            latest_results = analytics.model.predictor.results[0]
            im0_with_boxes = latest_results.plot()
        else:
            im0_with_boxes = im0  # Fallback to original frame if no targets are detected
            
        final_output = analytics_results.plot_im
        
        # Display outputs in their respective configured windows
        cv2.imshow(win_box, im0_with_boxes)
        cv2.imshow(win_chart, final_output)
        
        # Save the analytics visualization frame to the output file
        out.write(final_output)  
        
        # Break the loop early if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release all resources and close windows
cap.release()
out.release()
cv2.destroyAllWindows()