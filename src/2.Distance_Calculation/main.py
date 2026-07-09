import cv2
import math
from ultralytics import solutions

# Global variables to store tracked Object IDs instead of static pixels
selected_box_ids = []

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function to detect if a click falls inside any tracked object's bounding box.
    """
    global selected_box_ids
    if event == cv2.EVENT_LBUTTONDOWN:
        # Access the current frame's tracking data passed via param
        current_boxes = param.get("boxes", [])
        
        for box in current_boxes:
            # Get bounding box coordinates and track ID
            # box.xyxy[0] contains [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if box.id is not None:
                track_id = int(box.id[0])
                
                # Check if the clicked (x, y) is inside this bounding box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if len(selected_box_ids) >= 2:
                        selected_box_ids = [] # Reset selection if 2 objects are already selected
                    
                    if track_id not in selected_box_ids:
                        selected_box_ids.append(track_id)
                    break # Stop checking other boxes once a match is found

# 1. Initialize video capture
cap = cv2.VideoCapture("data/video.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# 2. Initialize video writer
video_writer = cv2.VideoWriter("distance_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 3. Initialize Distance Calculation object
distancecalculator = solutions.DistanceCalculation(
    model="model/yolov8n.pt",  
    show=False,
    classes = [0],
    tracker="bytetrack.yaml",
    conf=0.15,
)

# --- WINDOW CONFIGURATION & MOUSE BINDING ---
win_name = "YOLO Object-Locked Distance Calculation"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 640, 640)

# A dictionary shared with the mouse callback to pass dynamic tracking data
mouse_param = {"boxes": []}
cv2.setMouseCallback(win_name, mouse_callback, mouse_param)
# --------------------------------------------

# 4. Process video frames
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    # Process frame through YOLO distance calculation
    results = distancecalculator(im0)
    output_frame = results.plot_im.copy()

    # Safely extract tracking boxes from the internal predictor model
    current_boxes = []
    if hasattr(distancecalculator.model, 'predictor') and distancecalculator.model.predictor.results:
        current_boxes = distancecalculator.model.predictor.results[0].boxes
        # Update shared parameters for the mouse click event
        mouse_param["boxes"] = current_boxes

    # Dictionary to store the center points of the selected IDs in the current frame
    active_centers = {}

    # Calculate center points for all tracked objects in this frame
    for box in current_boxes:
        if box.id is not None:
            track_id = int(box.id[0])
            if track_id in selected_box_ids:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Calculate the center (X, Y) of the bounding box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                active_centers[track_id] = (cx, cy)

    # --- DYNAMIC DISTANCE CALCULATION BETWEEN LOCKED OBJECTS ---
    # Draw dots on the moving centers of selected objects
    for track_id, center in active_centers.items():
        cv2.circle(output_frame, center, 8, (0, 0, 255), -1) # Red tracking anchor dot

    # If both selected objects are visible in the current frame, calculate and draw distance
    if len(selected_box_ids) == 2:
        id1, id2 = selected_box_ids[0], selected_box_ids[1]
        
        if id1 in active_centers and id2 in active_centers:
            pt1, pt2 = active_centers[id1], active_centers[id2]
            
            # Draw dynamic line following the objects
            cv2.line(output_frame, pt1, pt2, (0, 255, 0), 3)
            
            # Dynamic Euclidean distance calculation
            pixel_distance = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            
            # Display real-time updated distance text
            mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv2.putText(output_frame, f"{pixel_distance:.1f} px", mid_point, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    # -----------------------------------------------------------

    # Display the dynamic tracking output
    cv2.imshow(win_name, output_frame)
    video_writer.write(output_frame)  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()