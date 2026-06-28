import cv2
from ultralytics import YOLO, solutions
import numpy as np

cap = cv2.VideoCapture("I.Analytical_Graph/data/person-bicycle-car-detection.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(
    "I.Analytical_Graph/2.Pie_Chart/analytics_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w*2, h),  # this is fixed
)

# Initialize YOLO and analytics object
model = YOLO("yolo26n.pt")
analytics = solutions.Analytics(
    show=True,  # display the output
    analytics_type="pie",  # pass the analytics type, could be "pie", "bar" or "area".
    model="yolo26n.pt",  # path to the YOLO26 model file
    classes=[0,1,2],  # display analytics for specific detection classes
)

# Process video
frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    
    yolo_results = model(im0, classes=[0,1,2])
    bbox_frame = yolo_results[0].plot()
    
    frame_count += 1
    try:

        chart_frame = analytics(im0, frame_count).plot_im
    except ValueError as e:
        chart_frame = im0.copy()

    bbox_resized = cv2.resize(bbox_frame, (w, h))
    chart_resized = cv2.resize(chart_frame, (w, h))

    combined = np.hstack((bbox_resized, chart_resized))
    out.write(combined)

cap.release()
out.release()
cv2.destroyAllWindows()  # destroy all opened windows