import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("data/video.mp4")
assert cap.isOpened(), "Error reading video file"

# Initialize object cropper
cropper = solutions.ObjectCropper(
    model="model/yolov8n.pt",  # model for object cropping, e.g., yolo26x.pt.
    classes=[0, 2],  # crop specific classes such as person and car with the COCO pretrained model.
    # conf=0.5,  # adjust confidence threshold for the objects.
    # crop_dir="cropped-detections",  # set the directory name for cropped detections
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = cropper(im0)

    # print(results)  # access the output

cap.release()
cv2.destroyAllWindows()  # destroy all opened windows