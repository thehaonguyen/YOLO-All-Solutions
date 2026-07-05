import streamlit as st
import cv2
from ultralytics import solutions
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="YOLO Ultra-Resolution Dashboard", layout="wide")

st.title("YOLO Native Analytics Dashboard (Enhanced Sight)")

# Global COCO reference array
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload your video source (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

video_path = None
if uploaded_file:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_file.read())
    temp_video.close()
    video_path = temp_video.name
    st.sidebar.success(f"Loaded: {uploaded_file.name}")

analytics_type = st.sidebar.selectbox("Select Native Chart Type", ["line", "pie", "bar", "area"])
model_name = st.sidebar.selectbox("Select YOLO Model Weights", ["yolov8n.pt", "yolov8s.pt", "yolov8l.pt"])

# --- Accuracy Optimization Controls ---
conf_threshold = st.sidebar.slider("Model Confidence Threshold", min_value=0.01, max_value=1.00, value=0.25, step=0.01)

# ENHANCEMENT: Choose inference frame sizes. 1280 makes it easier for the AI to see small details!
inference_size = st.sidebar.selectbox("Model Resolution Size (imgsz)", [640, 960, 1280], index=0)

selected_classes = st.sidebar.multiselect(
    "Filter Specific Classes (Leave empty to track all)", 
    COCO_NAMES, 
    default=[]
)

target_class_ids = [COCO_NAMES.index(name) for name in selected_classes if name in COCO_NAMES] or None

# --- Main Page Layout Split ---
video_col, chart_col = st.columns(2)

with video_col:
    st.subheader("📹 Live Tracking Video")
    video_placeholder = st.empty()

with chart_col:
    st.subheader("📊 YOLO Native Graph")
    chart_placeholder = st.empty()

progress_bar = st.progress(0)

# --- Processing Pipeline ---
if st.sidebar.button("Run Live Dashboard", use_container_width=True) and video_path:
    if not os.path.exists(video_path):
        st.sidebar.error("Video path invalid.")
        st.stop()

    os.makedirs("model", exist_ok=True)
    model_folder_path = os.path.join("model", model_name)

    cap = cv2.VideoCapture(video_path)
    
    # 1. Initialize native analytics manager using optimized settings
    analytics_manager = solutions.Analytics(
        show=False, 
        analytics_type=analytics_type,
        model=model_folder_path,
        classes=target_class_ids,
        conf=conf_threshold,
        imgsz=inference_size # Directs chart analysis frame size
    )
    
    # 2. Separate base model instance using cached loading
    @st.cache_resource
    def load_base_model(path):
        return YOLO(path)
    base_model = load_base_model(model_folder_path)

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_count += 1
            
            # Extract chart imagery from manager layer
            chart_data = analytics_manager(frame, frame_count)
            chart_frame = cv2.cvtColor(chart_data.plot_im, cv2.COLOR_BGR2RGB)
            
            # ENHANCEMENT: Explicitly pass 'imgsz' to the prediction loop to boost tracking performance
            video_results = base_model(frame, classes=target_class_ids, conf=conf_threshold, imgsz=inference_size)[0]
            video_frame = cv2.cvtColor(video_results.plot(), cv2.COLOR_BGR2RGB)
            
            # Send them to their respective columns on the website (auto-scaled to fit nicely layout-wise)
            video_placeholder.image(video_frame, use_container_width=True)
            chart_placeholder.image(chart_frame, use_container_width=True)

            if total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
                
        cap.release()
        st.success("🎉 Processing completed!")
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
    finally:
        if uploaded_file and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass
else:
    if not video_path:
        st.info("👈 Please upload a video source file on the configuration sidebar to activate the layout canvas.")