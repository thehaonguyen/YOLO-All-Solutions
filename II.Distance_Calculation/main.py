import cv2
import streamlit as st
from ultralytics import solutions
import os

st.set_page_config(page_title="YOLO Distance Calculator", layout="wide")
st.title("🛣️ YOLO Distance Estimation App")

# 1. Video Source Picker (Let's make a mock file or allow uploads)
video_path = "II.Distance_Calculation/video.mp4"

if not os.path.exists(video_path):
    st.error(f"Video file not found at: {video_path}. Please update the path.")
else:
    # 2. UI Action Button
    if st.button("Start Processing Video"):
        
        # Open the video source
        cap = cv2.VideoCapture(video_path)
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        
        # Setup Video Writer for output tracking saving
        video_writer = cv2.VideoWriter("distance_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Initialize the Ultralytics distance calculation object
        # CRITICAL FOR DOCKER/STREAMLIT: set show=False to avoid headless display errors
        distancecalculator = solutions.DistanceCalculation(
            model="yolo8n.pt",  
            show=False,           
        )

        # Create an empty placeholder container in the Streamlit UI to dynamically stream frames
        frame_placeholder = st.empty()
        
        # Progress bar
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        # 3. Processing loop
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                st.write("Video processing complete.")
                break

            # Calculate tracking & distance
            # NEW ULTRALYTICS API
            results = distancecalculator.process(im0)
            annotated_frame = results.plot_im

            # Write to output file
            video_writer.write(annotated_frame)

            # Streamlit needs RGB format, but OpenCV reads in BGR
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # 4. Render the frame in real time to the web page UI
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            
            # Update Progress
            frame_count += 1
            progress_bar.progress(min(frame_count / total_frames, 1.0))

        # Cleanup
        cap.release()
        video_writer.release()
        st.success("Processed video saved successfully as 'distance_output.avi'!")