# import cv2
# import mediapipe as mp
# import numpy as np

# # Initialize MediaPipe Pose
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=False, 
#                    min_detection_confidence=0.7, 
#                    min_tracking_confidence=0.7)
# mp_drawing = mp.solutions.drawing_utils

# # Get user height
# user_height_cm = float(input("Enter your height in centimeters: "))

# # Video capture
# cap = cv2.VideoCapture(0)

# # State variables
# state = "INITIALIZING"
# calibration_frames = []
# SAMPLE_FRAMES = 30
# baseline_y = None
# min_y = None
# conversion_rate = None  # Will be calculated dynamically

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w = frame.shape[:2]
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(frame_rgb)

#     jump_height_cm = 0.0  # Default value
    
#     if results.pose_landmarks:
#         landmarks = results.pose_landmarks.landmark
        
#         if state == "INITIALIZING":
#             try:
#                 # Get required landmarks with visibility check
#                 nose = landmarks[mp_pose.PoseLandmark.NOSE]
#                 left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
#                 right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
#                 left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
#                 right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

#                 # Check landmark visibility
#                 if all(lm.visibility > 0.7 for lm in [nose, left_ankle, right_ankle]):
#                     # Calculate body height in pixels
#                     nose_y = nose.y * h
#                     ankles_avg_y = (left_ankle.y + right_ankle.y)/2 * h
#                     pixel_height = ankles_avg_y - nose_y
                    
#                     # Store calibration data
#                     calibration_frames.append({
#                         'pixel_height': pixel_height,
#                         'hip_y': (left_hip.y + right_hip.y)/2 * h
#                     })
                    
#                     if len(calibration_frames) >= SAMPLE_FRAMES:
#                         # Calculate conversion rate
#                         avg_pixel_height = np.mean([f['pixel_height'] for f in calibration_frames])
#                         conversion_rate = user_height_cm / avg_pixel_height
                        
#                         # Set baseline hip position
#                         baseline_y = np.mean([f['hip_y'] for f in calibration_frames])
#                         min_y = baseline_y
#                         state = "TRACKING"
#             except (IndexError, KeyError):
#                 pass

#         elif state == "TRACKING":
#             # Track hip position for jump height
#             hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
#             hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
#             current_y = (hip_left.y + hip_right.y)/2 * h
            
#             if current_y < min_y:
#                 min_y = current_y
#             if baseline_y and min_y and conversion_rate:
#                 jump_height_cm = (baseline_y - min_y) * conversion_rate

#     # Draw pose landmarks
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#     # Display information
#     if state == "INITIALIZING":
#         status_text = f"Calibrating: {len(calibration_frames)}/{SAMPLE_FRAMES} - Stand straight!"
#         cv2.putText(frame, status_text, (10, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     else:
#         cv2.putText(frame, f"Jump Height: {jump_height_cm:.1f} cm", (10, 30),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         cv2.putText(frame, f"Conv Rate: {conversion_rate:.4f} cm/px", (10, 60),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     cv2.imshow('Dynamic Jump Height Measurement', frame)

#     # Handle key presses
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('r'):
#         state = "INITIALIZING"
#         calibration_frames = []
#         conversion_rate = None
#     elif key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp
import numpy as np
import av

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                   min_detection_confidence=0.7,
                   min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

class JumpHeightProcessor(VideoProcessorBase):
    def __init__(self):
        self.state = "INITIALIZING"
        self.calibration_frames = []
        self.baseline_y = None
        self.min_y = None
        self.conversion_rate = None
        self.user_height_cm = None
        self.SAMPLE_FRAMES = 30

    def set_user_height(self, height):
        self.user_height_cm = height
        self.reset_state()

    def reset_state(self):
        self.state = "INITIALIZING"
        self.calibration_frames = []
        self.baseline_y = None
        self.min_y = None
        self.conversion_rate = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        jump_height_cm = 0.0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            if self.state == "INITIALIZING" and self.user_height_cm:
                try:
                    nose = landmarks[mp_pose.PoseLandmark.NOSE]
                    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

                    if all(lm.visibility > 0.7 for lm in [nose, left_ankle, right_ankle]):
                        nose_y = nose.y * h
                        ankles_avg_y = (left_ankle.y + right_ankle.y)/2 * h
                        pixel_height = ankles_avg_y - nose_y
                        
                        self.calibration_frames.append({
                            'pixel_height': pixel_height,
                            'hip_y': (left_hip.y + right_hip.y)/2 * h
                        })
                        
                        if len(self.calibration_frames) >= self.SAMPLE_FRAMES:
                            avg_pixel_height = np.mean([f['pixel_height'] for f in self.calibration_frames])
                            self.conversion_rate = self.user_height_cm / avg_pixel_height
                            self.baseline_y = np.mean([f['hip_y'] for f in self.calibration_frames])
                            self.min_y = self.baseline_y
                            self.state = "TRACKING"

                except (IndexError, KeyError):
                    pass

            elif self.state == "TRACKING":
                hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                current_y = (hip_left.y + hip_right.y)/2 * h
                
                if current_y < self.min_y:
                    self.min_y = current_y
                if self.baseline_y and self.min_y and self.conversion_rate:
                    jump_height_cm = (self.baseline_y - self.min_y) * self.conversion_rate

            # Draw landmarks
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Add text overlay
        if self.state == "INITIALIZING":
            text = f"Calibrating: {len(self.calibration_frames)}/{self.SAMPLE_FRAMES} - Stand straight!"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(img, f"Jump Height: {jump_height_cm:.1f} cm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if self.conversion_rate:
                cv2.putText(img, f"Conv Rate: {self.conversion_rate:.4f} cm/px", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Real-Time Jump Height Measurement")
    
    # Initialize processor in session state
    if 'processor' not in st.session_state:
        st.session_state.processor = JumpHeightProcessor()

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        user_height = st.number_input("Enter your height (cm):", min_value=100, max_value=250, value=170)
        if st.button("Set Height"):
            st.session_state.processor.set_user_height(user_height)
        
        if st.button("Reset Calibration"):
            st.session_state.processor.reset_state()

    # Instructions
    st.markdown("""
    **Instructions:**
    1. Enter your height in centimeters
    2. Click 'Set Height'
    3. Stand straight in frame for calibration
    4. Jump vertically to measure height
    5. Click 'Reset Calibration' for new measurements
    """)

    # Webcam stream
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=JumpHeightProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.set_user_height(user_height)

if __name__ == "__main__":
    main()
