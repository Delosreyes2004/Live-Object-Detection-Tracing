import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2
import time
import os

st.set_page_config(
    page_title="Live Object Detection & Tracing", 
    layout="wide",
    initial_sidebar_state="expanded"
)

SAVE_DIR = "captured_frames"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    padding: 1rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #bfdbfe, #93c5fd);
}
[data-testid="stSidebar"] * {
    color: #1e40af !important;
    font-weight: 500;
}

h1 { font-size: clamp(1.8rem, 4vw, 2.5rem) !important; }
h2, h3 { font-size: clamp(1.2rem, 3vw, 1.5rem) !important; }
p, label { 
    color: #1e3a8a !important;
    font-weight: 500;
    font-size: clamp(0.9rem, 2.5vw, 1rem) !important;
}

video {
    border-radius: 12px !important;
    border: 2px solid #60a5fa !important;
    max-width: 100% !important;
    height: auto !important;
}

.st-emotion-cache-1r4b6p {
    max-width: 100% !important;
    padding: 1rem !important;
}

@media (max-width: 768px) {
    .stApp {
        padding: 0.5rem !important;
    }
    [data-testid="stSidebar"] {
        width: 280px !important;
    }
    h1 { font-size: 1.8rem !important; }
    .css-1d391kg {
        padding: 0.5rem !important;
    }
}

@media (max-width: 1024px) and (min-width: 769px) {
    .stApp {
        padding: 0.75rem !important;
    }
}

.stSelectbox > div {
    max-width: 100% !important;
}

.stCheckbox {
    padding: 0.5rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
class_names = list(model.names.values())

st.sidebar.title("⚙️ Settings")
save_frames = st.sidebar.checkbox("Save Frames", False)
st.sidebar.markdown("---")
enable_alert = st.sidebar.toggle("Enable Alert", True)

target_object = st.sidebar.selectbox(
    "Choose object to alert:", 
    options=class_names, 
    index=class_names.index("cell phone") if "cell phone" in class_names else 0
)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))

        results = model.track(img, persist=True, verbose=False)

        object_counts = {}
        target_detected = False

        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]

                    object_counts[label] = object_counts.get(label, 0) + 1

                    is_target = (label == target_object)
                    if is_target:
                        target_detected = True

                    color = (0, 0, 255) if is_target else (0, 255, 0)

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"{label}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if enable_alert and target_detected:
            cv2.rectangle(img, (0, 0), (640, 60), (0, 0, 150), -1) 
            cv2.putText(img, f"⚠️ ALERT: {target_object.upper()} DETECTED!",
                        (70, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        y_offset = 80
        for obj, cnt in object_counts.items():
            cv2.putText(img, f"{obj}: {cnt}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25

        if save_frames and self.frame_count % 60 == 0:
            file_path = os.path.join(SAVE_DIR, f"capture_{int(time.time())}.jpg")
            cv2.imwrite(file_path, img)

        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("📷 Live Object Detection & Tracing")
st.write(f"Monitoring for: **{target_object}**")

webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_html_attrs={
        "autoPlay": True,
        "playsInline": True,
        "style": {"width": "100%", "border-radius": "10px", "border": "2px solid #42a5f5", "max-width": "100%", "height": "auto"},
    },
)
