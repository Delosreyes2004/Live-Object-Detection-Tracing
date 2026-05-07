import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st
import av
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

st.set_page_config(
    page_title="Live Object Detection & Tracing",
    layout="wide",
    initial_sidebar_state="expanded"
)

SAVE_DIR = "captured_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 40%, #f1f5f9 100%);
    color: #1e293b;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f1f5f9 0%, #e0e7ff 60%, #c7d2fe 100%);
    border-right: 1px solid #cbd5e1;
}
[data-testid="stSidebar"] * {
    color: #1e293b !important;
    font-weight: 500;
}

h1, h2, h3 {
    color: #1d4ed8 !important;
    font-weight: 700 !important;
}

p, label {
    color: #334155 !important;
    font-weight: 500;
}

video {
    border-radius: 15px !important;
    border: 2px solid #3b82f6 !important;
    width: 100% !important;
    height: auto !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
class_names = list(model.names.values())

st.sidebar.title("⚙️ Detection Settings")

save_frames = st.sidebar.checkbox("📸 Save Frames", value=False)
enable_alert = st.sidebar.toggle("🚨 Enable Alert", value=True)

target_object = st.sidebar.selectbox(
    "🎯 Target Object",
    options=class_names,
    index=class_names.index("cell phone") if "cell phone" in class_names else 0
)

confidence = 0.4

st.title("📷 Live Object Detection & Tracing")

st.markdown(
    f"""
    <div class="success-box">
        Monitoring for: <b>{target_object.upper()}</b>
    </div>
    """,
    unsafe_allow_html=True
)

class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.frame_count = 0
        self.last_saved = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))

        results = model.predict(img, conf=confidence, verbose=False)

        object_counts = {}
        target_detected = False

        for r in results:
            boxes = r.boxes

            if boxes is not None:
                for box in boxes:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf_score = float(box.conf[0])

                    object_counts[label] = object_counts.get(label, 0) + 1

                    is_target = label == target_object

                    if is_target:
                        target_detected = True

                    color = (0, 0, 255) if is_target else (0, 255, 0)

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    cv2.putText(
                        img,
                        f"{label} {conf_score:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

        if enable_alert and target_detected:

            cv2.rectangle(img, (0, 0), (640, 60), (0, 0, 180), -1)

            cv2.putText(
                img,
                f"ALERT: {target_object.upper()} DETECTED!",
                (50, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

        y_offset = 90

        for obj, cnt in object_counts.items():

            cv2.putText(
                img,
                f"{obj}: {cnt}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            y_offset += 25

        current_time = time.time()

        if save_frames and current_time - self.last_saved >= 5:

            filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
            filepath = os.path.join(SAVE_DIR, filename)

            cv2.imwrite(filepath, img)
            self.last_saved = current_time

        self.frame_count += 1

        return av.VideoFrame.from_ndarray(img, format="bgr24")

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
}

st.info("📷 Please allow camera access when prompted")

webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    video_processor_factory=VideoProcessor,
    async_processing=False,
)
