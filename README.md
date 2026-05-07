# Live-Object-Detection-Tracing
A high-performance real-time computer vision application built with Streamlit, YOLOv8, and WebRTC. This app allows for browser-based object detection with minimal latency and custom alert features.

✨ Key Features Real-time Inference: Powered by YOLOv8 (Nano) for high-speed detection.

WebRTC Integration: Seamless webcam streaming directly in the browser.

Dynamic Alerts: Visual on-screen warnings when a specific target object (e.g., "cell phone") is detected.

Object Counting: Real-time tally of all detected objects visible in the frame.

Frame Archiving: Automatically saves snapshots to the captured_frames/ directory when enabled.

Responsive UI: Custom CSS styling for a modern, mobile-friendly experience.

🚀 Installation & Setup Clone the repository:

Bash git clone cd Install dependencies:

Bash pip install -r requirements.txt Run the application:

Bash streamlit run app.py
