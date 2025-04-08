# Jump Height Measurement System 🏀📏

A computer vision system that accurately measures vertical jump height using **real-time pose estimation** with dynamic calibration.

## Key Features ✨

- **📊 Dynamic Calibration** - Automatically calculates pixel-to-cm conversion using your actual height
- **⚡ Real-time Processing** - Measures jumps at 30+ FPS with MediaPipe pose estimation
- **👁️ Visual Feedback** - Displays body landmarks and jump metrics in real-time
- **🌐 Web-Based** - Accessible through any modern browser via Streamlit
- **🔄 One-Click Reset** - Easily recalibrate for different users or environments

## Installation & Setup 🛠️

### Prerequisites
- Python 3.8+
- Webcam
- Modern browser (Chrome/Firefox recommended)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/sreehari2430/Jump-Height-Measurement-System.git


# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run detect_height.py.py
