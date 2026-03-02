# PRATYAKSHA PRAMANA (Pt-pr)

AI-Based Traffic Violation & Queue Analysis System built with Streamlit, YOLOv8, and DeepSORT.

## Features
- Upload an MP4 traffic video and run analysis in the browser
- Vehicle detection (Car, Bike, Bus, Truck) using YOLOv8
- Multi-object tracking using DeepSORT
- Live metrics dashboard:
  - Queue length (meters)
  - Visible vehicles and total unique vehicles
  - Rash driving count and lane change count
  - Vehicle-type counts
- Violation detection:
  - Stop-zone crossing
  - Rash driving (speed threshold)
  - Aggressive lane changes

## Tech Stack
- Python
- Streamlit
- OpenCV (cv2)
- NumPy
- Ultralytics YOLO (yolov8n.pt)
- deep-sort-realtime (DeepSort)

## How it works (high level)
1. Upload a traffic video (MP4) in the sidebar.
2. Click **Start Analysis**.
3. Each frame is resized and passed through YOLO to detect vehicles.
4. DeepSORT assigns consistent IDs (QID) across frames.
5. The app computes:
   - Queue length from the spread of vehicle Y positions (px -> meters)
   - Speed from per-frame longitudinal motion
   - Lane-change events from lateral motion thresholds
6. Violators are highlighted in red and recent violations are listed.

## Controls
- **Stop Zone (Y):** horizontal line position for stop-zone detection
- **Rash Speed (px/frame):** threshold for rash driving detection
- **Lane Change (px):** lateral threshold for aggressive lane-change detection

## Run locally
```bash
pip install streamlit opencv-python numpy ultralytics deep-sort-realtime
streamlit run app.py
```

## Notes
- Requires downloading YOLOv8 weights (`yolov8n.pt`) on first run.
- Thresholds and pixel-to-meter conversion may need calibration for your camera setup.