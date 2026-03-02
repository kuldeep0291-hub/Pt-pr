import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import tempfile


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(layout="wide")
st.title("PRATYAKSHA PRAMANA")
st.caption("AI-Based Traffic Violation & Queue Analysis System")


# =========================================================
# VIDEO UPLOAD
# =========================================================
uploaded_video = st.sidebar.file_uploader(
    "Upload Traffic Video (MP4)",
    type=["mp4"]
)

if uploaded_video is None:
    st.info("👈 Upload a traffic video to begin analysis.")
    st.stop()

temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
temp_file.write(uploaded_video.read())
VIDEO_PATH = temp_file.name


# =========================================================
# PARAMETERS
# =========================================================
PIXEL_TO_METER = 0.04
FPS = 30


# =========================================================
# YOLO CLASS MAPPING
# =========================================================
CLASS_NAMES = {
    2: "Car",
    3: "Bike",
    5: "Bus",
    7: "Truck"
}


# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")


yolo = load_yolo()
tracker = DeepSort(max_age=30, n_init=3)


# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("Analysis Controls")

STOP_LINE_Y = st.sidebar.slider("Stop Zone (Y)", 150, 550, 350)

RASH_SPEED_THRESHOLD = st.sidebar.slider(
    "Rash Speed (px/frame)", 3, 20, 10
)

LANE_CHANGE_THRESHOLD = st.sidebar.slider(
    "Lane Change (px)", 20, 120, 60
)

start_analysis = st.sidebar.button("▶ Start Analysis")


# =========================================================
# UI LAYOUT
# =========================================================
col_video, col_stats = st.columns([2, 1])

video_box = col_video.empty()

queue_len_box = col_stats.metric("Queue Length (m)", 0)
visible_vehicle_box = col_stats.metric("Visible Vehicles", 0)
total_vehicle_box = col_stats.metric("Total Vehicles", 0)

rash_vehicle_box = col_stats.metric("Rash Driving", 0)
lane_change_box = col_stats.metric("Lane Changes", 0)

car_box = col_stats.metric("Cars", 0)
bike_box = col_stats.metric("Bikes", 0)
bus_box = col_stats.metric("Buses", 0)
truck_box = col_stats.metric("Trucks", 0)

col_stats.subheader("🚨 Recent Violations")
violation_box = col_stats.empty()


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def crossed_line(prev_y, curr_y, line_y):
    return (prev_y - line_y) * (curr_y - line_y) < 0


def longitudinal_speed(prev, curr):
    return abs(curr[1] - prev[1])


def aggressive_lane_change(prev, curr, threshold):
    dx = abs(curr[0] - prev[0])
    dy = abs(curr[1] - prev[1])
    return dx > threshold and dy < threshold


# =========================================================
# MAIN PROCESSING
# =========================================================
if start_analysis:

    cap = cv2.VideoCapture(VIDEO_PATH)

    trajectories = {}
    track_classes = {}

    total_vehicle_ids = set()
    rash_vehicle_ids = set()
    stop_zone_violators = set()
    violator_ids = set()

    lane_change_count = 0
    violations = []

    vehicle_counter = {
        "Car": 0,
        "Bike": 0,
        "Bus": 0,
        "Truck": 0
    }


    # ================== PROCESS LOOP ==================
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (900, 500))


        # ---------------- YOLO DETECTION ----------------
        results = yolo(frame)[0]

        detections = []
        detected_classes = []

        for box, cls, conf in zip(
            results.boxes.xyxy,
            results.boxes.cls,
            results.boxes.conf
        ):

            cls_id = int(cls)

            if cls_id in CLASS_NAMES:

                x1, y1, x2, y2 = map(int, box)

                detections.append(
                    ([x1, y1, x2 - x1, y2 - y1],
                     conf.item(),
                     "vehicle")
                )

                detected_classes.append(CLASS_NAMES[cls_id])


        # ---------------- DEEPSORT TRACKING ----------------
        tracks = tracker.update_tracks(detections, frame=frame)

        y_positions = []


        for i, track in enumerate(tracks):

            if not track.is_confirmed():
                continue

            tid = track.track_id
            total_vehicle_ids.add(tid)

            x, y, w, h = map(int, track.to_ltrb())

            cx = x + w // 2
            cy = y + h // 2

            y_positions.append(cy)


            # Assign class
            if tid not in track_classes and i < len(detected_classes):

                vtype = detected_classes[i]

                track_classes[tid] = vtype
                vehicle_counter[vtype] += 1


            vtype = track_classes.get(tid, "Unknown")


            # Store trajectory
            if tid not in trajectories:
                trajectories[tid] = []

            trajectories[tid].append((cx, cy))


            # ================= VIOLATION CHECK =================
            if len(trajectories[tid]) >= 2:

                prev = trajectories[tid][-2]
                curr = trajectories[tid][-1]


                # Stop Line
                if crossed_line(prev[1], curr[1], STOP_LINE_Y):

                    if tid not in stop_zone_violators:

                        stop_zone_violators.add(tid)
                        violator_ids.add(tid)

                        violations.append(
                            f"🚨 Stop Zone | {vtype} | QID {tid}"
                        )


                # Rash Driving
                speed = longitudinal_speed(prev, curr)

                if speed > RASH_SPEED_THRESHOLD:

                    if tid not in rash_vehicle_ids:

                        rash_vehicle_ids.add(tid)
                        violator_ids.add(tid)

                        violations.append(
                            f"⚠ Rash Driving | {vtype} | QID {tid}"
                        )


                # Lane Change
                if aggressive_lane_change(
                        prev, curr, LANE_CHANGE_THRESHOLD):

                    lane_change_count += 1
                    violator_ids.add(tid)

                    violations.append(
                        f"⚠ Lane Change | {vtype} | QID {tid}"
                    )


            # ================= DRAW BOX =================
            if tid in violator_ids:
                color = (0, 0, 255)   # Red
            else:
                color = (0, 255, 0)   # Green


            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )


            # Label
            label = f"{vtype} | QID {tid}"

            if tid in violator_ids:
                label = "🚨 VIOLATOR | " + label


            cv2.putText(
                frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )


        # ---------------- QUEUE METRICS ----------------
        if y_positions:

            queue_px = max(y_positions) - min(y_positions)
            queue_m = int(queue_px * PIXEL_TO_METER)

            queue_len_box.metric("Queue Length (m)", queue_m)

            visible_vehicle_box.metric(
                "Visible Vehicles", len(y_positions)
            )


        # ---------------- DASHBOARD ----------------
        total_vehicle_box.metric(
            "Total Vehicles", len(total_vehicle_ids)
        )

        rash_vehicle_box.metric(
            "Rash Driving", len(rash_vehicle_ids)
        )

        lane_change_box.metric(
            "Lane Changes", lane_change_count
        )

        car_box.metric("Cars", vehicle_counter["Car"])
        bike_box.metric("Bikes", vehicle_counter["Bike"])
        bus_box.metric("Buses", vehicle_counter["Bus"])
        truck_box.metric("Trucks", vehicle_counter["Truck"])


        # ---------------- DISPLAY ----------------
        cv2.line(
            frame,
            (0, STOP_LINE_Y),
            (900, STOP_LINE_Y),
            (0, 0, 255),
            2
        )

        video_box.image(frame, channels="BGR")

        violation_box.write("\n".join(violations[-6:]))


    cap.release()
