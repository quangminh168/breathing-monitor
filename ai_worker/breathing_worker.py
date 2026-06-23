"""
breathing_worker.py
--------------------
Script CHẠY ĐỘC LẬP trên máy có camera. Đo nhịp thở liên tục bằng
YOLO pose + chest ROI + motion tracking.

Hiển thị:
- Video tracking ROI ngực
- Biểu đồ nhịp thở realtime (OpenCV)

Push BPM lên Flask backend định kỳ.
"""

import argparse
import time
from collections import deque

import cv2
import requests
import numpy as np
from ultralytics import YOLO

from ai_worker.ai.roi_extractor import ROIExtractor
from ai_worker.ai.motion_tracker import MotionTracker
from ai_worker.ai.signal_filter import SignalFilter
from ai_worker.ai.breathing_rate import BreathingRateEstimator


# ===================== CHART =====================
def draw_signal_chart(values):
    if len(values) < 10:
        return

    chart = np.zeros((300, 600, 3), dtype=np.uint8)

    vals = np.array(values)

    v_min, v_max = np.min(vals), np.max(vals)

    if v_max - v_min < 1e-6:
        return

    normalized = (vals - v_min) / (v_max - v_min)

    for i in range(1, len(normalized)):
        x1 = int((i - 1) * 600 / len(normalized))
        y1 = int(300 - normalized[i - 1] * 250)

        x2 = int(i * 600 / len(normalized))
        y2 = int(300 - normalized[i] * 250)

        cv2.line(chart, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Breathing Signal", chart)


# ===================== ARGPARSE =====================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:5000")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--window", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


# ===================== PUSH BPM =====================
def push_bpm(api_base, bpm_value):
    url = f"{api_base}/api/breathing"
    try:
        resp = requests.post(
            url,
            json={"bpm": round(bpm_value, 2), "source": "camera_ai"},
            timeout=5,
        )
        if resp.status_code == 201:
            print(f"[worker] BPM sent: {bpm_value:.2f}")
        else:
            print(f"[worker] API error {resp.status_code}")
    except Exception as e:
        print(f"[worker] push failed: {e}")


# ===================== MAIN =====================
def main():
    args = parse_args()

    model = YOLO("yolo11n-pose.pt")
    roi_extractor = ROIExtractor()
    tracker = MotionTracker()

    cap = cv2.VideoCapture(args.camera)

    fixed_roi = None

    signal_buffer = deque()
    chart_buffer = deque(maxlen=200)

    last_push_time = time.time()

    print("Breathing worker started...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # ================= ROI DETECTION =================
            if fixed_roi is None:
                result = model(frame, verbose=False)[0]

                if result.keypoints is not None and len(result.keypoints.xy) > 0:
                    person = result.keypoints.xy[0]

                    ls, rs = person[5], person[6]
                    lh, rh = person[11], person[12]

                    x1, y1, x2, y2 = roi_extractor.get_chest_roi(
                        frame, ls, rs, lh, rh
                    )

                    h, w = frame.shape[:2]

                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w, int(x2)), min(h, int(y2))

                    if x2 > x1 and y2 > y1:
                        fixed_roi = (x1, y1, x2, y2)
                        print("[worker] ROI locked")

            # ================= MOTION =================
            if fixed_roi is not None:
                x1, y1, x2, y2 = fixed_roi
                roi = frame[y1:y2, x1:x2]

                motion = tracker.process(roi)

                if motion is not None:
                    now = time.time()

                    signal_buffer.append((now, motion))
                    chart_buffer.append(motion)

                    # sliding window
                    while signal_buffer and now - signal_buffer[0][0] > args.window:
                        signal_buffer.popleft()

            # ================= VIDEO DISPLAY =================
            if args.debug:
                vis = frame.copy()

                if fixed_roi is not None:
                    x1, y1, x2, y2 = fixed_roi

                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cv2.putText(
                        vis,
                        "Chest ROI Tracking",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                cv2.imshow("Chest Tracking", vis)

            # ================= CHART =================
            if args.debug:
                draw_signal_chart(list(chart_buffer))

            # ================= PUSH BPM =================
            now = time.time()

            if now - last_push_time >= args.interval and len(signal_buffer) >= 10:

                timestamps = [t for t, _ in signal_buffer]
                values = [v for _, v in signal_buffer]

                duration = timestamps[-1] - timestamps[0]

                if duration >= args.window * 0.5:
                    filtered = SignalFilter.smooth(values)

                    bpm, _ = BreathingRateEstimator.estimate(
                        filtered,
                        duration
                    )

                    push_bpm(args.api, bpm)

                last_push_time = now

            # ================= IMPORTANT =================
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()