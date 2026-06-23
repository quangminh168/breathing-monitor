"""
breathing_worker.py
--------------------
Script CHẠY ĐỘC LẬP trên máy có camera. Đo nhịp thở liên tục bằng
YOLO pose + chest ROI + motion tracking (tái sử dụng logic từ
test_roi.py của bạn), rồi PUSH kết quả BPM lên Flask backend qua
POST /api/breathing mỗi vài giây.

YÊU CẦU: các module bạn đã có sẵn phải nằm cùng cấp với script này
(hoặc trong PYTHONPATH):
    ai/roi_extractor.py
    ai/motion_tracker.py
    ai/signal_filter.py
    ai/breathing_rate.py

Khác với test_roi.py (chạy 900 frame rồi dừng, vẽ plot), script này
chạy VÔ HẠN, dùng một cửa sổ tín hiệu trượt (sliding window) để liên
tục tính lại BPM và gửi lên server theo chu kỳ.

Cách chạy:
    python breathing_worker.py
    python breathing_worker.py --api http://192.168.1.10:5000 --debug
    python breathing_worker.py --window 30 --interval 10
"""

import argparse
import time
from collections import deque

import cv2
import requests
from ultralytics import YOLO

from ai_worker.ai.roi_extractor import ROIExtractor
from ai_worker.ai.motion_tracker import MotionTracker
from ai_worker.ai.signal_filter import SignalFilter
from ai_worker.ai.breathing_rate import BreathingRateEstimator


def parse_args():
    parser = argparse.ArgumentParser(description="Breathing rate worker -> push to API")
    parser.add_argument("--api", default="http://localhost:5000",
                         help="Base URL của Flask backend (mặc định http://localhost:5000)")
    parser.add_argument("--camera", type=int, default=0,
                         help="Chỉ số camera cho cv2.VideoCapture (mặc định 0)")
    parser.add_argument("--window", type=float, default=30.0,
                         help="Độ dài cửa sổ tín hiệu dùng để tính BPM, đơn vị giây (mặc định 30)")
    parser.add_argument("--interval", type=float, default=10.0,
                         help="Khoảng thời gian giữa các lần gửi BPM lên API, đơn vị giây (mặc định 10)")
    parser.add_argument("--debug", action="store_true",
                         help="Hiện cửa sổ debug (ROI, motion) -- chỉ dùng khi test tại chỗ")
    return parser.parse_args()


def push_bpm(api_base, bpm_value):
    """Gửi giá trị BPM mới lên backend qua POST /api/breathing."""
    url = f"{api_base}/api/breathing"
    try:
        resp = requests.post(
            url,
            json={"bpm": round(bpm_value, 2), "source": "camera_ai"},
            timeout=5,
        )
        if resp.status_code == 201:
            print(f"[worker] Đã gửi BPM={bpm_value:.2f} lên API")
        else:
            print(f"[worker] API trả lỗi {resp.status_code}: {resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"[worker] Không gửi được lên API (sẽ thử lại lần sau): {e}")


def main():
    args = parse_args()

    model = YOLO("yolo11n-pose.pt")
    roi_extractor = ROIExtractor()
    tracker = MotionTracker()

    cap = cv2.VideoCapture(args.camera)
    fixed_roi = None

    # Bộ đệm tín hiệu trượt: mỗi phần tử là (timestamp, motion_value)
    signal_buffer = deque()
    last_push_time = time.time()

    print("Bắt đầu đo nhịp thở liên tục... (Ctrl+C để dừng)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[worker] Không đọc được frame từ camera, thử lại...")
                time.sleep(0.5)
                continue

            # --- Khoá ROI vùng ngực (chỉ chạy YOLO khi chưa khoá được ROI) ---
            if fixed_roi is None:
                result = model(frame, verbose=False)[0]

                if result.keypoints is not None and len(result.keypoints.xy) > 0:
                    person = result.keypoints.xy[0]
                    ls, rs, lh, rh = person[5], person[6], person[11], person[12]

                    x1, y1, x2, y2 = roi_extractor.get_chest_roi(frame, ls, rs, lh, rh)
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w, int(x2)), min(h, int(y2))

                    if x2 > x1 and y2 > y1:
                        fixed_roi = (x1, y1, x2, y2)
                        print("[worker] ROI đã được khoá")

            # --- Theo dõi motion trong ROI đã khoá ---
            if fixed_roi is not None:
                x1, y1, x2, y2 = fixed_roi
                roi = frame[y1:y2, x1:x2]
                motion = tracker.process(roi)

                if motion is not None:
                    now = time.time()
                    signal_buffer.append((now, motion))

                    # Bỏ các điểm cũ hơn window_seconds (giữ cửa sổ trượt)
                    while signal_buffer and now - signal_buffer[0][0] > args.window:
                        signal_buffer.popleft()

                if args.debug:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imshow("ROI", roi)

            if args.debug:
                cv2.imshow("Chest ROI", frame)
                if cv2.waitKey(1) == 27:
                    break

            # --- Định kỳ tính BPM từ cửa sổ tín hiệu và gửi lên API ---
            now = time.time()
            if now - last_push_time >= args.interval and len(signal_buffer) >= 10:
                timestamps = [t for t, _ in signal_buffer]
                values = [v for _, v in signal_buffer]
                duration = timestamps[-1] - timestamps[0]

                # Chỉ tính khi đã có đủ dữ liệu (tránh BPM rác lúc mới khởi động)
                if duration >= args.window * 0.5:
                    filtered_signal = SignalFilter.smooth(values)
                    bpm, peaks = BreathingRateEstimator.estimate(filtered_signal, duration)
                    push_bpm(args.api, bpm)

                last_push_time = now

    except KeyboardInterrupt:
        print("[worker] Đã dừng theo yêu cầu (Ctrl+C)")

    finally:
        cap.release()
        if args.debug:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
