

import argparse
import time

import cv2
import matplotlib.pyplot as plt
import requests
from ultralytics import YOLO

from ai.roi_extractor import ROIExtractor
from ai.motion_tracker import MotionTracker
from ai.signal_filter import SignalFilter
from ai.breathing_rate import BreathingRateEstimator


def parse_args():
    parser = argparse.ArgumentParser(description="Breathing rate worker -> push to API")
    parser.add_argument("--api", default="http://127.0.0.1:5000",
                         help="Base URL cua Flask backend (mac dinh http://127.0.0.1:5000)")
    parser.add_argument("--camera", type=int, default=0,
                         help="Chi so camera cho cv2.VideoCapture (mac dinh 0)")
    parser.add_argument("--interval", type=float, default=60.0,
                         help="Do dai MOI khoi tin hieu KHONG chong lap, don vi giay "
                              "(mac dinh 60 = dung 1 phut, cung la chu ky gui BPM + ve lai figure)")
    return parser.parse_args()


def push_bpm(api_base, bpm_value):
    """Gui gia tri BPM moi len backend qua POST /api/breathing."""
    url = f"{api_base}/api/breathing"
    try:
        resp = requests.post(
            url,
            json={"bpm": round(bpm_value, 2), "source": "camera_ai"},
            timeout=5,
        )
        if resp.status_code == 201:
            print(f"[worker] Da gui BPM={bpm_value:.2f} len API")
        else:
            print(f"[worker] API tra loi {resp.status_code}: {resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"[worker] Khong gui duoc len API (se thu lai o khoi tiep theo): {e}")


def main():
    args = parse_args()

    model = YOLO("yolo11n-pose.pt")
    roi_extractor = ROIExtractor()
    tracker = MotionTracker()

    cap = cv2.VideoCapture(args.camera)
    fixed_roi = None

    # --- Cua so matplotlib, ve lai sau moi khoi 1 phut ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    raw_line, = ax.plot([], [], label="Raw Signal")
    filtered_line, = ax.plot([], [], linewidth=2, label="Filtered Signal")
    ax.set_title("Respiration Signal Analysis")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Vertical Motion")
    ax.legend()
    ax.grid(True)
    try:
        fig.canvas.manager.set_window_title("Breathing Signal Analysis")
    except Exception:
        pass  # khong phai backend nao cung ho tro dat ten cua so
    fig.show()

    block_signal = []
    block_start_time = time.time()

    print(f"Bat dau do nhip tho theo khoi {args.interval:.0f} giay... (Esc o cua so video de dung)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[worker] Khong doc duoc frame tu camera, thu lai...")
                time.sleep(0.5)
                continue

            display_frame = frame

            # --- Khoa ROI vung nguc (chi chay YOLO khi chua khoa duoc ROI) ---
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
                        print("[worker] ROI da duoc khoa")

            # --- Theo doi motion trong ROI da khoa ---
            if fixed_roi is not None:
                x1, y1, x2, y2 = fixed_roi
                roi = frame[y1:y2, x1:x2]
                motion = tracker.process(roi)

                if motion is not None:
                    block_signal.append((time.time(), motion))
                    cv2.putText(
                        display_frame, f"Motion: {motion:.6f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # --- Cua so video ---
            cv2.imshow("Chest ROI", display_frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

            # Giu cua so matplotlib phan hoi (resize/dong cua so) ma khong
            # can ve lai du lieu - du lieu chi thuc su cap nhat moi khoi.
            if plt.fignum_exists(fig.number):
                fig.canvas.flush_events()

            # --- Dung moi --interval giay: tinh BPM, gui API, ve lai figure ---
            now = time.time()
            if now - block_start_time >= args.interval:
                if len(block_signal) >= 10:
                    timestamps = [t for t, _ in block_signal]
                    values = [v for _, v in block_signal]
                    duration = timestamps[-1] - timestamps[0]

                    filtered_signal = SignalFilter.smooth(values)
                    bpm, _peaks = BreathingRateEstimator.estimate(filtered_signal, duration)

                    push_bpm(args.api, bpm)
                    print(f"[worker] Respiration Rate: {bpm:.2f} BPM")

                    if plt.fignum_exists(fig.number):
                        x_values = list(range(len(values)))
                        raw_line.set_data(x_values, values)
                        filtered_line.set_data(x_values, filtered_signal)
                        ax.relim()
                        ax.autoscale_view()
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                else:
                    print("[worker] Khong du du lieu trong khoi nay, bo qua")

                # Reset cho khoi tiep theo (KHONG chong lap voi khoi vua xong)
                block_signal = []
                block_start_time = now

    except KeyboardInterrupt:
        print("[worker] Da dung theo yeu cau (Ctrl+C)")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.close(fig)


if __name__ == "__main__":
    main()
