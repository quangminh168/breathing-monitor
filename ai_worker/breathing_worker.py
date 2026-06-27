"""
breathing_worker.py
--------------------
Script CHAY DOC LAP tren may co camera. Do nhip tho theo TUNG KHOI 1
PHUT KHONG CHONG LAP (giong test_roi.py goc: gom tin hieu, loc, uoc
luong BPM, roi reset cho khoi tiep theo).

ROI vung nguc gio BAM THEO nguoi, khong con khoa co dinh 1 lan:
    - Chay lai YOLO pose moi --roi-update-every frame (mac dinh 10)
      de cap nhat vi tri ROI theo chuyen dong cua long nguc.
    - Vi tri ROI duoc LAM MUOT (EMA) bang --roi-smoothing de tranh
      giat cuc, vi giat manh se lam nhieu tin hieu motion dung de
      tinh nhip tho.
    - Giua 2 lan cap nhat, ROI giu nguyen vi tri cu (khong reset ve
      None) de motion tracker khong bi mat tin hieu lien tuc.

HIEN THI: 2 cua so desktop cuc bo (KHONG co web server nao):
    - Cua so OpenCV "Chest ROI": video truc tiep + ROI + motion.
    - Cua so matplotlib "Breathing Signal Analysis": ve lai sau moi
      khoi 1 phut (Raw Signal vs Filtered Signal).

Ket qua BPM van duoc POST len backend (POST /api/breathing) moi khoi.

YEU CAU: cac module ban da co san phai nam cung cap voi script nay
(hoac trong PYTHONPATH):
    ai/roi_extractor.py
    ai/motion_tracker.py
    ai/signal_filter.py
    ai/breathing_rate.py

Cach chay:
    python breathing_worker.py --api http://127.0.0.1:5000
    python breathing_worker.py --roi-update-every 5 --roi-smoothing 0.4
"""

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
    parser.add_argument("--roi-update-every", type=int, default=10,
                         help="So frame giua 2 lan chay lai YOLO de cap nhat vi tri ROI "
                              "theo long nguc (mac dinh 10; nho hon = bam sat hon nhung "
                              "ton CPU/GPU hon)")
    parser.add_argument("--roi-smoothing", type=float, default=0.3,
                         help="He so lam muot khi ROI di chuyen, 0-1 (mac dinh 0.3). "
                              "Nho hon = muot hon nhung phan ung cham hon; lon hon = "
                              "bam sat hon nhung de giat, lam nhieu tin hieu motion")
    return parser.parse_args()


def smooth_roi(old_roi, new_roi, alpha):
    """Lam muot vi tri/kich thuoc ROI bang EMA, tranh giat cuc giua 2 lan cap nhat."""
    if old_roi is None:
        return new_roi
    return tuple(
        int(old_roi[i] + alpha * (new_roi[i] - old_roi[i]))
        for i in range(4)
    )


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


def detect_chest_roi(model, roi_extractor, frame):
    """Chay YOLO pose 1 lan, tra ve (x1,y1,x2,y2) hoac None neu khong thay nguoi."""
    result = model(frame, verbose=False)[0]

    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return None

    person = result.keypoints.xy[0]
    ls, rs, lh, rh = person[5], person[6], person[11], person[12]

    x1, y1, x2, y2 = roi_extractor.get_chest_roi(frame, ls, rs, lh, rh)
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    if x2 > x1 and y2 > y1:
        return (x1, y1, x2, y2)
    return None


def main():
    args = parse_args()

    model = YOLO("yolo11n-pose.pt")
    roi_extractor = ROIExtractor()
    tracker = MotionTracker()

    cap = cv2.VideoCapture(args.camera)
    current_roi = None
    frame_idx = 0

    # --- Cua so matplotlib, ve lai sau moi khoi 1 phut ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    raw_line, = ax.plot([], [], label="Raw Signal")
    filtered_line, = ax.plot([], [], linewidth=2, label="Filtered Signal")
    ax.set_title("Respiration Signal Analysis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vertical Motion")
    ax.legend()
    ax.grid(True)
    try:
        fig.canvas.manager.set_window_title("Breathing Signal Analysis")
    except Exception:
        pass
    fig.show()

    block_signal = []
    block_start_time = time.time()
    roi_found_once = False

    print(f"Bat dau do nhip tho theo khoi {args.interval:.0f} giay... (Esc o cua so video de dung)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[worker] Khong doc duoc frame tu camera, thu lai...")
                time.sleep(0.5)
                continue

            frame_idx += 1
            display_frame = frame

            # --- Cap nhat ROI dinh ky (bam theo long nguc), khong khoa co dinh ---
            should_update_roi = (current_roi is None) or (frame_idx % args.roi_update_every == 0)

            if should_update_roi:
                detected = detect_chest_roi(model, roi_extractor, frame)
                if detected is not None:
                    current_roi = smooth_roi(current_roi, detected, args.roi_smoothing)
                    if not roi_found_once:
                        roi_found_once = True
                        print("[worker] Da phat hien vung nguc, bat dau theo doi")
                # Neu khong phat hien duoc (nguoi roi khung hinh tam thoi),
                # GIU NGUYEN current_roi cu thay vi reset -> tin hieu motion
                # khong bi dut doan lien tuc.

            # --- Theo doi motion trong ROI hien tai ---
            if current_roi is not None:
                x1, y1, x2, y2 = current_roi
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

            if plt.fignum_exists(fig.number):
                fig.canvas.flush_events()

            # --- Dung moi --interval giay: tinh BPM, gui API, ve lai figure ---
            now = time.time()
            if now - block_start_time >= args.interval:
                if len(block_signal) >= 10:
                    timestamps = [t for t, _ in block_signal]
                    values = [v for _, v in block_signal]
                    duration = timestamps[-1] - timestamps[0]

                    filtered_signal = SignalFilter.smooth(values, duration_seconds=duration)
                    bpm, _peaks = BreathingRateEstimator.estimate(filtered_signal, duration)

                    push_bpm(args.api, bpm)
                    print(f"[worker] Respiration Rate: {bpm:.2f} BPM")

                    if plt.fignum_exists(fig.number):
                        # Ve theo THOI GIAN THUC (giay, tinh tu luc bat dau
                        # khoi), khong dung chi so frame -- vi cac frame
                        # khong con cach deu nhau ve thoi gian (do YOLO
                        # chay lai dinh ky de cap nhat ROI lam mot so frame
                        # cham hon frame khac).
                        x_values = [t - timestamps[0] for t in timestamps]
                        raw_line.set_data(x_values, values)
                        filtered_line.set_data(x_values, filtered_signal)
                        ax.relim()
                        ax.autoscale_view()
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                else:
                    print("[worker] Khong du du lieu trong khoi nay, bo qua")

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
