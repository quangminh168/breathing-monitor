"""
breathing_worker.py
--------------------
Ho tro 2 nguon camera:
    1. Laptop webcam:  --camera 0   (mac dinh)
    2. ESP32-CAM:      --espcam http://192.168.x.x/stream

Khi dung ESP32-CAM, script tu dong:
    - Mo stream MJPEG qua HTTP thay vi USB.
    - Tang do nhieu loc (GaussianBlur 7x7 thay vi 5x5) vi JPEG artifact
      tu ESP32-CAM nhieu hon ảnh raw tu webcam.
    - Tu dong ket noi lai neu ESP32-CAM mat mang hoac reboot.
    - In canh bao neu FPS trung binh qua thap (<5) de nguoi dung biet
      chat luong tin hieu co the bi anh huong.

Do chinh xac cua mo hinh (bandpass 0.1-0.5Hz, prominence, distance)
KHONG thay doi -- cac tham so nay da duoc thiet ke theo thoi gian thuc
(tu tinh fps tu chinh du lieu), nen tu dong thich nghi voi FPS thap
hon cua ESP32-CAM ma khong can chinh tay.
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


# ----------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Breathing rate worker -> push to API")
    parser.add_argument(
        "--api", default="http://127.0.0.1:5000",
        help="Base URL cua Flask backend (mac dinh http://127.0.0.1:5000)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Chi so camera USB/laptop (mac dinh 0). Bo qua neu dung --espcam."
    )
    parser.add_argument(
        "--espcam", type=str, default=None,
        help=(
            "URL MJPEG stream cua ESP32-CAM, vd: http://192.168.1.50/stream. "
            "Neu truyen tham so nay, --camera se bi bo qua."
        )
    )
    parser.add_argument(
        "--interval", type=float, default=60.0,
        help="Do dai MOI khoi tin hieu KHONG chong lap, don vi giay (mac dinh 60)"
    )
    parser.add_argument(
        "--roi-update-every", type=int, default=10,
        help="So frame giua 2 lan chay lai YOLO de cap nhat vi tri ROI (mac dinh 10)"
    )
    parser.add_argument(
        "--roi-smoothing", type=float, default=0.3,
        help="He so lam muot EMA khi ROI di chuyen, 0-1 (mac dinh 0.3)"
    )
    return parser.parse_args()


# ----------------------------------------------------------------
# Mo camera: ho tro ca USB va ESP32-CAM MJPEG qua HTTP
# ----------------------------------------------------------------
def open_capture(args):
    """
    Tra ve (cap, is_espcam).
    is_espcam=True: can xu ly reconnect va kiem tra FPS rieng.
    """
    if args.espcam:
        url = args.espcam.strip()
        print(f"[worker] Dang ket noi toi ESP32-CAM: {url}")
        cap = cv2.VideoCapture(url)
        # Giam buffer cua OpenCV xuong 2 frame: tranh doc frame cu tich
        # dong trong hang cho khi xu ly cham (YOLO nang) -> giam do tre.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if not cap.isOpened():
            raise RuntimeError(
                f"Khong mo duoc stream ESP32-CAM tai {url}\n"
                "Kiem tra: (1) ESP32-CAM da ket noi WiFi chua? "
                "(2) Dung IP dung chua? (3) Mo http://IP/ping tren browser xem co tra 'OK' khong?"
            )
        print(f"[worker] Da ket noi ESP32-CAM: {url}")
        return cap, True

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc camera so {args.camera}")
    print(f"[worker] Da mo webcam so {args.camera}")
    return cap, False


def reopen_espcam(url, max_retries=10, wait_sec=3.0):
    """Thu mo lai stream ESP32-CAM sau khi mat ket noi."""
    for attempt in range(1, max_retries + 1):
        print(f"[worker] Thu ket noi lai ESP32-CAM lan {attempt}/{max_retries}...")
        time.sleep(wait_sec)
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print("[worker] Da ket noi lai ESP32-CAM thanh cong")
                return cap
        cap.release()
    raise RuntimeError(
        f"Khong the ket noi lai ESP32-CAM sau {max_retries} lan thu. "
        "Kiem tra nguon dien va WiFi cua ESP32-CAM."
    )


# ----------------------------------------------------------------
def smooth_roi(old_roi, new_roi, alpha):
    if old_roi is None:
        return new_roi
    return tuple(int(old_roi[i] + alpha * (new_roi[i] - old_roi[i])) for i in range(4))


def detect_chest_roi(model, roi_extractor, frame):
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


def push_bpm(api_base, bpm_value):
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
        print(f"[worker] Khong gui duoc len API: {e}")


# ----------------------------------------------------------------
def main():
    args = parse_args()
    cap, is_espcam = open_capture(args)

    model = YOLO("yolo11n-pose.pt")
    roi_extractor = ROIExtractor()

    # ESP32-CAM: dung blur manh hon (7x7) vi JPEG artifact nhieu hon
    # anh raw tu webcam, Farneback rat nhay voi block artifact JPEG.
    # Webcam: blur nhe (5x5) nhu truoc.
    blur_kernel = (7, 7) if is_espcam else (5, 5)
    tracker = MotionTracker(blur_kernel=blur_kernel)

    current_roi = None
    frame_idx = 0

    # --- Cua so matplotlib ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    raw_line, = ax.plot([], [], alpha=0.5, label="Raw Signal")
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

    # Theo doi FPS thuc te cua ESP32-CAM de canh bao nguoi dung
    fps_counter = 0
    fps_window_start = time.time()
    MIN_ACCEPTABLE_FPS = 5.0

    src_label = f"ESP32-CAM ({args.espcam})" if is_espcam else f"Webcam {args.camera}"
    print(f"[worker] Bat dau do nhip tho qua {src_label}")
    print(f"[worker] Khoi {args.interval:.0f} giay, Esc de dung")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                if is_espcam:
                    print("[worker] Mat ket noi ESP32-CAM, dang thu ket noi lai...")
                    cap.release()
                    cap = reopen_espcam(args.espcam)
                    fps_counter = 0
                    fps_window_start = time.time()
                    continue
                else:
                    print("[worker] Webcam khong tra frame, thu lai...")
                    time.sleep(0.5)
                    continue

            frame_idx += 1
            fps_counter += 1

            # Canh bao neu FPS trung binh trong 10 giay < 5
            elapsed_fps = time.time() - fps_window_start
            if elapsed_fps >= 10.0:
                actual_fps = fps_counter / elapsed_fps
                if is_espcam and actual_fps < MIN_ACCEPTABLE_FPS:
                    print(
                        f"[worker] CANH BAO: FPS trung binh {actual_fps:.1f} < {MIN_ACCEPTABLE_FPS}. "
                        "Tin hieu co the khong du de loc bandpass chinh xac. "
                        "Thu: (1) giam jpeg_quality trong firmware, (2) chuyen sang QVGA."
                    )
                fps_counter = 0
                fps_window_start = time.time()

            display_frame = frame
            should_update_roi = (current_roi is None) or (frame_idx % args.roi_update_every == 0)

            if should_update_roi:
                detected = detect_chest_roi(model, roi_extractor, frame)
                if detected is not None:
                    was_none = current_roi is None
                    current_roi = smooth_roi(current_roi, detected, args.roi_smoothing)
                    if was_none:
                        print("[worker] Da phat hien vung nguc, bat dau theo doi")

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

            # Hien thi nguon camera de phan biet
            label = "ESP32-CAM" if is_espcam else "Webcam"
            cv2.putText(
                display_frame, label, (20, display_frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )

            cv2.imshow("Chest ROI", display_frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

            if plt.fignum_exists(fig.number):
                fig.canvas.flush_events()

            # --- Tinh BPM sau moi khoi --interval giay ---
            now = time.time()
            if now - block_start_time >= args.interval:
                if len(block_signal) >= 10:
                    timestamps = [t for t, _ in block_signal]
                    values = [v for _, v in block_signal]
                    duration = timestamps[-1] - timestamps[0]
                    actual_fps_block = len(values) / duration if duration > 0 else 0

                    # Kiem tra FPS cua khoi: neu qua thap, canh bao truoc khi tinh
                    if actual_fps_block < MIN_ACCEPTABLE_FPS:
                        print(
                            f"[worker] CANH BAO: FPS khoi nay chi {actual_fps_block:.1f} fps "
                            f"({len(values)} frame / {duration:.1f}s). "
                            "BPM se duoc tinh nhung do chinh xac co the thap hon binh thuong."
                        )

                    filtered_signal = SignalFilter.smooth(values, duration_seconds=duration)
                    bpm, _peaks = BreathingRateEstimator.estimate(filtered_signal, duration)

                    push_bpm(args.api, bpm)
                    print(
                        f"[worker] Respiration Rate: {bpm:.2f} BPM "
                        f"(fps={actual_fps_block:.1f}, frames={len(values)}, duration={duration:.1f}s)"
                    )

                    if plt.fignum_exists(fig.number):
                        x_values = [t - timestamps[0] for t in timestamps]
                        raw_line.set_data(x_values, values)
                        filtered_line.set_data(x_values, filtered_signal)
                        ax.relim()
                        ax.autoscale_view()
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                else:
                    print(f"[worker] Khoi nay chi co {len(block_signal)} sample, bo qua")

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
