import cv2
import numpy as np


class MotionTracker:

    def __init__(self, blur_kernel=(5, 5)):
        """
        blur_kernel: kich thuoc Gaussian blur truoc khi tinh optical flow.
        - (5, 5): phu hop cho webcam laptop (anh raw it artifact).
        - (7, 7): phu hop cho ESP32-CAM (JPEG artifact nhieu hon, can
          lam mo manh hon de Farneback khong "bam" vao block artifact).
        - KHONG nen tang qua (9,9) vi se lam mo canh, Farneback se kho
          bam duoc chuyen dong tinh te cua long nguc.
        """
        self.blur_kernel = blur_kernel
        self.prev_gray = None
        self.signal = []

    def process(self, roi):

        if roi is None:
            return None

        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (200, 200))

        # Gaussian blur: giam nhieu cam bien / JPEG artifact truoc khi
        # tinh optical flow. Kernel duoc truyen vao luc khoi tao de co
        # the chinh khac nhau cho webcam vs ESP32-CAM.
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            0.5,   # pyr_scale
            3,     # levels
            15,    # winsize: cua so tim kiem tuong quan
            3,     # iterations
            5,     # poly_n
            1.2,   # poly_sigma
            0
        )

        vertical_motion = flow[..., 1]

        # Dung MEDIAN thay vi MEAN: ben hon voi cac vector flow loi cuc bo
        # (vai/ao it texture lam Farneback uoc luong sai o mot so pixel).
        motion_value = float(np.median(vertical_motion))

        self.signal.append(motion_value)
        self.prev_gray = gray

        return motion_value

    def reset(self):
        self.prev_gray = None
        self.signal.clear()
