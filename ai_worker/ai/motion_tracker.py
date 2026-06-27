import cv2
import numpy as np


class MotionTracker:

    def __init__(self):

        self.prev_gray = None

        self.signal = []

    def process(self, roi):

        if roi is None:
            return None

        if roi.size == 0:
            return None

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            (200, 200)
        )

        # Lam mo nhe truoc khi tinh optical flow -> giam nhieu cam bien
        # tung pixel, flow uoc luong on dinh hon (it bi giat do hat sang).
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:

            self.prev_gray = gray

            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        vertical_motion = flow[..., 1]

        # Dung MEDIAN thay vi MEAN: ben hon voi cac vector flow loi cuc bo
        # (vai/ao it texture lam Farneback uoc luong sai o mot so pixel,
        # mean bi keo lech boi nhung outlier nay, median thi khong).
        motion_value = float(
            np.median(vertical_motion)
        )

        self.signal.append(
            motion_value
        )

        self.prev_gray = gray

        return motion_value

    def reset(self):

        self.prev_gray = None

        self.signal.clear()
