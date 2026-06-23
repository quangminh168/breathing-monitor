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

        motion_value = float(
            np.mean(vertical_motion)
        )

        self.signal.append(
            motion_value
        )

        self.prev_gray = gray

        return motion_value

    def reset(self):

        self.prev_gray = None

        self.signal.clear()