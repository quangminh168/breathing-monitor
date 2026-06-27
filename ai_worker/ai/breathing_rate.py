import numpy as np
from scipy.signal import find_peaks


class BreathingRateEstimator:
    """
    Dem peak tren tin hieu da loc de tinh BPM.

    Ban cu dung distance=50 CO DINH THEO SAMPLE (khong theo thoi gian),
    va KHONG co nguong bien do (prominence/height) -- nen bat ky gon
    nhieu nho nao cung duoc tinh ngang hang voi 1 nhip tho thuc, gay
    "nhay so" giua cac lan do.

    Ban moi: distance tinh theo THOI GIAN THUC (tu fps = len(signal)/
    duration_seconds), va them prominence -- chi tinh la peak khi noi
    bat ro rang so voi muc dao dong chung cua tin hieu.
    """

    # Nguoi thuc te tho khoang 5-40 lan/phut -> 1 nhip cach nhau toi
    # thieu 60/40 = 1.5 giay. Dung de chan dem trung nhieu peak trong
    # cung 1 chu ky tho.
    MAX_PLAUSIBLE_BPM = 40
    MIN_SECONDS_BETWEEN_BREATHS = 60.0 / MAX_PLAUSIBLE_BPM  # = 1.5s

    # He so nguong noi bat cua peak so voi do lech chuan tin hieu.
    # Tang len (vd 0.4-0.5) neu van con nhay so do nhieu; giam xuong
    # (vd 0.15-0.2) neu BPM bi dem THIEU nhip thuc.
    PROMINENCE_FACTOR = 0.3

    @staticmethod
    def estimate(signal, duration_seconds):
        signal = np.asarray(signal, dtype=float)

        if len(signal) == 0 or duration_seconds <= 0:
            return 0.0, []

        fps = len(signal) / duration_seconds
        min_distance_samples = max(
            1,
            int(fps * BreathingRateEstimator.MIN_SECONDS_BETWEEN_BREATHS)
        )

        signal_std = float(np.std(signal))
        prominence = (
            BreathingRateEstimator.PROMINENCE_FACTOR * signal_std
            if signal_std > 0 else None
        )

        peaks, _ = find_peaks(
            signal,
            distance=min_distance_samples,
            prominence=prominence,
        )

        breaths = len(peaks)

        bpm = (
            breaths / duration_seconds
        ) * 60

        return bpm, peaks
