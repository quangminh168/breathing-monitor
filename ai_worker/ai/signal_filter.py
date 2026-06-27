import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


class SignalFilter:
    """
    Loc tin hieu motion ve dung dai tan nhip tho nguoi thuc te:
    ~6-30 lan/phut tuong duong 0.1-0.5 Hz.

    Ban cu dung Savitzky-Golay voi window_length=21 SAMPLE CO DINH.
    O ~20-30 FPS, 21 sample chi la ~0.7-1 giay -- ngan hon ca 1/4 chu
    ky tho thuc (3-5 giay) -- nen gan nhu khong loc duoc nhieu tan so
    cao, va cung khong loai duoc troi nen tan so thap. Ban moi nay
    dung bandpass Butterworth, tu tinh sample rate thuc te (fps) tu
    chinh du lieu thay vi gia dinh co dinh.
    """

    LOW_HZ = 0.1   # ~6 lan/phut
    HIGH_HZ = 0.5  # ~30 lan/phut

    @staticmethod
    def smooth(signal, fps=None, duration_seconds=None):
        """
        signal: tin hieu motion thoi gian.
        fps: so sample/giay (uu tien dung truc tiep neu co).
        duration_seconds: neu khong co fps, se tu tinh
            fps = len(signal) / duration_seconds.

        Neu khong truyen fps/duration_seconds (de tuong thich nguoc),
        fallback ve Savitzky-Golay nhu ban cu (kem chinh xac hon).
        """
        signal = np.asarray(signal, dtype=float)

        if fps is None and duration_seconds:
            fps = len(signal) / duration_seconds

        if fps and fps > 0 and len(signal) > 30:
            filtered = SignalFilter._bandpass(signal, fps)
            if filtered is not None:
                return filtered

        # Fallback: khong biet fps hoac khong du sample cho bandpass on dinh
        if len(signal) < 21:
            return signal
        return savgol_filter(signal, window_length=21, polyorder=3)

    @staticmethod
    def _bandpass(signal, fps):
        nyquist = fps / 2.0
        low = SignalFilter.LOW_HZ / nyquist
        high = SignalFilter.HIGH_HZ / nyquist

        if high >= 1.0:
            high = 0.99
        if low <= 0:
            low = 1e-4
        if low >= high:
            return None  # fps qua thap so voi dai tan can loc

        try:
            b, a = butter(N=3, Wn=[low, high], btype="band")
            padlen = 3 * max(len(a), len(b))
            if len(signal) <= padlen:
                return None  # qua it sample de filtfilt chay on dinh
            return filtfilt(b, a, signal)
        except Exception:
            return None
