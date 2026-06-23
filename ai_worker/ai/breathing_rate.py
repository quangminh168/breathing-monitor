from scipy.signal import find_peaks


class BreathingRateEstimator:

    @staticmethod
    def estimate(signal, duration_seconds):

        peaks, _ = find_peaks(
            signal,
            distance=50
        )

        breaths = len(peaks)

        bpm = (
            breaths / duration_seconds
        ) * 60

        return bpm, peaks