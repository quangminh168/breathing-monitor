from scipy.signal import savgol_filter


class SignalFilter:

    @staticmethod
    def smooth(signal):

        if len(signal) < 21:
            return signal

        return savgol_filter(
            signal,
            window_length=21,
            polyorder=3
        )