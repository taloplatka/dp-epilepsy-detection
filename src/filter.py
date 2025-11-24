from scipy.signal import butter, sosfiltfilt
from consts import FS


# See scipy.siganl docs
# https://docs.scipy.org/doc//scipy-1.16.2/reference/generated/scipy.signal.sosfiltfilt.html
# https://docs.scipy.org/doc//scipy-1.16.2/reference/generated/scipy.signal.butter.html#scipy.signal.butter
def bandpass_filter_bonn_eeg(x, low, high, fs=FS):
    sos = butter(N=4, Wn=[low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)
