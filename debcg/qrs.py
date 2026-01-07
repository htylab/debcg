import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def qrs(ecg, sfreq):
    """
    Detect R-peak triggers (QRS peaks) from ECG.

    Parameters
    ----------
    ecg:
        1D ECG array.
    sfreq:
        Sampling frequency (Hz).

    Returns
    -------
    r_trigger:
        1D int64 array of sample indices of detected R-peaks.
    """
    fs = float(sfreq)
    if fs <= 0:
        raise ValueError(f"sfreq must be positive, got {sfreq}")

    ecg_1d = np.asarray(ecg, dtype=np.float64).reshape(-1)
    ecg_1d = ecg_1d - np.nanmean(ecg_1d)

    # Defaults (repo-wide): Pan–Tompkins style with a practical bandpass and
    # minimum RR constraint.
    bandpass_hz = (5.0, 15.0)
    min_distance_s = 0.6
    prominence = None

    low_hz, high_hz = bandpass_hz
    if not (0 < low_hz < high_hz < fs / 2):
        raise ValueError(f"Invalid bandpass_hz={bandpass_hz} for sfreq={sfreq}")

    b, a = butter(3, [low_hz, high_hz], btype="bandpass", fs=fs)
    ecg_f = filtfilt(b, a, ecg_1d)

    min_distance = max(1, int(round(min_distance_s * fs)))

    # Derivative (Pan–Tompkins 5-point operator), scale to approximate d/dt.
    der_k = np.array([1.0, 2.0, 0.0, -2.0, -1.0], dtype=np.float64) * (fs / 8.0)
    der = np.convolve(ecg_f, der_k, mode="same")

    sq = der * der

    # Moving-window integration (~150 ms).
    win = max(1, int(round(0.150 * fs)))
    mwi = np.convolve(sq, np.ones(win, dtype=np.float64) / float(win), mode="same")

    peaks_i, _props = find_peaks(mwi, distance=min_distance, prominence=prominence)
    peaks_i = peaks_i.astype(np.int64, copy=False)
    if peaks_i.size == 0:
        return np.array([], dtype=np.int64)

    # Refine to R-peak location on bandpassed ECG (local max of |ECG|).
    search = max(1, int(round(0.150 * fs)))
    refined = []
    for p in peaks_i.tolist():
        lo = max(0, int(p) - search)
        hi = min(ecg_f.shape[0], int(p) + search + 1)
        if lo >= hi:
            continue
        local = int(np.argmax(np.abs(ecg_f[lo:hi])) + lo)
        refined.append(local)

    if not refined:
        return np.array([], dtype=np.int64)

    refined_arr = np.unique(np.asarray(refined, dtype=np.int64))
    refined_arr = refined_arr[np.argsort(refined_arr)]

    # Enforce min_distance again (keep larger-amplitude peak if conflict).
    keep = []
    for p in refined_arr.tolist():
        if not keep:
            keep.append(int(p))
            continue
        if int(p) - int(keep[-1]) >= min_distance:
            keep.append(int(p))
            continue
        if abs(float(ecg_f[int(p)])) > abs(float(ecg_f[int(keep[-1])] )):
            keep[-1] = int(p)

    return np.asarray(keep, dtype=np.int64)

