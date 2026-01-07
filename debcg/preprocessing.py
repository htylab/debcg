from fractions import Fraction

import numpy as np
from scipy.signal import butter, decimate, resample_poly, sosfiltfilt


class ResampleResult:
    def __init__(self, data, sfreq):
        self.data = np.asarray(data, dtype=np.float64)
        self.sfreq = float(sfreq)


def _as_float64(x):
    return np.asarray(x, dtype=np.float64)


def downsample_to_sfreq(
    data,
    sfreq,
    target_sfreq,
    *,
    axis=-1,
    method="resample_poly",
):
    if sfreq <= 0:
        raise ValueError(f"sfreq must be positive, got {sfreq}")
    if target_sfreq <= 0:
        raise ValueError(f"target_sfreq must be positive, got {target_sfreq}")

    if float(sfreq) == float(target_sfreq):
        return ResampleResult(data=_as_float64(data), sfreq=float(target_sfreq))

    ratio = float(sfreq) / float(target_sfreq)
    q = int(round(ratio))
    is_integer_ratio = abs(ratio - q) < 1e-9

    x = _as_float64(data)
    if method == "decimate" and is_integer_ratio and q >= 2:
        # Match existing notebooks: IIR decimation with zero-phase compensation.
        y = decimate(x, q, axis=axis, zero_phase=True)
        return ResampleResult(data=_as_float64(y), sfreq=float(target_sfreq))

    frac = Fraction(float(target_sfreq) / float(sfreq)).limit_denominator(10_000)
    y = resample_poly(x, frac.numerator, frac.denominator, axis=axis)
    return ResampleResult(data=_as_float64(y), sfreq=float(target_sfreq))


def bandpass_zerophase(
    eeg,
    sfreq,
    *,
    band_hz=(0.5, 40.0),
    order=4,
    axis=-1,
):
    """
    Zero-phase band-pass filtering for EEG.

    Default band is 0.5–40 Hz, a common EEG range for resting-state/ERP work.
    """
    if sfreq <= 0:
        raise ValueError(f"sfreq must be positive, got {sfreq}")
    if order <= 0:
        raise ValueError(f"order must be positive, got {order}")

    low_hz, high_hz = float(band_hz[0]), float(band_hz[1])
    if not (0 < low_hz < high_hz < float(sfreq) / 2.0):
        raise ValueError(f"Invalid band_hz={band_hz} for sfreq={sfreq}")

    x = _as_float64(eeg)
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=float(sfreq), output="sos")
    y = sosfiltfilt(sos, x, axis=axis)
    return _as_float64(y)




def trim_to_minlenX(
    *arrays,
    axis=-1,
):
    present = [a for a in arrays if a is not None]
    if not present:
        return tuple(None for _ in arrays)

    lengths = [int(np.asarray(a).shape[axis]) for a in present]
    min_len = min(lengths)
    out = []
    for a in arrays:
        if a is None:
            out.append(None)
            continue
        x = _as_float64(np.asarray(a))
        slicer = [slice(None)] * x.ndim
        slicer[axis] = slice(0, min_len)
        out.append(_as_float64(x[tuple(slicer)]))
    return tuple(out)


def trim_to_minlen(*arrays, axis=-1):
    present = [a for a in arrays if a is not None]
    if not present:
        return tuple(None for _ in arrays)

    lengths = [int(np.asarray(a).shape[axis]) for a in present]
    min_len = min(lengths)

    out = []
    for a in arrays:
        if a is None:
            out.append(None)
            continue

        x = _as_float64(np.asarray(a))

        ax = axis if axis >= 0 else x.ndim + axis  # normalize

        slicer = [slice(None)] * x.ndim
        if min_len == 0:
            slicer[ax] = slice(0, 0)
        else:
            slicer[ax] = slice(-min_len, None)  # take from the end

        out.append(_as_float64(x[tuple(slicer)]))

    return tuple(out)
