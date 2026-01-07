from types import ModuleType

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from debcg.qrs import qrs


'''
Python re-implementation of OBS and PanTompkins method
OBS algorithm:
Origin: https://github.com/fahsuanlin/fhlin_toolbox/blob/master/codes/eeg_bcg.m

Origin of PanTompkins method:
[1] PAN.J, TOMPKINS. W.J,"A Real-Time QRS Detection Algorithm" IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. BME-32, NO. 3, MARCH 1985.
[2] Sedghamiz. H, "Matlab Implementation of Pan Tompkins ECG QRS detector.",2014.
'''

class QRSDetectionResult:
    def __init__(self, peaks, ecg_processed):
        self.peaks = np.asarray(peaks, dtype=np.int64)
        self.ecg_processed = np.asarray(ecg_processed, dtype=np.float64)


def detect_qrs_peaks(
    ecg,
    fs,
    *,
    method="pan_tompkins",
    bandpass_hz=(5.0, 15.0),
    min_distance_s=0.6,
    prominence=None,
):
    """
    QRS detection from ECG for OBS timing.

    This is a practical detector for EEG-fMRI ECG channels.

    Supported methods
    -----------------
    - ``scipy_find_peaks_abs`` (default): 5–15 Hz zero-phase bandpass then
      `find_peaks` on the absolute filtered signal.
    - ``pan_tompkins``: Pan–Tompkins-style pipeline (bandpass -> derivative ->
      squaring -> moving-window integration -> peak picking on the integrated
      signal), then refine to the nearest local max in the bandpassed ECG.


    """
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    ecg_1d = np.asarray(ecg, dtype=np.float64).reshape(-1)
    ecg_1d = ecg_1d - np.nanmean(ecg_1d)

    low_hz, high_hz = bandpass_hz
    if not (0 < low_hz < high_hz < fs / 2):
        raise ValueError(f"Invalid bandpass_hz={bandpass_hz} for fs={fs}")

    b, a = butter(3, [low_hz, high_hz], btype="bandpass", fs=fs)
    ecg_f = filtfilt(b, a, ecg_1d)

    min_distance = max(1, int(round(min_distance_s * fs)))

    if method == "scipy_find_peaks_abs":
        x = np.abs(ecg_f)
        peaks, _props = find_peaks(x, distance=min_distance, prominence=prominence)
        peaks = peaks.astype(np.int64, copy=False)
        return QRSDetectionResult(peaks=peaks, ecg_processed=ecg_f)

    if method == "pan_tompkins":
        # Derivative (Pan–Tompkins 5-point operator), scale to approximate d/dt.
        der_k = np.array([1.0, 2.0, 0.0, -2.0, -1.0], dtype=np.float64) * (float(fs) / 8.0)
        der = np.convolve(ecg_f, der_k, mode="same")

        sq = der * der

        # Moving-window integration (~150 ms).
        win = max(1, int(round(0.150 * float(fs))))
        mwi = np.convolve(sq, np.ones(win, dtype=np.float64) / float(win), mode="same")

        peaks_i, _props = find_peaks(mwi, distance=min_distance, prominence=prominence)
        peaks_i = peaks_i.astype(np.int64, copy=False)
        if peaks_i.size == 0:
            return QRSDetectionResult(peaks=np.array([], dtype=np.int64), ecg_processed=ecg_f)

        # With a sufficiently strict `min_distance_s`, the MWI peaks are usually
        # already dominated by QRS complexes for these datasets, so we take all
        # candidates and refine to R-peaks on the bandpassed ECG.
        qrs_i = peaks_i.tolist()

        # Refine to R-peak location on bandpassed ECG (local max of |ECG|).
        search = max(1, int(round(0.150 * float(fs))))
        refined = []
        for p in qrs_i:
            lo = max(0, int(p) - search)
            hi = min(ecg_f.shape[0], int(p) + search + 1)
            if lo >= hi:
                continue
            local = int(np.argmax(np.abs(ecg_f[lo:hi])) + lo)
            refined.append(local)

        if not refined:
            return QRSDetectionResult(peaks=np.array([], dtype=np.int64), ecg_processed=ecg_f)

        refined_arr = np.unique(np.asarray(refined, dtype=np.int64))
        refined_arr = refined_arr[np.argsort(refined_arr)]

        # Enforce min_distance again (keep the larger-amplitude peak if conflict).
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

        return QRSDetectionResult(peaks=np.asarray(keep, dtype=np.int64), ecg_processed=ecg_f)

    raise ValueError(f"Unknown method: {method!r}")


def rr10_window(
    qrs_peaks,
    fs,
    *,
    q=0.1,
    pre_frac=0.2,
    post_frac=0.8,
):
    """
    Match the window choice used in `OBS/read_smsini_eeg.m` before calling `eeg_bcg.m`.
    """
    qrs = np.asarray(qrs_peaks, dtype=np.int64).reshape(-1)
    if qrs.size < 3:
        return 0.5, 0.5

    rr_s = np.diff(qrs).astype(np.float64) / float(fs)
    rr_s = rr_s[np.isfinite(rr_s) & (rr_s > 0)]
    if rr_s.size == 0:
        return 0.5, 0.5

    rr_q = float(np.quantile(rr_s, q))
    return float(pre_frac * rr_q), float(post_frac * rr_q)


def _trial_selection_indices(trial_idx, n_trials, n_ma_bcg):
    if n_trials <= 0:
        return np.array([], dtype=np.int64)

    if n_ma_bcg <= 0 or n_trials <= n_ma_bcg:
        return np.arange(n_trials, dtype=np.int64)

    half = int(round((n_ma_bcg - 1) / 2))
    if trial_idx <= half:
        start = 0
    elif trial_idx >= n_trials - half - 1:
        start = n_trials - n_ma_bcg
    else:
        start = trial_idx - half

    return np.arange(start, start + n_ma_bcg, dtype=np.int64)


class OBSConfig:
    """
    Default OBS settings for the unified (read_smsini-style) pipeline.

    Notes
    -----
    We detect QRS peaks from the provided ECG (no TRIGGER_ECG_*), then derive
    the OBS window from RR10 (10th percentile RR) like `read_smsini_eeg.m`.
    """

    FIELD_NAMES = (
        "bcg_nsvd",
        "n_ma_bcg",
        "rr_quantile",
        "rr_pre_frac",
        "rr_post_frac",
    )

    def __init__(
        self,
        bcg_nsvd=3,
        n_ma_bcg=21,
        rr_quantile=0.1,
        rr_pre_frac=0.2,
        rr_post_frac=0.8,
    ):
        self.bcg_nsvd = bcg_nsvd
        self.n_ma_bcg = n_ma_bcg
        self.rr_quantile = rr_quantile
        self.rr_pre_frac = rr_pre_frac
        self.rr_post_frac = rr_post_frac


class OBSResult:
    def __init__(self, eeg_bcg, qrs_peaks, bcg_tpre_s, bcg_tpost_s):
        self.eeg_bcg = np.asarray(eeg_bcg, dtype=np.float64)
        self.qrs_peaks = np.asarray(qrs_peaks, dtype=np.int64)
        self.bcg_tpre_s = float(bcg_tpre_s)
        self.bcg_tpost_s = float(bcg_tpost_s)


def obs_correction(
    eeg,
    ecg,
    fs,
    *,
    r_trigger=None,
    bcg_tpre_s=0.5,
    bcg_tpost_s=0.5,
    bcg_nsvd=3,
    n_ma_bcg=21,
    dynamic=True,
    anchor_ends=False,
    bad_rejection=False,
    post_ssp=False,
):
    """
    OBS/PC correction for BCG artifacts (channel-wise), MATLAB `eeg_bcg.m`-style.

    This is a lightweight port aligned with the parameters used in
    `OBS/read_smsini_eeg.m` for dynamic OBS (RR10 window, nsvd=3, n_ma=21).
    """
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2:
        raise ValueError(f"eeg must have shape (n_channels, n_samples); got {eeg.shape}")
    n_channels, n_samples = eeg.shape

    ecg = np.asarray(ecg, dtype=np.float64).reshape(-1)
    if ecg.shape[0] != n_samples:
        raise ValueError(f"ecg length {ecg.shape[0]} must match eeg samples {n_samples}")

    if r_trigger is None:
        r_trigger = qrs(ecg, float(fs))
    else:
        r_trigger = np.asarray(r_trigger, dtype=np.int64).reshape(-1)

    tpre = int(round(bcg_tpre_s * fs))
    tpost = int(round(bcg_tpost_s * fs))
    if tpre < 0 or tpost < 0:
        raise ValueError("bcg_tpre_s and bcg_tpost_s must be non-negative")
    win_len = tpre + tpost + 1
    if win_len <= 1:
        raise ValueError("BCG window length is too small")

    starts = r_trigger - tpre
    stops = r_trigger + tpost + 1
    valid = (starts >= 0) & (stops <= n_samples)
    r_trigger = r_trigger[valid]
    starts = starts[valid]
    stops = stops[valid]
    n_trials = int(r_trigger.shape[0])
    if n_trials == 0:
        return OBSResult(eeg_bcg=eeg.copy(), qrs_peaks=r_trigger, bcg_tpre_s=bcg_tpre_s, bcg_tpost_s=bcg_tpost_s)

    eeg_bcg = eeg.copy()

    for ch in range(n_channels):
        epochs = np.stack([eeg[ch, s:e] for s, e in zip(starts, stops)], axis=0)
        if epochs.shape != (n_trials, win_len):
            raise RuntimeError("Unexpected epoch extraction shape")

        # In unified comparisons, keep data dense (no NaN holes).
        # `bad_rejection` is left as an option but defaults to False.
        bad = np.zeros(n_trials, dtype=bool)
        if bad_rejection:
            max_abs = np.nanmax(np.abs(epochs), axis=1)
            bad = max_abs > 200.0
            if bad.mean() > 0.02:
                n_extreme = max(1, int(round(n_trials / 50)))
                idx_sorted = np.argsort(max_abs)
                bad = np.zeros(n_trials, dtype=bool)
                bad[idx_sorted[-n_extreme:]] = True

        if not dynamic:
            trial_power = np.nanmax(np.abs(epochs), axis=1)
            idx_sorted = np.argsort(trial_power)
            idx1 = int(round(n_trials * 0.2))
            idx2 = int(round(n_trials * 0.8))
            idx2 = min(idx2, idx1 + 100)
            trial_sel = idx_sorted[idx1:idx2]

            X = epochs[trial_sel].copy()
            row_energy = np.nansum(np.abs(X), axis=1)
            X = X[row_energy > np.finfo(np.float64).eps]
            if X.shape[0] < 2:
                continue
            _u, _s, vt = np.linalg.svd(X, full_matrices=False)
            n_bases = max(1, min(int(bcg_nsvd), int(vt.shape[0])))
            bases_static = vt[:n_bases].T

        for trial_idx in range(n_trials):
            if bad[trial_idx]:
                continue

            y = epochs[trial_idx].copy()
            if dynamic:
                trial_sel = _trial_selection_indices(trial_idx, n_trials, int(n_ma_bcg))
                X = epochs[trial_sel].copy()
                row_energy = np.nansum(np.abs(X), axis=1)
                X = X[row_energy > np.finfo(np.float64).eps]
                if X.shape[0] < 2:
                    continue
                _u, _s, vt = np.linalg.svd(X, full_matrices=False)
                n_bases = max(1, min(int(bcg_nsvd), int(vt.shape[0])))
                bases = vt[:n_bases].T
            else:
                bases = bases_static

            beta = bases.T @ y
            y_corr = y - bases @ beta

            if anchor_ends:
                bnd_delta = np.array([y_corr[0] - y[0], y_corr[-1] - y[-1]], dtype=np.float64)
                ramp = (np.arange(1, win_len + 1, dtype=np.float64) / float(win_len)).reshape(-1, 1)
                bnd_bases = np.concatenate([np.ones((win_len, 1), dtype=np.float64), ramp], axis=1)
                end_bases = bnd_bases[[0, -1], :]
                coef = np.linalg.solve(end_bases.T @ end_bases, end_bases.T @ bnd_delta)
                y_corr = y_corr - bnd_bases @ coef

            eeg_bcg[ch, starts[trial_idx] : stops[trial_idx]] = y_corr

    if post_ssp:
        # Not enabled in the MATLAB call used in `read_smsini_eeg.m`.
        pass

    return OBSResult(eeg_bcg=eeg_bcg, qrs_peaks=r_trigger, bcg_tpre_s=bcg_tpre_s, bcg_tpost_s=bcg_tpost_s)


def run_obs_read_smsini_style(
    eeg,
    ecg,
    sfreq,
    *,
    bcg_nsvd=3,
    n_ma_bcg=21,
    r_trigger=None,
):
    """
    OBS settings matching the call site in `OBS/read_smsini_eeg.m` (but without using TRIGGER_ECG_*).
    """
    if r_trigger is None:
        r_trigger = qrs(ecg, float(sfreq))
    else:
        r_trigger = np.asarray(r_trigger, dtype=np.int64).reshape(-1)
    tpre_s, tpost_s = rr10_window(r_trigger, sfreq)
    return obs_correction(
        eeg,
        ecg,
        sfreq,
        r_trigger=r_trigger,
        bcg_tpre_s=tpre_s,
        bcg_tpost_s=tpost_s,
        bcg_nsvd=int(bcg_nsvd),
        n_ma_bcg=int(n_ma_bcg),
        dynamic=True,
        anchor_ends=False,
        bad_rejection=False,
        post_ssp=False,
    )


def run(
    eeg,
    ecg,
    config=None,
    *,
    sfreq=500.0,
    r_trigger=None,
    **kwargs,
):
    """
    OBS convenience wrapper returning cleaned EEG.

    Call signature matches the unified comparison pipeline:
    `debcg.obs(EEG, ECG, obs_config) -> filtered_EEG`.
    """
    extra_kwargs = {}
    if config is None:
        cfg = OBSConfig()
    elif isinstance(config, OBSConfig):
        cfg = config
    elif isinstance(config, dict):
        known = set(OBSConfig.FIELD_NAMES)
        cfg_kwargs = {k: v for k, v in config.items() if k in known}
        extra_kwargs = {k: v for k, v in config.items() if k not in known}
        cfg = OBSConfig(**cfg_kwargs)
    else:
        raise TypeError(f"config must be OBSConfig | dict | None, got {type(config)!r}")

    if r_trigger is None:
        r_trigger = qrs(ecg, float(sfreq))
    else:
        r_trigger = np.asarray(r_trigger, dtype=np.int64).reshape(-1)

    tpre_s, tpost_s = rr10_window(
        r_trigger,
        sfreq,
        q=float(cfg.rr_quantile),
        pre_frac=float(cfg.rr_pre_frac),
        post_frac=float(cfg.rr_post_frac),
    )

    call_kwargs = {
        "r_trigger": r_trigger,
        "bcg_tpre_s": tpre_s,
        "bcg_tpost_s": tpost_s,
        "bcg_nsvd": int(cfg.bcg_nsvd),
        "n_ma_bcg": int(cfg.n_ma_bcg),
        "dynamic": True,
        "anchor_ends": False,
        "bad_rejection": False,
        "post_ssp": False,
    }
    call_kwargs.update(extra_kwargs)
    call_kwargs.update(kwargs)
    return obs_correction(eeg, ecg, sfreq, **call_kwargs).eeg_bcg


def obs(
    eeg,
    ecg,
    config=None,
    *,
    sfreq=500.0,
    **kwargs,
):
    return run(eeg, ecg, config, sfreq=sfreq, **kwargs)


class _CallableModule(ModuleType):
    def __call__(self, eeg, ecg, config=None, *, sfreq=500.0, **kwargs):
        return run(eeg, ecg, config, sfreq=sfreq, **kwargs)


# Allow `import debcg; debcg.obs(eeg, ecg, ...)` while keeping the module namespace.
import sys as _sys

_sys.modules[__name__].__class__ = _CallableModule
