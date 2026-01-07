from types import ModuleType

import numpy as np
from scipy.signal import detrend


class DMHConfig:
    """
    DMH implementation aligned to `eeg_bcg_dmh_reg.m` (MATLAB).
    Origin: https://github.com/fahsuanlin/fhlin_toolbox/blob/master/codes/eeg_bcg_dmh_reg.m

    Notes
    -----
    This re-implementation follows the MATLAB structure:
    - Detect QRS peaks from ECG.
    - Segment the recording into cardiac cycles.
    - Find nearest-neighbor cardiac cycles using ECG dynamics (kNN in ECG-cycle space).
    - For each cycle and EEG channel, fit a linear model using neighbor-cycle EEG
      segments (detrended) + intercept + linear ramp, then subtract the modeled signal.
    """

    FIELD_NAMES = (
        "nn",
        "flag_auto_hp",
        "flag_reg",
        "outlier_threshold_factor",
    )

    def __init__(
        self,
        nn=10,
        flag_auto_hp=False,
        flag_reg=False,
        outlier_threshold_factor=4.0,
    ):
        self.nn = nn
        self.flag_auto_hp = flag_auto_hp
        self.flag_reg = flag_reg
        self.outlier_threshold_factor = outlier_threshold_factor


def _robust_outlier_mask_mad(x, *, threshold_factor):
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad == 0 or not np.isfinite(mad) or threshold_factor <= 0:
        return np.zeros(v.shape[0], dtype=bool)
    z = np.abs(v - med) / (1.4826 * mad)
    return (z > float(threshold_factor)).astype(bool, copy=False)


def _fill_next(arr):
    x = np.asarray(arr, dtype=np.float64).reshape(-1).copy()
    next_val = np.nan
    for i in range(x.size - 1, -1, -1):
        if np.isfinite(x[i]):
            next_val = x[i]
        else:
            x[i] = next_val
    return x


def _cycle_onsets_offsets_from_qrs(qrs_peaks, n_samples):
    qrs = np.asarray(qrs_peaks, dtype=np.int64).reshape(-1)
    qrs = qrs[(qrs >= 0) & (qrs < n_samples)]
    if qrs.size == 0:
        return np.array([0], dtype=np.int64), np.array([n_samples - 1], dtype=np.int64)

    qrs = np.unique(qrs)
    ecg_idx = np.full(n_samples, np.nan, dtype=np.float64)
    ecg_idx[qrs] = np.arange(1, qrs.size + 1, dtype=np.float64)

    # MATLAB: fillmissing(ecg_idx,'next'); then trailing NaN -> max+1
    ecg_idx = _fill_next(ecg_idx)
    if np.any(~np.isfinite(ecg_idx)):
        mx = float(np.nanmax(ecg_idx[np.isfinite(ecg_idx)])) if np.any(np.isfinite(ecg_idx)) else 0.0
        ecg_idx[~np.isfinite(ecg_idx)] = mx + 1.0

    idx = np.where(np.diff(ecg_idx) != 0)[0].astype(np.int64, copy=False)
    onsets = np.concatenate([np.array([0], dtype=np.int64), idx + 1])
    offsets = np.concatenate([idx, np.array([n_samples - 1], dtype=np.int64)])
    return onsets, offsets


def _knn_cycles_by_ecg(
    ecg,
    *,
    onsets,
    offsets,
    nn,
):
    ecg = np.asarray(ecg, dtype=np.float64).reshape(-1)
    on = np.asarray(onsets, dtype=np.int64).reshape(-1)
    off = np.asarray(offsets, dtype=np.int64).reshape(-1)
    n_cycles = int(on.shape[0])
    if n_cycles != int(off.shape[0]):
        raise ValueError("onsets/offsets length mismatch")
    if nn <= 0:
        return np.full((n_cycles, 0), -1, dtype=np.int64), np.full((n_cycles, 0), np.nan, dtype=np.float64)

    ll = (off - on + 1).astype(np.int64, copy=False)
    lmax = int(np.max(ll))
    peak_idx = np.arange(lmax, dtype=np.int64)

    ecg_ccm_idx = (on[:, None] + peak_idx[None, :]).astype(np.float64)
    ecg_ccm_idx[ecg_ccm_idx >= ecg.size] = np.nan
    ecg_ccm_idx[ecg_ccm_idx < 0] = np.nan

    IDX = np.full((n_cycles, int(nn)), -1, dtype=np.int64)
    D = np.full((n_cycles, int(nn)), np.nan, dtype=np.float64)

    for ii in range(n_cycles):
        l_now = int(ll[ii])
        ecg_ccm_idx_now = ecg_ccm_idx[:, :l_now]
        if np.isnan(ecg_ccm_idx_now[ii]).any():
            continue

        tmp = np.full(ecg_ccm_idx_now.shape, np.nan, dtype=np.float64)
        mask = ~np.isnan(ecg_ccm_idx_now)
        tmp[mask] = ecg[ecg_ccm_idx_now[mask].astype(np.int64, copy=False)]
        if tmp.shape[1] > 2:
            ok_rows = ~np.isnan(tmp).any(axis=1)
            if np.any(ok_rows):
                tmp[ok_rows] = detrend(tmp[ok_rows], axis=1, type="linear")

        valid = ~np.isnan(tmp).any(axis=1)
        if not valid[ii]:
            continue
        idx_valid = np.where(valid)[0]
        tmp_valid = tmp[idx_valid]

        row = tmp[ii]
        dist = np.linalg.norm(tmp_valid - row[None, :], axis=1)
        order = np.argsort(dist)
        # include self (dist=0), then drop it.
        k = min(int(nn) + 1, order.size)
        picks = idx_valid[order[:k]]
        picks = picks[picks != ii]
        picks = picks[: int(nn)]
        IDX[ii, : picks.size] = picks.astype(np.int64, copy=False)
        D[ii, : picks.size] = dist[order[:k]][idx_valid[order[:k]] != ii][: picks.size].astype(np.float64, copy=False)

    return IDX, D


def run(
    eeg,
    ecg,
    config=None,
    *,
    sfreq=500.0,
    r_trigger=None,
):
    from debcg.qrs import qrs

    if config is None:
        cfg = DMHConfig()
    elif isinstance(config, DMHConfig):
        cfg = config
    elif isinstance(config, dict):
        known = set(DMHConfig.FIELD_NAMES)
        cfg_kwargs = {k: v for k, v in config.items() if k in known}
        cfg = DMHConfig(**cfg_kwargs)
    else:
        raise TypeError(f"config must be DMHConfig | dict | None, got {type(config)!r}")

    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2:
        raise ValueError(f"eeg must have shape (n_channels,n_samples), got {eeg.shape}")
    n_ch, n_samples = eeg.shape

    ecg = np.asarray(ecg, dtype=np.float64).reshape(-1)
    if ecg.shape[0] != n_samples:
        raise ValueError("ecg length must match eeg samples")

    ecg_use = ecg.copy()
    out_mask = _robust_outlier_mask_mad(ecg_use, threshold_factor=float(cfg.outlier_threshold_factor))
    if np.any(out_mask):
        ecg_use[out_mask] = float(np.median(ecg_use))

    if r_trigger is None:
        r_trigger = qrs(ecg_use, float(sfreq))
    else:
        r_trigger = np.asarray(r_trigger, dtype=np.int64).reshape(-1)
    onsets, offsets = _cycle_onsets_offsets_from_qrs(r_trigger, n_samples)
    ll = (offsets - onsets + 1).astype(np.int64, copy=False)

    IDX, _D = _knn_cycles_by_ecg(ecg_use, onsets=onsets, offsets=offsets, nn=int(cfg.nn))

    pred = np.zeros_like(eeg, dtype=np.float64)

    for seg_idx in range(onsets.size):
        onset = int(onsets[seg_idx])
        l_now = int(ll[seg_idx])
        if l_now <= 1:
            continue
        target_slice = slice(onset, onset + l_now)

        neigh = IDX[seg_idx]
        neigh = neigh[neigh >= 0]
        if neigh.size == 0:
            continue

        # Precompute neighbor onsets for this segment.
        neigh_onsets = onsets[neigh].astype(np.int64, copy=False)

        for ch in range(n_ch):
            y = eeg[ch, target_slice].astype(np.float64, copy=False)

            segs = []
            for no in neigh_onsets:
                s0 = int(no)
                s1 = s0 + l_now
                if s1 > n_samples:
                    segs = []
                    break
                segs.append(eeg[ch, s0:s1])
            if not segs:
                continue

            Xn = np.stack(segs, axis=0).astype(np.float64, copy=False)
            if Xn.shape[1] > 2:
                Xn = detrend(Xn, axis=1, type="linear")

            X = Xn.T  # (time, nn)
            ramp = (np.arange(l_now, dtype=np.float64) / float(l_now)).reshape(-1, 1)
            X = np.concatenate([X, np.ones((l_now, 1), dtype=np.float64), ramp], axis=1)

            beta, *_rest = np.linalg.lstsq(X, y.reshape(-1, 1), rcond=None)
            pred_seg = (X @ beta).reshape(-1)
            pred[ch, target_slice] = pred_seg

    pred[~np.isfinite(pred)] = 0.0

    if cfg.flag_reg:
        # MATLAB supports a second-stage regression; default is subtraction.
        raise NotImplementedError("DMH flag_reg=True is not implemented; use default subtraction (flag_reg=False).")

    return (eeg - pred).astype(np.float64, copy=False)


def dmh(
    eeg,
    ecg,
    config=None,
    *,
    sfreq=500.0,
    r_trigger=None,
):
    return run(eeg, ecg, config, sfreq=sfreq, r_trigger=r_trigger)


class _CallableModule(ModuleType):
    def __call__(self, eeg, ecg, config=None, *, sfreq=500.0, r_trigger=None, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        return run(eeg, ecg, config, sfreq=sfreq, r_trigger=r_trigger)


import sys as _sys

_sys.modules[__name__].__class__ = _CallableModule
