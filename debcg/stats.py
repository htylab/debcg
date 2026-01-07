import numpy as np
from scipy.signal import coherence, welch


class WelchPSD:
    def __init__(self, freqs, psd):
        self.freqs = np.asarray(freqs, dtype=np.float64)
        self.psd = np.asarray(psd, dtype=np.float64)


class AlphaHBStats:
    """
    Alpha/HB score summary based on the original s3_psd notebook definition.

    Score (per-channel):
        (sum PSD_close[8-12Hz] - sum PSD_open[8-12Hz]) / mean(sum PSD_{open,close}[hb_band])

    hb_band is estimated from ECG by taking the PSD peak frequency in open/close,
    averaging the peak frequencies, and using +/-0.5 Hz around the average.
    """

    def __init__(self, scores, avg, hb_band, alpha_diff, hb_avg):
        self.scores = np.asarray(scores, dtype=np.float64)
        self.avg = float(avg)
        self.hb_band = tuple(hb_band)
        self.alpha_diff = np.asarray(alpha_diff, dtype=np.float64)
        self.hb_avg = np.asarray(hb_avg, dtype=np.float64)


def welch_psd(
    signal,
    *,
    sfreq,
    nperseg=1024,
    axis=-1,
):
    """
    Vectorized Welch PSD.

    Parameters
    ----------
    signal:
        Array-like, any shape. Welch runs along `axis`.
    sfreq:
        Sampling frequency (Hz).
    nperseg:
        Segment length for Welch.
    axis:
        Axis corresponding to time.
    """
    if sfreq <= 0:
        raise ValueError(f"sfreq must be positive, got {sfreq}")
    if nperseg <= 0:
        raise ValueError(f"nperseg must be positive, got {nperseg}")

    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        raise ValueError("signal is empty")

    n_time = int(x.shape[axis])
    if n_time <= 0:
        raise ValueError("signal has no time samples")
    nperseg_eff = min(int(nperseg), n_time)

    freqs, pxx = welch(x, fs=float(sfreq), nperseg=nperseg_eff, axis=axis)
    return WelchPSD(freqs=np.asarray(freqs, dtype=np.float64), psd=np.asarray(pxx, dtype=np.float64))


def get_psd_welch(
    signal,
    *,
    sample_rate=500.0,
    nperseg=1024,
):
    """
    Backwards-compatible helper matching the original notebook signature.

    Returns
    -------
    freqs:
        (n_channels, n_freq) array (each row identical)
    power:
        (n_channels, n_freq) PSD array
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"signal must be 1D or 2D (n_channels,n_samples), got {x.shape}")

    out = welch_psd(x, sfreq=float(sample_rate), nperseg=int(nperseg), axis=-1)
    freqs = np.tile(out.freqs[None, :], (x.shape[0], 1)).astype(np.float64, copy=False)
    power = out.psd.astype(np.float64, copy=False)
    return freqs, power


def get_hb_band(
    ecg_open,
    ecg_close,
    *,
    sfreq,
    nperseg=1024,
    half_width_hz=0.5,
):
    """
    Estimate the heart-beat frequency band from ECG using a PSD peak.

    This follows the original notebook behavior (including allowing a 0-Hz peak).
    """
    if half_width_hz <= 0:
        raise ValueError(f"half_width_hz must be positive, got {half_width_hz}")

    ecg_o = np.asarray(ecg_open, dtype=np.float64).reshape(-1)
    ecg_c = np.asarray(ecg_close, dtype=np.float64).reshape(-1)

    psd_o = welch_psd(ecg_o, sfreq=float(sfreq), nperseg=int(nperseg), axis=-1)
    psd_c = welch_psd(ecg_c, sfreq=float(sfreq), nperseg=int(nperseg), axis=-1)

    f_o = float(psd_o.freqs[int(np.argmax(psd_o.psd))])
    f_c = float(psd_c.freqs[int(np.argmax(psd_c.psd))])
    f0 = (f_o + f_c) / 2.0

    if f0 != 0.0:
        return (f0 - float(half_width_hz), f0 + float(half_width_hz))
    return (0.0, float(half_width_hz))


def alpha_hb_stats(
    eeg_open,
    eeg_close,
    ecg_open,
    ecg_close,
    *,
    sfreq,
    nperseg=1024,
    alpha_band=(8.0, 12.0),
    hb_half_width_hz=0.5,
):
    """
    Compute the alpha/HB score (per channel) and summary statistics.
    """
    eeg_o = np.asarray(eeg_open, dtype=np.float64)
    eeg_c = np.asarray(eeg_close, dtype=np.float64)
    if eeg_o.ndim != 2 or eeg_c.ndim != 2:
        raise ValueError("eeg_open/eeg_close must have shape (n_channels,n_samples)")
    if eeg_o.shape[0] != eeg_c.shape[0]:
        raise ValueError("eeg_open/eeg_close channel count mismatch")

    hb_band = get_hb_band(
        ecg_open,
        ecg_close,
        sfreq=float(sfreq),
        nperseg=int(nperseg),
        half_width_hz=float(hb_half_width_hz),
    )

    psd_o = welch_psd(eeg_o, sfreq=float(sfreq), nperseg=int(nperseg), axis=-1)
    psd_c = welch_psd(eeg_c, sfreq=float(sfreq), nperseg=int(nperseg), axis=-1)
    if psd_o.freqs.shape != psd_c.freqs.shape or not np.allclose(psd_o.freqs, psd_c.freqs):
        raise RuntimeError("Welch frequency grids differ between open/close")

    freqs = psd_o.freqs
    a0, a1 = float(alpha_band[0]), float(alpha_band[1])
    alpha_mask = (a0 < freqs) & (freqs < a1)
    hb_mask = (float(hb_band[0]) < freqs) & (freqs < float(hb_band[1]))

    alpha_open = np.sum(psd_o.psd[:, alpha_mask], axis=1)
    alpha_close = np.sum(psd_c.psd[:, alpha_mask], axis=1)
    alpha_diff = alpha_close - alpha_open

    hb_open = np.sum(psd_o.psd[:, hb_mask], axis=1)
    hb_close = np.sum(psd_c.psd[:, hb_mask], axis=1)
    hb_avg = (hb_open + hb_close) / 2.0

    with np.errstate(divide="ignore", invalid="ignore"):
        scores = alpha_diff / hb_avg
    scores = scores.astype(np.float64, copy=False)
    scores = np.where(hb_avg == 0, np.nan, scores).astype(np.float64, copy=False)

    avg = float(np.nanmean(scores)) if np.any(np.isfinite(scores)) else float("nan")
    return AlphaHBStats(
        scores=scores,
        avg=avg,
        hb_band=hb_band,
        alpha_diff=alpha_diff.astype(np.float64, copy=False),
        hb_avg=hb_avg.astype(np.float64, copy=False),
    )


def _nanmean_or_nan(x):
    return float(np.nanmean(x)) if np.any(np.isfinite(x)) else float("nan")


def _multi_band_mask(freqs, bands):
    f = np.asarray(freqs, dtype=np.float64).reshape(-1)
    m = np.zeros(f.shape[0], dtype=bool)
    for lo, hi in bands:
        m |= (float(lo) < f) & (f < float(hi))
    return m


def _rr_intervals_s(
    qrs_peaks,
    *,
    sfreq,
    min_rr_s=0.3,
    max_rr_s=2.0,
):
    qrs = np.asarray(qrs_peaks, dtype=np.int64).reshape(-1)
    if qrs.size < 2:
        return np.array([], dtype=np.float64)
    rr = np.diff(qrs).astype(np.float64) / float(sfreq)
    rr = rr[np.isfinite(rr) & (rr > float(min_rr_s)) & (rr < float(max_rr_s))]
    return rr.astype(np.float64, copy=False)


def _estimate_hr_hz_from_psd(
    ecg_open,
    ecg_close,
    *,
    sfreq,
    nperseg,
    hr_search_hz=(0.4, 3.0),
):
    ecg_o = np.asarray(ecg_open, dtype=np.float64).reshape(-1)
    ecg_c = np.asarray(ecg_close, dtype=np.float64).reshape(-1)
    ecg = np.concatenate([ecg_o, ecg_c], axis=0)
    if ecg.size == 0:
        return float("nan")

    ecg = ecg - float(np.nanmean(ecg))
    out = welch_psd(ecg, sfreq=float(sfreq), nperseg=int(nperseg), axis=-1)
    f = out.freqs
    p = np.asarray(out.psd, dtype=np.float64).reshape(-1)

    lo, hi = float(hr_search_hz[0]), float(hr_search_hz[1])
    mask = (lo < f) & (f < hi)
    if not np.any(mask):
        return float("nan")

    f_use = f[mask]
    p_use = p[mask]
    if p_use.size == 0:
        return float("nan")
    return float(f_use[int(np.argmax(p_use))])


def _harmonic_bands(
    hr_hz,
    *,
    harmonics,
    half_width_hz,
    fmax,
):
    if not np.isfinite(hr_hz) or hr_hz <= 0:
        return ()
    if harmonics <= 0:
        return ()
    if half_width_hz <= 0:
        raise ValueError(f"half_width_hz must be positive, got {half_width_hz}")
    if fmax <= 0:
        raise ValueError(f"fmax must be positive, got {fmax}")

    bands = []
    for k in range(1, int(harmonics) + 1):
        f0 = float(k) * float(hr_hz)
        if f0 > float(fmax):
            break
        bands.append((max(0.0, f0 - float(half_width_hz)), f0 + float(half_width_hz)))
    return tuple(bands)


def _rta_rms(
    eeg,
    qrs_peaks,
    *,
    sfreq,
    pre_s,
    post_s,
):
    x = np.asarray(eeg, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"eeg must have shape (n_channels,n_samples), got {x.shape}")
    n_ch, n_samples = x.shape

    pre_n = int(round(float(pre_s) * float(sfreq)))
    post_n = int(round(float(post_s) * float(sfreq)))
    if pre_n < 0 or post_n <= 0:
        raise ValueError(f"Invalid pre_s/post_s: {pre_s}, {post_s}")

    offsets = np.arange(-pre_n, post_n, dtype=np.int64)
    win_len = int(offsets.size)

    qrs = np.asarray(qrs_peaks, dtype=np.int64).reshape(-1)
    if qrs.size == 0:
        return np.full(n_ch, np.nan, dtype=np.float64), 0

    valid = qrs[(qrs - pre_n >= 0) & (qrs + post_n <= n_samples)]
    if valid.size == 0:
        return np.full(n_ch, np.nan, dtype=np.float64), 0

    acc = np.zeros((n_ch, win_len), dtype=np.float64)
    for p in valid:
        idx = (int(p) + offsets).astype(np.int64, copy=False)
        acc += x[:, idx]

    avg = acc / float(valid.size)
    avg = avg - np.mean(avg, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(avg * avg, axis=1))
    return rms.astype(np.float64, copy=False), int(valid.size)


def _nanmean_pair(a, b):
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("a/b shape mismatch")

    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    mx = np.isfinite(x)
    my = np.isfinite(y)
    both = mx & my
    out[both] = (x[both] + y[both]) / 2.0
    out[mx & ~my] = x[mx & ~my]
    out[~mx & my] = y[~mx & my]
    return out


def _coherence_harmonics_mean(
    eeg,
    ecg,
    *,
    sfreq,
    nperseg,
    harmonic_bands,
):
    x = np.asarray(eeg, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"eeg must have shape (n_channels,n_samples), got {x.shape}")
    y = np.asarray(ecg, dtype=np.float64).reshape(-1)
    if x.shape[-1] != y.shape[0]:
        raise ValueError("ecg length must match eeg samples")
    if not harmonic_bands:
        return np.full(x.shape[0], np.nan, dtype=np.float64)

    f, cxy = coherence(x, y, fs=float(sfreq), nperseg=int(nperseg), axis=-1)
    band_mask = _multi_band_mask(f, harmonic_bands)
    if not np.any(band_mask):
        return np.full(x.shape[0], np.nan, dtype=np.float64)
    return np.nanmean(cxy[:, band_mask], axis=1).astype(np.float64, copy=False)


class BCGMetrics:
    """
    Five complementary metrics to evaluate BCG filtering.

    - `ari` (Alpha Reactivity Index): log(Pα_close / Pα_open). Higher is better.
    - `api` (Alpha Peakiness Index): Pα_close / Pneighbor_close. Higher is better.
    - `hpr` (HR Harmonics Power Ratio): P(harmonics of HR) / P(broadband). Lower is better.
    - `rta_rms` (R-peak Triggered Average RMS): RMS of ECG-locked average waveform. Lower is better.
    - `coh_hr` (ECG–EEG Coherence @ HR harmonics): mean coherence in harmonic bands. Lower is better.

    Alphahb（與其他指標一致的介面）：
    - `alphahb`：每個 channel 一個值（array, shape = (n_channels,)）
    - `alphahb_avg`：跨 channel 的平均（float）
    - `alphahb_stats`：完整統計物件（含 hb_band/alpha_diff/hb_avg）
    """

    def __init__(
        self,
        ari,
        api,
        hpr,
        rta_rms,
        coh_hr,
        hr_hz,
        harmonic_bands,
        alpha_band,
        neighbor_bands,
        broadband_band,
        qrs_n_open,
        qrs_n_close,
        ari_avg,
        api_avg,
        hpr_avg,
        rta_rms_avg,
        coh_hr_avg,
        alphahb,
        alphahb_avg,
        *,
        alphahb_stats=None,
    ):
        self.ari = np.asarray(ari, dtype=np.float64)
        self.api = np.asarray(api, dtype=np.float64)
        self.hpr = np.asarray(hpr, dtype=np.float64)
        self.rta_rms = np.asarray(rta_rms, dtype=np.float64)
        self.coh_hr = np.asarray(coh_hr, dtype=np.float64)

        # alphahb per-channel like others
        self.alphahb = np.asarray(alphahb, dtype=np.float64)
        self.alphahb_avg = float(alphahb_avg)

        self.hr_hz = float(hr_hz)
        self.harmonic_bands = tuple(harmonic_bands)
        self.alpha_band = tuple(alpha_band)
        self.neighbor_bands = tuple(neighbor_bands)
        self.broadband_band = tuple(broadband_band)
        self.qrs_n_open = int(qrs_n_open)
        self.qrs_n_close = int(qrs_n_close)

        self.ari_avg = float(ari_avg)
        self.api_avg = float(api_avg)
        self.hpr_avg = float(hpr_avg)
        self.rta_rms_avg = float(rta_rms_avg)
        self.coh_hr_avg = float(coh_hr_avg)

        self.alphahb_stats = alphahb_stats


def bcg_metrics(
    eeg_open,
    eeg_close,
    ecg_open,
    ecg_close,
    *,
    sfreq,
    nperseg=1024,
    alpha_band=(8.0, 12.0),
    neighbor_bands=((5.0, 7.0), (13.0, 15.0)),
    broadband_band=(1.0, 40.0),
    harmonics=3,
    harmonic_half_width_hz=0.5,
    compute_hpr=True,
    rta_pre_s=0.2,
    rta_post_s=0.6,
    r_trigger=None,
    coherence_nperseg=None,
    # ---- merged alphahb controls ----
    compute_alphahb=True,
    alphahb_hb_half_width_hz=0.5,
):
    """
    Compute BCG metrics.

    回傳的 BCGMetrics 物件中：
    - ari/api/hpr/rta_rms/coh_hr：皆為 per-channel array
    - ari_avg/api_avg/...：皆為跨 channel 平均
    - alphahb：per-channel array（與 ari 同型）
    - alphahb_avg：跨 channel 平均（與 ari_avg 同型）
    - alphahb_stats：完整 alphahb 統計（可取 hb_band 等）
    """
    from debcg.qrs import qrs

    eeg_o = np.asarray(eeg_open, dtype=np.float64)
    eeg_c = np.asarray(eeg_close, dtype=np.float64)
    if eeg_o.ndim != 2 or eeg_c.ndim != 2:
        raise ValueError("eeg_open/eeg_close must have shape (n_channels,n_samples)")
    if eeg_o.shape != eeg_c.shape:
        raise ValueError("eeg_open/eeg_close shape mismatch")

    ecg_o = np.asarray(ecg_open, dtype=np.float64).reshape(-1)
    ecg_c = np.asarray(ecg_close, dtype=np.float64).reshape(-1)
    if ecg_o.shape[0] != eeg_o.shape[1] or ecg_c.shape[0] != eeg_c.shape[1]:
        raise ValueError("ecg_open/ecg_close length must match eeg samples")

    trig_open = None
    trig_close = None
    if r_trigger is None:
        pass
    elif isinstance(r_trigger, (tuple, list)) and len(r_trigger) == 2:
        trig_open, trig_close = r_trigger
    elif isinstance(r_trigger, dict):
        trig_open = r_trigger.get("open")
        trig_close = r_trigger.get("close")
    else:
        raise TypeError("r_trigger must be None, (r_open, r_close), or {'open': ..., 'close': ...}")

    qrs_o = qrs(ecg_o, float(sfreq)) if trig_open is None else np.asarray(trig_open, dtype=np.int64).reshape(-1)
    qrs_c = qrs(ecg_c, float(sfreq)) if trig_close is None else np.asarray(trig_close, dtype=np.int64).reshape(-1)

    rr = np.concatenate(
        [
            _rr_intervals_s(qrs_o, sfreq=float(sfreq)),
            _rr_intervals_s(qrs_c, sfreq=float(sfreq)),
        ],
        axis=0,
    )
    if rr.size > 0:
        hr_hz = float(1.0 / float(np.median(rr)))
    else:
        hr_hz = _estimate_hr_hz_from_psd(ecg_o, ecg_c, sfreq=float(sfreq), nperseg=int(nperseg))

    b0, b1 = float(broadband_band[0]), float(broadband_band[1])
    harmonic_bands = _harmonic_bands(
        hr_hz,
        harmonics=int(harmonics),
        half_width_hz=float(harmonic_half_width_hz),
        fmax=b1,
    )
    nperseg_coh = int(nperseg if coherence_nperseg is None else coherence_nperseg)

    psd_o = welch_psd(eeg_o, sfreq=float(sfreq), nperseg=int(nperseg), axis=-1)
    psd_c = welch_psd(eeg_c, sfreq=float(sfreq), nperseg=int(nperseg), axis=-1)
    if psd_o.freqs.shape != psd_c.freqs.shape or not np.allclose(psd_o.freqs, psd_c.freqs):
        raise RuntimeError("Welch frequency grids differ between open/close")
    freqs = psd_o.freqs

    a0, a1 = float(alpha_band[0]), float(alpha_band[1])
    alpha_mask = (a0 < freqs) & (freqs < a1)

    alpha_open = np.sum(psd_o.psd[:, alpha_mask], axis=1)
    alpha_close = np.sum(psd_c.psd[:, alpha_mask], axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ari = np.log(alpha_close) - np.log(alpha_open)
    ari = np.where((alpha_open > 0) & (alpha_close > 0), ari, np.nan).astype(np.float64, copy=False)

    neigh_mask = _multi_band_mask(freqs, tuple((float(x0), float(x1)) for x0, x1 in neighbor_bands))
    neigh_close = np.sum(psd_c.psd[:, neigh_mask], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        api = alpha_close / neigh_close
    api = np.where(neigh_close == 0, np.nan, api).astype(np.float64, copy=False)

    if compute_hpr:
        broad_mask = (b0 < freqs) & (freqs < b1)
        harm_mask = _multi_band_mask(freqs, harmonic_bands)

        def _harm_ratio(psd):
            if not np.any(broad_mask) or not np.any(harm_mask):
                return np.full(psd.shape[0], np.nan, dtype=np.float64)
            harm_pow = np.sum(psd[:, harm_mask], axis=1)
            broad_pow = np.sum(psd[:, broad_mask], axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = harm_pow / broad_pow
            r = np.where(broad_pow == 0, np.nan, r)
            return r.astype(np.float64, copy=False)

        hpr_open = _harm_ratio(psd_o.psd)
        hpr_close = _harm_ratio(psd_c.psd)
        hpr = _nanmean_pair(hpr_open, hpr_close)
    else:
        hpr = np.full(eeg_o.shape[0], np.nan, dtype=np.float64)

    rta_open, _n_open = _rta_rms(
        eeg_o,
        qrs_o,
        sfreq=float(sfreq),
        pre_s=float(rta_pre_s),
        post_s=float(rta_post_s),
    )
    rta_close, _n_close = _rta_rms(
        eeg_c,
        qrs_c,
        sfreq=float(sfreq),
        pre_s=float(rta_pre_s),
        post_s=float(rta_post_s),
    )
    rta_rms = _nanmean_pair(rta_open, rta_close)

    coh_open = _coherence_harmonics_mean(
        eeg_o,
        ecg_o,
        sfreq=float(sfreq),
        nperseg=nperseg_coh,
        harmonic_bands=harmonic_bands,
    )
    coh_close = _coherence_harmonics_mean(
        eeg_c,
        ecg_c,
        sfreq=float(sfreq),
        nperseg=nperseg_coh,
        harmonic_bands=harmonic_bands,
    )
    coh_hr = _nanmean_pair(coh_open, coh_close)

    # ---- alphahb (make it like other metrics) ----
    alphahb_stats = None
    alphahb_scores = np.full(eeg_o.shape[0], np.nan, dtype=np.float64)
    alphahb_avg = float("nan")

    if compute_alphahb:
        alphahb_stats = alpha_hb_stats(
            eeg_open,
            eeg_close,
            ecg_open,
            ecg_close,
            sfreq=float(sfreq),
            nperseg=int(nperseg),
            alpha_band=(a0, a1),
            hb_half_width_hz=float(alphahb_hb_half_width_hz),
        )
        alphahb_scores = np.asarray(alphahb_stats.scores, dtype=np.float64)
        alphahb_avg = float(alphahb_stats.avg)

    return BCGMetrics(
        ari=ari,
        api=api,
        hpr=hpr,
        rta_rms=rta_rms,
        coh_hr=coh_hr,
        hr_hz=float(hr_hz) if np.isfinite(hr_hz) else float("nan"),
        harmonic_bands=harmonic_bands,
        alpha_band=(a0, a1),
        neighbor_bands=tuple((float(x0), float(x1)) for x0, x1 in neighbor_bands),
        broadband_band=(b0, b1),
        qrs_n_open=int(qrs_o.size),
        qrs_n_close=int(qrs_c.size),
        ari_avg=_nanmean_or_nan(ari),
        api_avg=_nanmean_or_nan(api),
        hpr_avg=_nanmean_or_nan(hpr),
        rta_rms_avg=_nanmean_or_nan(rta_rms),
        coh_hr_avg=_nanmean_or_nan(coh_hr),
        alphahb=alphahb_scores,
        alphahb_avg=alphahb_avg,
        alphahb_stats=alphahb_stats,
    )
