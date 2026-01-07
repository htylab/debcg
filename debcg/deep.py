import numpy as np

class TrainConfig:
    """
    Unified training/inference protocol shared by BRNet and BCGNet.

    Notes
    -----
    - Epoching is fixed-length, non-overlapping (default 4 s).
    - MAD rejection matches the BCGNet-style EEG epoch criterion:
      mean(abs(EEG_epoch_z)) -> MAD -> z > threshold.
    - Inference supports three strategies:
      - "chunk": non-overlapping windows (fast, may have seams).
      - "overlap": sliding-window with blending (slower, fewer seams).
      - "full": single forward pass over the full recording (may be memory-heavy).
    """

    FIELD_NAMES = (
        "epoch_s",
        "mad_threshold",
        "per_training",
        "per_valid",
        "per_test",
        "num_epochs",
        "lr",
        "batch_size",
        "es_patience",
        "es_min_delta",
        "random_seed",
        "device",
        "verbose",
        "inference_mode",
        "inference_overlap",
    )

    def __init__(
        self,
        epoch_s=4.0,
        mad_threshold=7.0,
        per_training=0.7,
        per_valid=0.15,
        per_test=0.15,
        num_epochs=200,
        lr=1e-3,
        batch_size=1,
        es_patience=10,
        es_min_delta=1e-5,
        random_seed=1997,
        device=None,
        verbose=True,
        inference_mode="chunk",
        inference_overlap=0.5,
    ):
        self.epoch_s = epoch_s
        self.mad_threshold = mad_threshold
        self.per_training = per_training
        self.per_valid = per_valid
        self.per_test = per_test
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.es_patience = es_patience
        self.es_min_delta = es_min_delta
        self.random_seed = random_seed
        self.device = device
        self.verbose = verbose
        self.inference_mode = inference_mode
        self.inference_overlap = inference_overlap


def _median_abs_deviation_1d(x):
    x1 = np.asarray(x, dtype=np.float64).reshape(-1)
    med = float(np.median(x1))
    return float(np.median(np.abs(x1 - med)))


def standardize_eeg_ecg(
    eeg,
    ecg,
    *,
    eps=1e-12,
):
    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2:
        raise ValueError(f"eeg must have shape (n_channels, n_samples), got {eeg.shape}")
    n_ch, n_samples = eeg.shape

    ecg = np.asarray(ecg, dtype=np.float64).reshape(-1)
    if ecg.shape[0] != n_samples:
        raise ValueError(f"ecg length {ecg.shape[0]} must match eeg samples {n_samples}")

    eeg_mean = np.mean(eeg, axis=1, keepdims=True)
    eeg_std = np.std(eeg, axis=1, keepdims=True)
    eeg_std = np.where(eeg_std < eps, 1.0, eeg_std)
    eeg_z = (eeg - eeg_mean) / eeg_std

    ecg_mean = float(np.mean(ecg))
    ecg_std = float(np.std(ecg))
    if ecg_std < eps:
        ecg_std = 1.0
    ecg_z = (ecg - ecg_mean) / ecg_std

    if eeg_z.shape != (n_ch, n_samples) or ecg_z.shape != (n_samples,):
        raise RuntimeError("Unexpected standardized shapes")

    return eeg_z, ecg_z, eeg_mean.astype(np.float64, copy=False), eeg_std.astype(np.float64, copy=False), ecg_mean, ecg_std


def make_epochs_nonoverlap(
    eeg_z,
    ecg_z,
    *,
    sfreq,
    epoch_s,
):
    if sfreq <= 0:
        raise ValueError(f"sfreq must be positive, got {sfreq}")
    if epoch_s <= 0:
        raise ValueError(f"epoch_s must be positive, got {epoch_s}")

    eeg_z = np.asarray(eeg_z, dtype=np.float64)
    ecg_z = np.asarray(ecg_z, dtype=np.float64).reshape(-1)
    if eeg_z.ndim != 2:
        raise ValueError("eeg_z must be 2D")
    n_ch, n_samples = eeg_z.shape
    if ecg_z.shape[0] != n_samples:
        raise ValueError("ecg_z length mismatch")

    epoch_len = int(round(float(epoch_s) * float(sfreq)))
    if epoch_len <= 0:
        raise ValueError("epoch_len must be >= 1")

    n_epochs = int(n_samples // epoch_len)
    if n_epochs <= 0:
        return np.zeros((0, 1, epoch_len), dtype=np.float64), np.zeros((0, n_ch, epoch_len), dtype=np.float64), epoch_len

    starts = (np.arange(n_epochs, dtype=np.int64) * epoch_len).tolist()
    x = np.stack([ecg_z[s : s + epoch_len] for s in starts], axis=0).astype(np.float64, copy=False)
    y = np.stack([eeg_z[:, s : s + epoch_len] for s in starts], axis=0).astype(np.float64, copy=False)

    x = x[:, None, :]  # (n_epoch, 1, time)
    if x.shape != (n_epochs, 1, epoch_len) or y.shape != (n_epochs, n_ch, epoch_len):
        raise RuntimeError("Unexpected epoch tensor shapes")

    return x, y, epoch_len


def reject_epochs_mad_bcgnet_style(
    y_epochs_z,
    *,
    mad_threshold,
):
    y = np.asarray(y_epochs_z, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError(f"y_epochs_z must be 3D (n_epoch,n_ch,time), got {y.shape}")
    if mad_threshold <= 0:
        raise ValueError("mad_threshold must be positive")

    score = np.mean(np.abs(y), axis=(1, 2))
    med = float(np.median(score))
    mad = _median_abs_deviation_1d(score)
    if mad == 0:
        return np.arange(y.shape[0], dtype=np.int64)

    z = (score - med) / mad
    idx_good = np.where(z <= float(mad_threshold))[0].astype(np.int64, copy=False)
    return idx_good


def split_train_valid_test(
    n_epochs,
    *,
    per_training,
    per_valid,
    per_test,
    random_seed,
):
    if n_epochs <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if per_training < 0 or per_valid < 0 or per_test < 0:
        raise ValueError("Split percentages must be non-negative")
    if per_training + per_valid > 1:
        raise ValueError("per_training + per_valid must be <= 1")

    rng = np.random.default_rng(int(random_seed))
    idx = rng.permutation(int(n_epochs))
    n_train = int(round(n_epochs * float(per_training)))
    n_valid = int(round(n_epochs * float(per_valid)))
    idx_train = idx[:n_train]
    idx_valid = idx[n_train : n_train + n_valid]
    idx_test = idx[n_train + n_valid :]
    return idx_train.astype(np.int64), idx_valid.astype(np.int64), idx_test.astype(np.int64)


def train_and_apply(
    eeg,
    ecg,
    sfreq,
    *,
    model,
    config,
):
    """
    Train a model to predict EEG-from-ECG and return cleaned EEG.

    The model must accept input shape (batch, 1, time) and output
    (batch, n_channels, time).
    """
    try:
        import torch
        import torch.nn as nn
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required for debcg deep models") from e

    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2:
        raise ValueError(f"eeg must have shape (n_channels, n_samples); got {eeg.shape}")
    n_ch, n_samples = eeg.shape
    ecg = np.asarray(ecg, dtype=np.float64).reshape(-1)
    if ecg.shape[0] != n_samples:
        raise ValueError("ecg length must match eeg samples")

    if config.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(config.device)

    eeg_z, ecg_z, eeg_mean, eeg_std, _ecg_mean, _ecg_std = standardize_eeg_ecg(eeg, ecg)
    x_ep, y_ep, epoch_len = make_epochs_nonoverlap(eeg_z, ecg_z, sfreq=float(sfreq), epoch_s=float(config.epoch_s))
    if x_ep.shape[0] == 0:
        return eeg.copy()

    idx_good = reject_epochs_mad_bcgnet_style(y_ep, mad_threshold=float(config.mad_threshold))
    if idx_good.size == 0:
        return eeg.copy()

    x_ep = x_ep[idx_good]
    y_ep = y_ep[idx_good]

    idx_train, idx_valid, _idx_test = split_train_valid_test(
        x_ep.shape[0],
        per_training=float(config.per_training),
        per_valid=float(config.per_valid),
        per_test=float(config.per_test),
        random_seed=int(config.random_seed),
    )
    if idx_train.size == 0 or idx_valid.size == 0:
        return eeg.copy()

    x_train = x_ep[idx_train]
    y_train = y_ep[idx_train]
    x_valid = x_ep[idx_valid]
    y_valid = y_ep[idx_valid]

    torch.manual_seed(int(config.random_seed))
    np.random.seed(int(config.random_seed))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    epochs_no_improve = 0

    rng = np.random.default_rng(int(config.random_seed))

    def _iter_batches(x_np, y_np, *, shuffle):
        idx = np.arange(x_np.shape[0], dtype=np.int64)
        if shuffle:
            rng.shuffle(idx)
        bs = max(1, int(config.batch_size))
        for start in range(0, idx.size, bs):
            sel = idx[start : start + bs]
            yield x_np[sel], y_np[sel]

    for epoch in range(int(config.num_epochs)):
        model.train()
        train_losses = []
        for xb_np, yb_np in _iter_batches(x_train, y_train, shuffle=True):
            xb = torch.from_numpy(np.ascontiguousarray(xb_np)).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(np.ascontiguousarray(yb_np)).to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            if pred.shape != yb.shape:
                raise RuntimeError(f"Model output shape {tuple(pred.shape)} does not match target {tuple(yb.shape)}")
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb_np, yb_np in _iter_batches(x_valid, y_valid, shuffle=False):
                xb = torch.from_numpy(np.ascontiguousarray(xb_np)).to(device=device, dtype=torch.float32)
                yb = torch.from_numpy(np.ascontiguousarray(yb_np)).to(device=device, dtype=torch.float32)
                pred = model(xb)
                if pred.shape != yb.shape:
                    raise RuntimeError("Validation: model output shape mismatch")
                val_losses.append(float(loss_fn(pred, yb).detach().cpu().item()))

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if config.verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
            tr_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            print(f"[debcg] epoch {epoch+1:04d} train={tr_loss:.6f} valid={val_loss:.6f}")

        if val_loss < best_val - float(config.es_min_delta):
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= int(config.es_patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _infer_chunk():
        pred = np.zeros((n_ch, n_samples), dtype=np.float64)
        model.eval()
        with torch.no_grad():
            for start in range(0, n_samples, epoch_len):
                stop = min(start + epoch_len, n_samples)
                seg_len = int(stop - start)
                # UNet-style models with 4x pooling need at least 16 samples.
                if seg_len < 16:
                    break
                xb_np = ecg_z[start:stop].astype(np.float64, copy=False)[None, None, :]
                xb = torch.from_numpy(np.ascontiguousarray(xb_np)).to(device=device, dtype=torch.float32)
                yb = model(xb).detach().cpu().numpy()[0].astype(np.float64, copy=False)
                if yb.shape != (n_ch, seg_len):
                    raise RuntimeError(f"Inference(chunk): expected {(n_ch, seg_len)}, got {yb.shape}")
                pred[:, start:stop] = yb
        return pred

    def _infer_full():
        if n_samples < 16:
            return np.zeros((n_ch, n_samples), dtype=np.float64)
        xb_np = ecg_z.astype(np.float64, copy=False)[None, None, :]
        xb = torch.from_numpy(np.ascontiguousarray(xb_np)).to(device=device, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            yb = model(xb).detach().cpu().numpy()[0].astype(np.float64, copy=False)
        if yb.shape != (n_ch, n_samples):
            raise RuntimeError(f"Inference(full): expected {(n_ch, n_samples)}, got {yb.shape}")
        return yb

    def _gaussian_weights_1d(length, *, sigma_scale=0.125):
        if length <= 0:
            raise ValueError("length must be positive")
        if sigma_scale <= 0:
            raise ValueError("sigma_scale must be positive")
        if length == 1:
            return np.ones(1, dtype=np.float64)
        center = (length - 1) / 2.0
        sigma = float(sigma_scale) * float(length)
        if sigma <= 0:
            return np.ones(length, dtype=np.float64)
        x = (np.arange(length, dtype=np.float64) - center) / sigma
        w = np.exp(-0.5 * x * x)
        w = np.maximum(w, 1e-12)
        return w.astype(np.float64, copy=False)

    def _infer_overlap():
        overlap = float(config.inference_overlap)
        if not (0.0 < overlap < 1.0):
            raise ValueError(f"inference_overlap must be in (0,1), got {overlap}")

        win_len = int(epoch_len)
        if n_samples < 16:
            return np.zeros((n_ch, n_samples), dtype=np.float64)
        if win_len < 16 or win_len > n_samples:
            return _infer_full()

        step = int(round(win_len * (1.0 - overlap)))
        step = max(1, step)

        starts = list(range(0, n_samples - win_len + 1, step))
        last = int(n_samples - win_len)
        if not starts or starts[-1] != last:
            starts.append(last)

        pred_sum = np.zeros((n_ch, n_samples), dtype=np.float64)
        w_sum = np.zeros((n_samples,), dtype=np.float64)
        w_win = _gaussian_weights_1d(win_len)

        model.eval()
        with torch.no_grad():
            for start in starts:
                stop = start + win_len
                xb_np = ecg_z[start:stop].astype(np.float64, copy=False)[None, None, :]
                xb = torch.from_numpy(np.ascontiguousarray(xb_np)).to(device=device, dtype=torch.float32)
                yb = model(xb).detach().cpu().numpy()[0].astype(np.float64, copy=False)
                if yb.shape != (n_ch, win_len):
                    raise RuntimeError(f"Inference(overlap): expected {(n_ch, win_len)}, got {yb.shape}")
                pred_sum[:, start:stop] += yb * w_win[None, :]
                w_sum[start:stop] += w_win

        w_sum = np.maximum(w_sum, 1e-12)
        return (pred_sum / w_sum[None, :]).astype(np.float64, copy=False)

    mode = str(config.inference_mode)
    if mode == "chunk":
        pred_z = _infer_chunk()
    elif mode == "full":
        pred_z = _infer_full()
    elif mode == "overlap":
        pred_z = _infer_overlap()
    else:  # pragma: no cover
        raise ValueError(f"Unknown inference_mode: {mode!r}")

    eeg_clean_z = eeg_z - pred_z
    eeg_clean = (eeg_clean_z * eeg_std) + eeg_mean
    return eeg_clean.astype(np.float64, copy=False)
