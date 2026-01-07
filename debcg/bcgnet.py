from types import ModuleType

import numpy as np


class BCGNetConfig:
    """
    BCGNet (RNN) configuration for the unified debcg training/inference protocol.
    Origin: https://github.com/jiaangyao/BCGNet

    Notes
    -----
    The training/inference loop is intentionally shared with BRNet via `debcg.deep`.
    Only the model architecture and model-specific parameters differ.
    """

    FIELD_NAMES = ("train", "dropout")

    def __init__(self, train=None, dropout=0.327):
        if train is None:
            train = _default_train_config()
        self.train = train
        self.dropout = dropout


def _default_train_config():
    from debcg.deep import TrainConfig

    return TrainConfig()


def _make_model(*, n_eeg, cfg):
    try:
        import torch
        import torch.nn as nn
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required for BCGNet") from e

    class BCGNetRNN(nn.Module):
        def __init__(self, n_output, dropout):
            super().__init__()
            self.gru1 = nn.GRU(input_size=1, hidden_size=16, batch_first=True, bidirectional=True)
            self.gru2 = nn.GRU(input_size=32, hidden_size=16, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(32, 8)
            self.drop = nn.Dropout(p=float(dropout))
            self.gru3 = nn.GRU(input_size=8, hidden_size=16, batch_first=True, bidirectional=True)
            self.gru4 = nn.GRU(input_size=32, hidden_size=64, batch_first=True, bidirectional=True)
            self.out = nn.Linear(128, n_output)

        def forward(self, x):
            # x: (batch, 1, time) -> (batch, time, 1)
            x = torch.transpose(x, 1, 2)
            x, _ = self.gru1(x)
            x, _ = self.gru2(x)
            x = torch.relu(self.fc(x))
            x = self.drop(x)
            x, _ = self.gru3(x)
            x, _ = self.gru4(x)
            x = self.out(x)  # (batch, time, n_output)
            return torch.transpose(x, 1, 2)  # (batch, n_output, time)

    return BCGNetRNN(n_output=int(n_eeg), dropout=float(cfg.dropout))


def _cfg_from_dict(d):
    from debcg.deep import TrainConfig

    bcg_fields = set(BCGNetConfig.FIELD_NAMES)
    train_fields = set(TrainConfig.FIELD_NAMES)

    bcg_kwargs = {}
    train_kwargs = {}

    for k, v in dict(d).items():
        if k == "train":
            if isinstance(v, TrainConfig):
                bcg_kwargs["train"] = v
            elif isinstance(v, dict):
                train_kwargs.update(v)
            else:
                raise TypeError("BCGNetConfig['train'] must be TrainConfig or dict")
        elif k in bcg_fields:
            bcg_kwargs[k] = v
        elif k in train_fields:
            train_kwargs[k] = v

    base_train = TrainConfig()
    if train_kwargs:
        base_train = TrainConfig(**{k: train_kwargs[k] for k in train_kwargs if k in train_fields})
    if "train" not in bcg_kwargs:
        bcg_kwargs["train"] = base_train
    return BCGNetConfig(**bcg_kwargs)


def run(
    eeg,
    ecg,
    config=None,
    *,
    sfreq=500.0,
    device=None,
):
    """
    Train BCGNet per-recording and return cleaned EEG.

    Call signature matches the unified comparison pipeline:
    `debcg.bcgnet(EEG, ECG, bcgnet_config) -> filtered_EEG`.

    Parameters
    ----------
    device : str or None
        Device override. If provided, overrides config.train.device.
        Use 'cuda', 'cpu', or None (auto-detect).
    """
    from debcg.deep import train_and_apply

    if config is None:
        cfg = BCGNetConfig()
    elif isinstance(config, BCGNetConfig):
        cfg = config
    elif isinstance(config, dict):
        cfg = _cfg_from_dict(config)
    else:
        raise TypeError(f"config must be BCGNetConfig | dict | None, got {type(config)!r}")

    # Override device if provided at call site
    train_cfg = cfg.train
    if device is not None:
        from debcg.deep import TrainConfig
        train_cfg = TrainConfig(
            epoch_s=train_cfg.epoch_s,
            mad_threshold=train_cfg.mad_threshold,
            per_training=train_cfg.per_training,
            per_valid=train_cfg.per_valid,
            per_test=train_cfg.per_test,
            num_epochs=train_cfg.num_epochs,
            lr=train_cfg.lr,
            batch_size=train_cfg.batch_size,
            es_patience=train_cfg.es_patience,
            es_min_delta=train_cfg.es_min_delta,
            random_seed=train_cfg.random_seed,
            device=device,
            verbose=train_cfg.verbose,
            inference_mode=train_cfg.inference_mode,
            inference_overlap=train_cfg.inference_overlap,
        )

    eeg_np = np.asarray(eeg, dtype=np.float64)
    if eeg_np.ndim != 2:
        raise ValueError(f"eeg must have shape (n_channels,n_samples), got {eeg_np.shape}")
    n_eeg = int(eeg_np.shape[0])

    model = _make_model(n_eeg=n_eeg, cfg=cfg)
    return train_and_apply(eeg_np, np.asarray(ecg, dtype=np.float64), float(sfreq), model=model, config=train_cfg)


def bcgnet(
    eeg,
    ecg,
    config=None,
    *,
    sfreq=500.0,
    device=None,
):
    return run(eeg, ecg, config, sfreq=sfreq, device=device)


class _CallableModule(ModuleType):
    def __call__(self, eeg, ecg, config=None, *, sfreq=500.0, device=None, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        return run(eeg, ecg, config, sfreq=sfreq, device=device)


import sys as _sys

_sys.modules[__name__].__class__ = _CallableModule
