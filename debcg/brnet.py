from types import ModuleType

import numpy as np


class BRNetConfig:
    """
    BRNet configuration for the unified debcg training/inference protocol.
    Origin: https://github.com/WANGICHEN/BRNet

    Notes
    -----
    This implementation is self-contained inside `debcg` (no dependency on the repo-level `brnet/` package).
    """

    FIELD_NAMES = ("train", "nfilter", "kernels", "up_weight")

    def __init__(self, train=None, nfilter=16, kernels=(5,), up_weight=(0, 0, 1, 0)):
        if train is None:
            train = _default_train_config()
        self.train = train
        self.nfilter = nfilter
        self.kernels = kernels
        self.up_weight = up_weight


def _default_train_config():
    from debcg.deep import TrainConfig

    return TrainConfig()


def _make_model(*, n_eeg, cfg):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required for BRNet") from e

    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
            super().__init__()
            if mid_channels is None:
                mid_channels = out_channels
            self.double_conv = nn.Sequential(
                nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(num_groups=8, num_channels=mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.double_conv(x)

    class Down(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_list):
            super().__init__()
            self.modules_list = nn.ModuleList(
                [nn.Sequential(nn.MaxPool1d(2), DoubleConv(in_channels, out_channels, kernel_size=k)) for k in kernel_list]
            )

        def forward(self, x):
            return sum(module(x) for module in self.modules_list)

    class Up(nn.Module):
        def __init__(self, in_channels, out_channels, weight):
            super().__init__()
            self.weight = int(weight)
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            if self.weight == 0:
                self.conv = DoubleConv(in_channels // 2, out_channels, in_channels // 2)
            else:
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            diff_x = x2.size()[2] - x1.size()[2]
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2])
            if self.weight == 1:
                x1 = torch.cat([x2 * self.weight, x1], dim=1)
            return self.conv(x1)

    class OutConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    class UNet1d(nn.Module):
        def __init__(self, n_classes, *, nfilter, kernel_list, up_weight):
            super().__init__()
            self.inc = DoubleConv(1, nfilter)
            self.down1 = Down(nfilter, nfilter * 2, kernel_list)
            self.down2 = Down(nfilter * 2, nfilter * 4, kernel_list)
            self.down3 = Down(nfilter * 4, nfilter * 8, kernel_list)
            self.down4 = Down(nfilter * 8, nfilter * 8, kernel_list)
            self.up1 = Up(nfilter * 16, nfilter * 4, up_weight[0])
            self.up2 = Up(nfilter * 8, nfilter * 2, up_weight[1])
            self.up3 = Up(nfilter * 4, nfilter * 1, up_weight[2])
            self.up4 = Up(nfilter * 2, nfilter, up_weight[3])
            self.outc = OutConv(nfilter, n_classes)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            return self.outc(x)

    return UNet1d(n_classes=int(n_eeg), nfilter=int(cfg.nfilter), kernel_list=tuple(int(k) for k in cfg.kernels), up_weight=cfg.up_weight)


def _cfg_from_dict(d):
    from debcg.deep import TrainConfig

    br_fields = set(BRNetConfig.FIELD_NAMES)
    train_fields = set(TrainConfig.FIELD_NAMES)

    br_kwargs = {}
    train_kwargs = {}

    for k, v in dict(d).items():
        if k == "train":
            if isinstance(v, TrainConfig):
                br_kwargs["train"] = v
            elif isinstance(v, dict):
                train_kwargs.update(v)
            else:
                raise TypeError("BRNetConfig['train'] must be TrainConfig or dict")
        elif k in br_fields:
            br_kwargs[k] = v
        elif k in train_fields:
            train_kwargs[k] = v

    base_train = TrainConfig()
    if train_kwargs:
        base_train = TrainConfig(**{k: train_kwargs[k] for k in train_kwargs if k in train_fields})
    if "train" not in br_kwargs:
        br_kwargs["train"] = base_train
    return BRNetConfig(**br_kwargs)


def run(
    eeg,
    ecg,
    config=None,
    *,
    sfreq=500.0,
    device=None,
):
    """
    Train BRNet per-recording and return cleaned EEG.

    Call signature matches the unified comparison pipeline:
    `debcg.brnet(EEG, ECG, brnet_config) -> filtered_EEG`.

    Parameters
    ----------
    device : str or None
        Device override. If provided, overrides config.train.device.
        Use 'cuda', 'cpu', or None (auto-detect).
    """
    from debcg.deep import train_and_apply

    if config is None:
        cfg = BRNetConfig()
    elif isinstance(config, BRNetConfig):
        cfg = config
    elif isinstance(config, dict):
        cfg = _cfg_from_dict(config)
    else:
        raise TypeError(f"config must be BRNetConfig | dict | None, got {type(config)!r}")

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


def brnet(
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
