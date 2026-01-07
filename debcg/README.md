# debcg

`debcg` is a toolbox for **EEG-fMRI ballistocardiogram (BCG) artifact suppression** with a **unified pipeline** so different algorithms can be compared on the same basis.

## Quick Start

All four BCG correction methods can be called with minimal arguments. Config parameters are optional and use sensible defaults.

```python
import debcg

# Load your data: EEG (n_channels, n_samples), ECG (n_samples,)
# Both should be preprocessed (500 Hz, 0.25 Hz high-pass)

# Template-based methods (no GPU required)
eeg_obs = debcg.obs(eeg, ecg, sfreq=500.0)   # OBS: bcg_nsvd=3, n_ma_bcg=21
eeg_dmh = debcg.dmh(eeg, ecg, sfreq=500.0)   # DMH: nn=10

# Deep learning methods (GPU optional)
eeg_bcgnet = debcg.bcgnet(eeg, ecg, sfreq=500.0)              # auto-detect device
eeg_brnet = debcg.brnet(eeg, ecg, sfreq=500.0, device='cuda') # force GPU
eeg_brnet = debcg.brnet(eeg, ecg, sfreq=500.0, device='cpu')  # force CPU
```

### Custom configurations (optional)

```python
# OBS with custom parameters
cfg = debcg.obs.OBSConfig(bcg_nsvd=5, n_ma_bcg=31)
eeg_obs = debcg.obs(eeg, ecg, cfg, sfreq=500.0)

# DMH with custom parameters
cfg = debcg.dmh.DMHConfig(nn=15, outlier_threshold_factor=5.0)
eeg_dmh = debcg.dmh(eeg, ecg, cfg, sfreq=500.0)

# BCGNet/BRNet with custom TrainConfig
train_cfg = debcg.deep.TrainConfig(num_epochs=100, verbose=False)
cfg = debcg.bcgnet.BCGNetConfig(train=train_cfg)
eeg_bcgnet = debcg.bcgnet(eeg, ecg, cfg, sfreq=500.0, device='cuda')
```

## Design concept (unified protocol)

All algorithms start from the same MATLAB-exported inputs (from `read_smsini_eeg.m`):
- `EEG_before_bcg_{open,close}` (not device-raw)
- `ECG_{open,close}`

All algorithms share the same baseline preprocessing and output conventions:
- Downsample to **500 Hz** (polyphase resampling)
- EEG **zero-phase high-pass 0.25 Hz** (5th-order Butterworth IIR)
- **Min-length cut** so open/close are comparable
- No `TRIGGER_ECG_*` inputs are used; timing is derived from the provided ECG via `debcg.qrs`
- Output is always `filtered_EEG` with the same shape `(n_channels, n_samples)`

## Modules

### Core algorithms
- `debcg.obs`: OBS/PC style correction (MATLAB `eeg_bcg.m`-style). Parameters: `n_svd=3`, `n_ma_bcg=21`, RR10 window (20%/80%).
- `debcg.dmh`: Dynamic Modeling of Heartbeats (DMH). Parameters: `k=10` neighbors, MAD threshold=4.0.
- `debcg.brnet`: BRNet (UNet1d). Architecture: kernel=5, n_filter=16, GroupNorm 8 groups, 5 encoder levels.
- `debcg.bcgnet`: BCGNet (RNN). Architecture: 4 bidirectional GRU layers (16,16,16,64), dense 8, dropout=0.327.

### Utilities
- `debcg.qrs`: Unified QRS detection using Pan-Tompkins algorithm (5–15 Hz bandpass, 0.6 s min RR).
- `debcg.preprocessing`: Polyphase downsampling, zero-phase filtering.
- `debcg.stats`: Six-metric evaluation framework (AlphaHB, ARI, API, HPR, RTA_RMS, COH_HR).
- `debcg.deep`: Shared training/inference protocol for deep learning models.

### Alpha band definition
- **8–12 Hz** (used consistently across all evaluation metrics)

### Deep models: identical training/inference loop

`debcg.brnet` and `debcg.bcgnet` intentionally share the same training/inference protocol:
- Per-condition **z-score** using that condition's full recording.
- **4 s non-overlapping epochs** for training.
- **MAD-based epoch rejection** (threshold = 7×MAD).
- Data split: **70%/15%/15%** (train/val/test), random seed **1997**.
- Optimizer: **Adam** (lr=1e-3), batch size **1**, MSE loss.
- Early stopping: patience **10 epochs**, min improvement **1e-5**.
- Inference: chunk-based (non-overlapping), outputs `clean = eeg - pred` and un-zscores.

Only the **model architecture** (UNet vs RNN) and model-specific hyperparameters differ.

## References / provenance

- OBS + DMH: Python reimplementations based on the workflows in `fhlin_toolbox` (https://github.com/fahsuanlin/fhlin_toolbox).
- DMH paper: Lee, H. J., Graham, S. J., Kuo, W. J., & Lin, F. H. (2022). Ballistocardiogram suppression in concurrent EEG-MRI by dynamic modeling of heartbeats. *Human Brain Mapping*, 43(14), 4444–4457. https://doi.org/10.1002/hbm.25965
- BCGNet: PyTorch reimplementation based on https://github.com/jiaangyao/BCGNet and the paper: McIntosh, J. R., Yao, J., Hong, L., Faller, J., & Sajda, P. (2020). Ballistocardiogram artifact reduction in simultaneous EEG-fMRI using deep learning. *IEEE Transactions on Biomedical Engineering*.

