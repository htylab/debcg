# debcg

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`debcg` is a unified toolbox for **EEG-fMRI ballistocardiogram (BCG) artifact suppression**. It provides four correction algorithms under a single API, enabling fair comparison across methods.

## Installation

### From GitHub (recommended)

```bash
# Basic installation (OBS, DMH only - no GPU required)
pip install git+https://github.com/your-username/debcg.git

# With deep learning support (BCGNet, BRNet)
pip install "git+https://github.com/your-username/debcg.git#egg=debcg[deep]"
```

### From source

```bash
git clone https://github.com/htylab/debcg.git
cd debcg
pip install -e .           # Basic
pip install -e ".[deep]"   # With PyTorch
pip install -e ".[dev]"    # With dev tools
```

## Quick Start

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

## Algorithms

| Method | Type | Description |
|--------|------|-------------|
| `debcg.obs` | Template-based | OBS/PC correction using SVD on heartbeat-locked epochs |
| `debcg.dmh` | Adaptive | Dynamic Modeling of Heartbeats using k-NN regression |
| `debcg.bcgnet` | Deep Learning | ECG-conditioned RNN (4 bidirectional GRU layers) |
| `debcg.brnet` | Deep Learning | ECG-conditioned 1D U-Net encoder-decoder |

## Unified Protocol

All algorithms share:
- **500 Hz** sampling rate
- **0.25 Hz high-pass** zero-phase filtering
- QRS detection via Pan-Tompkins (5–15 Hz bandpass, 0.6 s min RR)
- Output shape: `(n_channels, n_samples)`

Deep learning methods additionally share:
- Per-recording z-score normalization
- 4 s non-overlapping epochs
- 70%/15%/15% train/val/test split
- Adam optimizer, MSE loss, early stopping

## Evaluation Metrics

`debcg.stats` provides a six-metric evaluation framework:

**Alpha preservation** (higher is better):
- `AlphaHB`: Alpha heartbeat-normalized reactivity
- `ARI`: Alpha reactivity index (log-ratio)
- `API`: Alpha peakiness index

**Artifact reduction** (lower is better):
- `HPR`: Heart-rate harmonics power ratio
- `RTA_RMS`: R-peak triggered average RMS
- `COH_HR`: EEG-ECG coherence at HR harmonics

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- PyTorch ≥ 1.10.0 (optional, for BCGNet/BRNet)

## Citation

If you use this package, please cite:

```bibtex
@article{wang2024debcg,
  title={ECG-Conditioned Learning for Ballistocardiogram Artifact Suppression in EEG-MRI with Validation of Alpha-Band Reactivity},
  author={Wang, I-Chen and Huang, Teng-Yi and Lee, Hsin-Ju and Lin, Fa-Hsuan},
  year={2024}
}
```

## Additional References

- **OBS/DMH**: Python reimplementations based on [fhlin_toolbox](https://github.com/fahsuanlin/fhlin_toolbox)
- **DMH paper**: Lee et al. (2022). Human Brain Mapping, 43(14), 4444–4457. [DOI](https://doi.org/10.1002/hbm.25965)
- **BCGNet**: Based on [jiaangyao/BCGNet](https://github.com/jiaangyao/BCGNet) and McIntosh et al. (2020). IEEE TBME.

## License

MIT License - see [LICENSE](LICENSE) for details.
