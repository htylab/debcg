"""Basic tests for debcg package."""

import numpy as np
import pytest


def test_import():
    """Test that debcg can be imported."""
    import debcg
    assert hasattr(debcg, "__version__")
    assert debcg.__version__ == "0.1.0"


def test_obs_import():
    """Test OBS module import."""
    import debcg
    assert hasattr(debcg.obs, "OBSConfig")
    assert hasattr(debcg.obs, "run")


def test_dmh_import():
    """Test DMH module import."""
    import debcg
    assert hasattr(debcg.dmh, "DMHConfig")
    assert hasattr(debcg.dmh, "run")


def test_obs_basic():
    """Test OBS with synthetic data."""
    import debcg
    
    np.random.seed(42)
    n_ch, n_samples = 3, 5000
    eeg = np.random.randn(n_ch, n_samples)
    ecg = np.random.randn(n_samples)
    
    # Add some artificial R-peaks
    for i in range(0, n_samples, 500):  # ~1 Hz heart rate
        if i < n_samples:
            ecg[i] = 5.0
    
    result = debcg.obs(eeg, ecg, sfreq=500.0)
    
    assert result.shape == eeg.shape
    assert np.isfinite(result).all()


def test_dmh_basic():
    """Test DMH with synthetic data."""
    import debcg
    
    np.random.seed(42)
    n_ch, n_samples = 3, 5000
    eeg = np.random.randn(n_ch, n_samples)
    ecg = np.random.randn(n_samples)
    
    # Add some artificial R-peaks
    for i in range(0, n_samples, 500):
        if i < n_samples:
            ecg[i] = 5.0
    
    result = debcg.dmh(eeg, ecg, sfreq=500.0)
    
    assert result.shape == eeg.shape
    assert np.isfinite(result).all()


def test_qrs_detection():
    """Test QRS detection."""
    import debcg
    
    np.random.seed(42)
    n_samples = 5000
    ecg = np.random.randn(n_samples) * 0.1
    
    # Add clear R-peaks
    for i in range(250, n_samples, 500):
        ecg[i] = 3.0
    
    peaks = debcg.qrs.qrs(ecg, 500.0)
    
    assert len(peaks) > 0
    assert all(isinstance(p, (int, np.integer)) for p in peaks)


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="torch not installed"),
    reason="torch not installed"
)
def test_bcgnet_import():
    """Test BCGNet module import (requires torch)."""
    import debcg
    assert hasattr(debcg.bcgnet, "BCGNetConfig")


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="torch not installed"),
    reason="torch not installed"
)
def test_brnet_import():
    """Test BRNet module import (requires torch)."""
    import debcg
    assert hasattr(debcg.brnet, "BRNetConfig")
