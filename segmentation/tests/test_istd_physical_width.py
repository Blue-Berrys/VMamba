import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluate_istd_physical_width.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("istd_physical_width", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_measure_transition_width_recovers_logistic_20_80_width():
    module = load_module()
    offsets = np.linspace(-16.0, 16.0, 257)
    scale = 2.5
    confidence = 1.0 / (1.0 + np.exp(offsets / scale))

    result = module.measure_transition_width(offsets, confidence)

    expected = 2.0 * scale * np.log(4.0)
    assert result is not None
    assert abs(result.width - expected) < 0.05
    assert result.monotonicity > 0.99


def test_normalize_attenuation_uses_local_inside_and_outside_levels():
    module = load_module()
    offsets = np.arange(-8.0, 9.0)
    physical = 1.0 / (1.0 + np.exp(offsets / 1.8))
    raw_attenuation = 0.07 + 0.32 * physical

    normalized, contrast = module.normalize_attenuation_profile(
        offsets, raw_attenuation, endpoint_radius=5.0)

    assert contrast > 0.25
    assert normalized[0] > 0.9
    assert normalized[-1] < 0.1
    assert np.corrcoef(normalized, physical)[0, 1] > 0.99


def test_measure_transition_width_rejects_flat_profile():
    module = load_module()
    offsets = np.arange(-8.0, 9.0)

    result = module.measure_transition_width(offsets, np.full_like(offsets, 0.5))

    assert result is None
