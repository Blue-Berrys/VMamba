import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "build_sbu_soft_split.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("sbu_soft_split", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weighted_quantile_respects_pixel_reliability():
    module = load_module()
    values = np.array([1.0, 4.0, 8.0])
    weights = np.array([8.0, 1.0, 1.0])

    result = module.weighted_quantile(values, weights, 0.5)

    assert result == 1.0


def test_select_extreme_groups_is_deterministic_and_disjoint():
    module = load_module()
    rows = [{"sample": f"s{i}", "softness_score": float(i)} for i in range(10)]

    groups = module.select_extreme_groups(rows, fraction=0.2)

    assert [row["sample"] for row in groups["hard"]] == ["s0", "s1"]
    assert [row["sample"] for row in groups["soft"]] == ["s9", "s8"]
    assert not ({row["sample"] for row in groups["hard"]} &
                {row["sample"] for row in groups["soft"]})


def test_measure_luminance_transition_recovers_unclipped_width():
    module = load_module()
    offsets = np.linspace(-24.0, 24.0, 385)
    scale = 4.0
    shadow_confidence = 1.0 / (1.0 + np.exp(offsets / scale))
    luminance = 0.2 + 0.5 * (1.0 - shadow_confidence)

    result = module.measure_luminance_transition(
        offsets, luminance, endpoint_radius=16.0)

    assert result is not None
    assert abs(result["width"] - 2.0 * scale * np.log(4.0)) < 0.2
    assert result["contrast"] > 0.45
