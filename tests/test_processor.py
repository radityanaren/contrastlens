# tests/test_processor.py

import numpy as np
import pytest

from contrastlens.config import ContrastLensConfig
from contrastlens.processor import ContrastLens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grey_bgr(value: int, h: int = 16, w: int = 16) -> np.ndarray:
    """Return a solid-colour BGR image where R == G == B == value."""
    return np.full((h, w, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# ContrastLensConfig
# ---------------------------------------------------------------------------


class TestContrastLensConfig:
    def test_low_mode_returns_required_keys(self):
        params = ContrastLensConfig.get_parameters(0.5, "low")
        assert "contrast_gain" in params
        assert "probability_gamma" in params

    def test_high_mode_returns_required_keys(self):
        params = ContrastLensConfig.get_parameters(0.5, "high")
        assert "contrast_gain" in params
        assert "probability_gamma" in params

    def test_high_mode_stronger_than_low(self):
        low = ContrastLensConfig.get_parameters(0.5, "low")
        high = ContrastLensConfig.get_parameters(0.5, "high")
        assert high["contrast_gain"] > low["contrast_gain"]

    def test_contrast_zero_gives_unity_gain(self):
        params = ContrastLensConfig.get_parameters(0.0, "low")
        assert params["contrast_gain"] == pytest.approx(1.0)

    def test_contrast_one_high_mode(self):
        params = ContrastLensConfig.get_parameters(1.0, "high")
        # 1.0 + 1.0 * 4.0 == 5.0
        assert params["contrast_gain"] == pytest.approx(5.0)

    def test_contrast_one_low_mode(self):
        params = ContrastLensConfig.get_parameters(1.0, "low")
        # 1.0 + 1.0 * 1.5 == 2.5
        assert params["contrast_gain"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# ContrastLens.rgb_to_luminance
# ---------------------------------------------------------------------------


class TestRgbToLuminance:
    def test_output_range_is_zero_to_one(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        lum = ContrastLens.rgb_to_luminance(img)
        assert lum.min() >= 0.0
        assert lum.max() <= 1.0

    def test_black_image_gives_zero_luminance(self):
        img = _make_grey_bgr(0)
        lum = ContrastLens.rgb_to_luminance(img)
        np.testing.assert_array_almost_equal(lum, 0.0)

    def test_white_image_gives_one_luminance(self):
        img = _make_grey_bgr(255)
        lum = ContrastLens.rgb_to_luminance(img)
        np.testing.assert_array_almost_equal(lum, 1.0)

    def test_output_shape_matches_input_spatial_dims(self):
        img = _make_grey_bgr(128, h=10, w=20)
        lum = ContrastLens.rgb_to_luminance(img)
        assert lum.shape == (10, 20)


# ---------------------------------------------------------------------------
# ContrastLens.adaptive_contrast
# ---------------------------------------------------------------------------


class TestAdaptiveContrast:
    def test_midpoint_is_invariant(self):
        mid = np.full((4, 4), 0.5, dtype=np.float32)
        result = ContrastLens.adaptive_contrast(mid, strength=3.0)
        np.testing.assert_array_almost_equal(result, 0.5)

    def test_output_clamped_between_zero_and_one(self):
        gray = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
        result = ContrastLens.adaptive_contrast(gray, strength=10.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_strength_one_is_identity(self):
        gray = np.linspace(0.01, 0.99, 100, dtype=np.float32).reshape(10, 10)
        result = ContrastLens.adaptive_contrast(gray, strength=1.0)
        np.testing.assert_array_almost_equal(result, gray)

    def test_higher_strength_increases_spread(self):
        gray = np.array([[0.3, 0.7]], dtype=np.float32)
        low = ContrastLens.adaptive_contrast(gray, strength=1.5)
        high = ContrastLens.adaptive_contrast(gray, strength=3.0)
        # spread = distance between the two pixels
        assert (high[0, 1] - high[0, 0]) >= (low[0, 1] - low[0, 0])


# ---------------------------------------------------------------------------
# ContrastLens.tone_to_probability
# ---------------------------------------------------------------------------


class TestToneToProbability:
    def _params(self, contrast=0.5, mode="low"):
        return ContrastLensConfig.get_parameters(contrast, mode)

    def test_output_range_is_zero_to_one(self):
        gray = np.random.rand(16, 16).astype(np.float32)
        prob = ContrastLens.tone_to_probability(gray, self._params())
        assert prob.min() >= 0.0
        assert prob.max() <= 1.0

    def test_near_white_pixels_have_zero_probability(self):
        gray = np.full((8, 8), 0.97, dtype=np.float32)
        prob = ContrastLens.tone_to_probability(gray, self._params())
        np.testing.assert_array_equal(prob, 0.0)

    def test_dark_pixels_have_higher_probability_than_light(self):
        dark = np.full((4, 4), 0.1, dtype=np.float32)
        light = np.full((4, 4), 0.8, dtype=np.float32)
        params = self._params()
        assert (
            ContrastLens.tone_to_probability(dark, params).mean()
            > ContrastLens.tone_to_probability(light, params).mean()
        )

    def test_output_shape_preserved(self):
        gray = np.random.rand(7, 13).astype(np.float32)
        prob = ContrastLens.tone_to_probability(gray, self._params())
        assert prob.shape == (7, 13)


# ---------------------------------------------------------------------------
# ContrastLens.stochastic_sampling
# ---------------------------------------------------------------------------


class TestStochasticSampling:
    def test_all_zero_probability_gives_all_zeros(self):
        prob = np.zeros((16, 16), dtype=np.float32)
        mask = ContrastLens.stochastic_sampling(prob)
        np.testing.assert_array_equal(mask, 0)

    def test_all_one_probability_gives_all_ones(self):
        prob = np.ones((16, 16), dtype=np.float32)
        mask = ContrastLens.stochastic_sampling(prob)
        np.testing.assert_array_equal(mask, 1)

    def test_output_is_binary(self):
        prob = np.random.rand(32, 32).astype(np.float32)
        mask = ContrastLens.stochastic_sampling(prob)
        unique = set(np.unique(mask).tolist())
        assert unique.issubset({0, 1})

    def test_output_shape_preserved(self):
        prob = np.random.rand(5, 9).astype(np.float32)
        mask = ContrastLens.stochastic_sampling(prob)
        assert mask.shape == (5, 9)


# ---------------------------------------------------------------------------
# ContrastLens.process  (integration)
# ---------------------------------------------------------------------------


class TestProcess:
    def test_output_shape_matches_input_spatial_dims(self):
        img = _make_grey_bgr(128, h=20, w=30)
        result = ContrastLens.process(img, contrast=0.5, mode="low")
        assert result.shape == (20, 30)

    def test_output_is_binary_black_or_white(self):
        img = _make_grey_bgr(100)
        result = ContrastLens.process(img, contrast=0.5, mode="low")
        unique = set(np.unique(result).tolist())
        assert unique.issubset({0, 255})

    def test_white_input_produces_mostly_white_output(self):
        img = _make_grey_bgr(255, h=64, w=64)
        result = ContrastLens.process(img, contrast=0.5, mode="low")
        white_ratio = (result == 255).mean()
        assert white_ratio > 0.95

    def test_black_input_produces_mostly_black_output(self):
        img = _make_grey_bgr(0, h=64, w=64)
        result = ContrastLens.process(img, contrast=0.5, mode="low")
        black_ratio = (result == 0).mean()
        assert black_ratio > 0.70

    def test_high_contrast_mode_accepted(self):
        img = _make_grey_bgr(128)
        result = ContrastLens.process(img, contrast=0.8, mode="high")
        assert result is not None

    @pytest.mark.parametrize("contrast", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_various_contrast_values_dont_raise(self, contrast):
        img = _make_grey_bgr(128)
        result = ContrastLens.process(img, contrast=contrast, mode="low")
        assert result.shape == img.shape[:2]
