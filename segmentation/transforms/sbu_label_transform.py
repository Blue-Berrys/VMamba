"""
SBU label transforms.

SBU uses binary 0/255 masks. SBU-Refine additionally provides 0-255 soft
masks, where intermediate values carry boundary confidence. For penumbra
training, keep the binary target for the main detector and expose the soft mask
as a separate PixelData field.
"""

import numpy as np
from mmcv.transforms import to_tensor
from mmengine.structures import PixelData
from mmseg.datasets.transforms import LoadAnnotations, PackSegInputs
from mmseg.registry import TRANSFORMS


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _safe_image_luma(results: dict, shape) -> np.ndarray:
    img = results.get("img")
    if not isinstance(img, np.ndarray):
        return np.zeros(shape, dtype=np.float32)

    img = img.astype(np.float32)
    if img.ndim == 2:
        luma = img
    else:
        # Use channel mean to avoid depending on RGB/BGR ordering.
        luma = img[..., :3].mean(axis=2)
    if luma.max() > 1.5:
        luma = luma / 255.0
    return np.clip(luma, 0.0, 1.0).astype(np.float32)


def _signed_distance(mask: np.ndarray, band_width: int) -> np.ndarray:
    mask = mask.astype(np.uint8)
    if mask.max() == 0:
        return np.full(mask.shape, -10.0 * band_width, dtype=np.float32)
    if mask.min() == 1:
        return np.full(mask.shape, 10.0 * band_width, dtype=np.float32)

    try:
        import cv2
        dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    except Exception:
        from scipy import ndimage
        dist_in = ndimage.distance_transform_edt(mask)
        dist_out = ndimage.distance_transform_edt(1 - mask)
    return (dist_in - dist_out).astype(np.float32)


def _transition_score(luma: np.ndarray,
                      sigma: float,
                      illum_tau: float,
                      grad_percentile: float) -> np.ndarray:
    try:
        import cv2
        local_ref = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
        gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
    except Exception:
        local_ref = luma
        gy, gx = np.gradient(luma)
        grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    denom = np.percentile(grad, grad_percentile)
    if denom < 1e-6:
        edge_score = np.zeros_like(grad, dtype=np.float32)
    else:
        edge_score = np.clip(grad / denom, 0.0, 1.0).astype(np.float32)

    dark_score = _sigmoid_np((local_ref - luma) / max(illum_tau, 1e-6))
    return np.sqrt(np.clip(edge_score * dark_score, 0.0, 1.0)).astype(np.float32)


@TRANSFORMS.register_module(force=True)
class SBULabelTransform(LoadAnnotations):
    """Convert SBU labels from 0/255 or non-zero values to binary 0/1."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, results):
        results = super().transform(results)
        if "gt_seg_map" in results:
            seg_map = results["gt_seg_map"]
            if isinstance(seg_map, np.ndarray):
                results["gt_seg_map"] = (seg_map > 0).astype(np.uint8)
        return results


@TRANSFORMS.register_module(force=True)
class RefineAnnTransform(LoadAnnotations):
    """Load SBU-Refine annotations as both binary and soft targets.

    SBU-Refine masks are soft labels in [0, 255]. The binary detector still uses
    a thresholded mask, while the penumbra confidence head can regress the
    original continuous mask through gt_soft_seg_map.
    """

    def transform(self, results):
        results = super().transform(results)
        if "gt_seg_map" in results:
            seg_map = results["gt_seg_map"]
            if isinstance(seg_map, np.ndarray):
                soft_map = seg_map.astype(np.float32) / 255.0
                results["gt_soft_seg_map"] = np.clip(soft_map, 0.0, 1.0)
                if "seg_fields" in results and "gt_soft_seg_map" not in results["seg_fields"]:
                    results["seg_fields"].append("gt_soft_seg_map")
                results["gt_seg_map"] = (seg_map >= 128).astype(np.uint8)
        return results


@TRANSFORMS.register_module(force=True)
class RefineIlluminationSoftAnnTransform(LoadAnnotations):
    """Load SBU-Refine labels and correct soft penumbra targets with image cues.

    The binary detector target remains thresholded. The auxiliary soft target is
    only adjusted around the signed-distance boundary band: outer-side soft
    labels without local darkening/gradient evidence are suppressed to reduce
    dark-object false positives, while inner-side labels are kept conservative
    to avoid recall loss.
    """

    def __init__(self,
                 band_width: int = 4,
                 tau: float = 2.0,
                 distance_weight: float = 0.15,
                 outer_suppress: float = 0.35,
                 inner_distance_weight: float = 0.0,
                 reliability_min: float = 0.35,
                 reliability_power: float = 1.0,
                 illum_sigma: float = 3.0,
                 illum_tau: float = 0.08,
                 grad_percentile: float = 90.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.band_width = band_width
        self.tau = tau
        self.distance_weight = distance_weight
        self.outer_suppress = outer_suppress
        self.inner_distance_weight = inner_distance_weight
        self.reliability_min = reliability_min
        self.reliability_power = reliability_power
        self.illum_sigma = illum_sigma
        self.illum_tau = illum_tau
        self.grad_percentile = grad_percentile

    def transform(self, results):
        results = super().transform(results)
        if "gt_seg_map" not in results:
            return results

        seg_map = results["gt_seg_map"]
        if not isinstance(seg_map, np.ndarray):
            return results

        refine_soft = np.clip(seg_map.astype(np.float32) / 255.0, 0.0, 1.0)
        binary = (seg_map >= 128).astype(np.uint8)
        signed = _signed_distance(binary, max(int(self.band_width), 1))
        dist_soft = _sigmoid_np(signed / max(self.tau, 1e-6)).astype(np.float32)
        band = np.abs(signed) <= float(self.band_width)

        luma = _safe_image_luma(results, refine_soft.shape)
        trans = _transition_score(
            luma,
            sigma=self.illum_sigma,
            illum_tau=self.illum_tau,
            grad_percentile=self.grad_percentile,
        )

        soft_map = refine_soft.copy()
        soft_weight = np.ones_like(soft_map, dtype=np.float32)
        outer = band & (signed <= 0)
        inner = band & (signed > 0)

        if outer.any():
            base_outer = (
                (1.0 - self.distance_weight) * soft_map[outer] +
                self.distance_weight * dist_soft[outer]
            )
            suppress = 1.0 - self.outer_suppress * (1.0 - trans[outer])
            soft_map[outer] = base_outer * np.clip(suppress, 0.0, 1.0)

        if band.any():
            reliability = np.clip(trans, 0.0, 1.0)
            if self.reliability_power != 1.0:
                reliability = np.power(reliability, self.reliability_power)
            soft_weight[band] = (
                self.reliability_min +
                (1.0 - self.reliability_min) * reliability[band]
            )

        if inner.any() and self.inner_distance_weight > 0:
            soft_map[inner] = np.maximum(
                soft_map[inner],
                ((1.0 - self.inner_distance_weight) * soft_map[inner] +
                 self.inner_distance_weight * dist_soft[inner]),
            )

        results["gt_soft_seg_map"] = np.clip(soft_map, 0.0, 1.0).astype(np.float32)
        results["gt_soft_weight_map"] = np.clip(soft_weight, 0.0, 1.0).astype(np.float32)
        if "seg_fields" in results and "gt_soft_seg_map" not in results["seg_fields"]:
            results["seg_fields"].append("gt_soft_seg_map")
        if "seg_fields" in results and "gt_soft_weight_map" not in results["seg_fields"]:
            results["seg_fields"].append("gt_soft_weight_map")
        results["gt_seg_map"] = binary
        return results


@TRANSFORMS.register_module(force=True)
class RefineAdaptivePenumbraAnnTransform(LoadAnnotations):
    """Build local penumbra-width supervision from RGB and binary masks.

    The binary annotation locates the contour. Normalized luminance profiles
    sampled along signed-distance normals estimate local 20--80% transition
    widths. Low-contrast or non-monotonic profiles receive low reliability.
    """

    def __init__(self,
                 band_width: int = 12,
                 min_width: float = 1.0,
                 max_width: float = 16.0,
                 reliability_min: float = 0.05,
                 contrast_tau: float = 0.12,
                 **kwargs):
        super().__init__(**kwargs)
        self.band_width = int(band_width)
        self.min_width = float(min_width)
        self.max_width = float(max_width)
        self.reliability_min = float(reliability_min)
        self.contrast_tau = float(contrast_tau)

    def _estimate_profiles(self, luma, signed):
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError(
                "RefineAdaptivePenumbraAnnTransform requires OpenCV") from exc

        smooth = cv2.GaussianBlur(luma, (0, 0), sigmaX=0.8, sigmaY=0.8)
        gy, gx = np.gradient(signed.astype(np.float32))
        norm = np.sqrt(gx * gx + gy * gy).clip(1e-6)
        nx, ny = gx / norm, gy / norm
        yy, xx = np.indices(luma.shape, dtype=np.float32)
        radius = max(self.band_width, int(np.ceil(self.max_width)))
        offsets = np.arange(-radius, radius + 1, dtype=np.float32)
        profiles = []
        for offset in offsets:
            profiles.append(cv2.remap(
                smooth,
                xx + offset * nx,
                yy + offset * ny,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            ))
        profiles = np.stack(profiles, axis=0)

        # Positive signed distance points toward the shadow interior.
        lit_level = profiles[:3].mean(axis=0)
        shadow_level = profiles[-3:].mean(axis=0)
        contrast = lit_level - shadow_level
        valid_contrast = contrast > 1e-4
        normalized = ((lit_level[None] - profiles) /
                      np.maximum(contrast[None], 1e-4))
        normalized = np.clip(normalized, 0.0, 1.0)
        monotonic = np.maximum.accumulate(normalized, axis=0)

        above20 = monotonic >= 0.2
        above80 = monotonic >= 0.8
        idx20 = above20.argmax(axis=0)
        idx80 = above80.argmax(axis=0)
        has_crossings = above20.any(axis=0) & above80.any(axis=0)
        width = (idx80 - idx20).astype(np.float32)
        width = np.clip(width, self.min_width, self.max_width)

        negative_steps = np.maximum(
            normalized[:-1] - normalized[1:], 0.0).mean(axis=0)
        monotonicity = np.exp(-8.0 * negative_steps)
        contrast_score = np.clip(
            contrast / max(self.contrast_tau, 1e-6), 0.0, 1.0)
        reliability = contrast_score * monotonicity
        reliability *= (valid_contrast & has_crossings).astype(np.float32)
        return width, np.clip(reliability, 0.0, 1.0)

    def transform(self, results):
        results = super().transform(results)
        seg_map = results.get("gt_seg_map")
        if not isinstance(seg_map, np.ndarray):
            return results

        binary = (seg_map >= 128).astype(np.uint8)
        signed = _signed_distance(binary, max(self.band_width, 1))
        band = np.abs(signed) <= float(self.band_width)
        luma = _safe_image_luma(results, binary.shape)
        width, reliability = self._estimate_profiles(luma, signed)
        width_norm = np.clip(
            (width - self.min_width) /
            max(self.max_width - self.min_width, 1e-6), 0.0, 1.0)

        # w20-80 = 2*ln(4)*tau for a logistic transition.
        tau_map = width / (2.0 * np.log(4.0))
        adaptive_soft = _sigmoid_np(signed / np.maximum(tau_map, 1e-3))
        soft_map = binary.astype(np.float32)
        soft_map[band] = adaptive_soft[band]
        adaptive_support = band & (
            np.abs(signed) <= np.maximum(2.0, 0.75 * width))
        soft_weight = np.zeros_like(soft_map, dtype=np.float32)
        soft_weight[adaptive_support] = (
            self.reliability_min +
            (1.0 - self.reliability_min) * reliability[adaptive_support]
        )

        results["gt_soft_seg_map"] = soft_map.astype(np.float32)
        results["gt_soft_weight_map"] = soft_weight.astype(np.float32)
        results["gt_penumbra_width_map"] = width_norm.astype(np.float32)
        for key in ("gt_soft_seg_map", "gt_soft_weight_map",
                    "gt_penumbra_width_map"):
            if "seg_fields" in results and key not in results["seg_fields"]:
                results["seg_fields"].append(key)
        results["gt_seg_map"] = binary
        return results


@TRANSFORMS.register_module(force=True)
class RefineHardNegativeAnnTransform(RefineAnnTransform):
    """Load SBU-Refine labels and add dark non-shadow hard-negative weights.

    The main detector still receives a binary mask and the penumbra branch keeps
    the original soft SBU-Refine target.  The extra `gt_dark_neg_map` marks dark
    non-shadow pixels outside the boundary band, providing a supervised handle
    for dark-object false positives.
    """

    def __init__(self,
                 boundary_exclude: int = 6,
                 dark_percentile: float = 20.0,
                 illum_sigma: float = 5.0,
                 illum_tau: float = 0.08,
                 dark_score_threshold: float = 0.45,
                 **kwargs):
        super().__init__(**kwargs)
        self.boundary_exclude = boundary_exclude
        self.dark_percentile = dark_percentile
        self.illum_sigma = illum_sigma
        self.illum_tau = illum_tau
        self.dark_score_threshold = dark_score_threshold

    def _dark_score(self, luma: np.ndarray) -> np.ndarray:
        try:
            import cv2
            local_ref = cv2.GaussianBlur(
                luma, (0, 0), sigmaX=self.illum_sigma,
                sigmaY=self.illum_sigma)
        except Exception:
            local_ref = luma
        return _sigmoid_np((local_ref - luma) / max(self.illum_tau, 1e-6)).astype(np.float32)

    def transform(self, results):
        results = super().transform(results)
        if "gt_seg_map" not in results:
            return results

        binary = results["gt_seg_map"]
        if not isinstance(binary, np.ndarray):
            return results

        binary = binary.astype(np.uint8)
        signed = _signed_distance(binary, max(int(self.boundary_exclude), 1))
        non_shadow_far = (binary == 0) & (signed < -float(self.boundary_exclude))

        luma = _safe_image_luma(results, binary.shape)
        hard_neg = np.zeros(binary.shape, dtype=np.float32)
        if non_shadow_far.any():
            dark_pool = luma[non_shadow_far]
            dark_thr = np.percentile(dark_pool, self.dark_percentile)
            dark_score = self._dark_score(luma)
            candidate = (
                non_shadow_far &
                ((luma <= dark_thr) |
                 (dark_score >= float(self.dark_score_threshold)))
            )
            hard_neg[candidate] = dark_score[candidate]

        results["gt_dark_neg_map"] = np.clip(hard_neg, 0.0, 1.0).astype(np.float32)
        if "seg_fields" in results and "gt_dark_neg_map" not in results["seg_fields"]:
            results["seg_fields"].append("gt_dark_neg_map")
        return results


@TRANSFORMS.register_module(force=True)
class PackSegInputsWithSoft(PackSegInputs):
    """Pack segmentation inputs and optional SBU-Refine soft mask."""

    def transform(self, results: dict) -> dict:
        packed_results = super().transform(results)
        if ("gt_soft_seg_map" not in results and
                "gt_soft_weight_map" not in results and
                "gt_penumbra_width_map" not in results and
                "gt_dark_neg_map" not in results):
            return packed_results

        extra_data = {}
        if "gt_soft_seg_map" in results:
            soft_map = results["gt_soft_seg_map"]
            if len(soft_map.shape) == 2:
                data = to_tensor(soft_map[None, ...].astype(np.float32))
            else:
                data = to_tensor(soft_map.astype(np.float32))
            extra_data["gt_soft_seg"] = PixelData(data=data)

        if "gt_soft_weight_map" in results:
            soft_weight = results["gt_soft_weight_map"]
            if len(soft_weight.shape) == 2:
                data = to_tensor(soft_weight[None, ...].astype(np.float32))
            else:
                data = to_tensor(soft_weight.astype(np.float32))
            extra_data["gt_soft_weight"] = PixelData(data=data)

        if "gt_penumbra_width_map" in results:
            width_map = results["gt_penumbra_width_map"]
            if len(width_map.shape) == 2:
                data = to_tensor(width_map[None, ...].astype(np.float32))
            else:
                data = to_tensor(width_map.astype(np.float32))
            extra_data["gt_penumbra_width"] = PixelData(data=data)

        if "gt_dark_neg_map" in results:
            hard_neg_map = results["gt_dark_neg_map"]
            if len(hard_neg_map.shape) == 2:
                data = to_tensor(hard_neg_map[None, ...].astype(np.float32))
            else:
                data = to_tensor(hard_neg_map.astype(np.float32))
            extra_data["gt_dark_neg"] = PixelData(data=data)

        packed_results["data_samples"].set_data(extra_data)
        return packed_results
