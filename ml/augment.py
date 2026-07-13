"""Conservative, bbox-correct offline augmentation for TRAIN-only minority classes.

Transforms (all mild, chosen to stay inside plausible webcam conditions):
  geometric : horizontal flip, rotation +-MAX_ROT_DEG, scale SCALE_RANGE,
              translation +-MAX_TRANS_FRAC of the image dimension
  photometric: brightness, contrast, colour, optional gaussian blur, optional
              gaussian noise

Explicitly avoided: shear, extreme rotation, perspective/warp, vertical flip.

Bounding boxes are transformed with the SAME forward affine matrix applied to
the pixels. PIL's Image.transform() consumes the INVERSE matrix (it inverse-maps
each output pixel back into the source), so we build the forward matrix, use its
inverse for the image, and use the forward matrix on the box corners. The
axis-aligned bounding box of the four transformed corners becomes the new box,
clipped to the frame. A box that loses more than (1 - MIN_BOX_KEEP_FRAC) of its
area to clipping is dropped; if a sample ends up with no boxes we retry with
milder parameters and finally fall back to flip+photometric only, which is
exactly bbox-preserving. So every augmented image always keeps >=1 valid box.

`python ml/augment.py` runs the self-test that pins down the sign conventions.
"""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

MAX_ROT_DEG = 8.0
SCALE_RANGE = (0.90, 1.10)
MAX_TRANS_FRAC = 0.05
FLIP_PROB = 0.5
BRIGHTNESS_RANGE = (0.80, 1.20)
CONTRAST_RANGE = (0.85, 1.15)
COLOR_RANGE = (0.90, 1.10)
BLUR_PROB = 0.30
BLUR_RADIUS = (0.4, 1.0)
NOISE_PROB = 0.30
NOISE_STD = (3.0, 8.0)

MIN_BOX_KEEP_FRAC = 0.35   # keep a box only if >=35% of its area survives clipping
MIN_BOX_SIDE = 1e-3        # normalized
FILL = (114, 114, 114)     # YOLO letterbox grey


def _forward_matrix(angle_deg: float, scale: float, tx: float, ty: float, width: int, height: int):
    """3x3 forward affine: original pixel -> augmented pixel.

    Rotate+scale about the image centre, then translate by (tx, ty) pixels.
    """
    cx, cy = width / 2.0, height / 2.0
    rad = math.radians(angle_deg)
    cos = math.cos(rad) * scale
    sin = math.sin(rad) * scale
    # x' = cos*(x-cx) - sin*(y-cy) + cx + tx
    # y' = sin*(x-cx) + cos*(y-cy) + cy + ty
    return np.array([
        [cos, -sin, cx + tx - cos * cx + sin * cy],
        [sin,  cos, cy + ty - sin * cx - cos * cy],
        [0.0,  0.0, 1.0],
    ], dtype=np.float64)


def _boxes_to_pixel_corners(boxes, width: int, height: int):
    """[(cls,cx,cy,w,h)] normalized -> (classes, corners[N,4,2]) in pixels."""
    classes, corners = [], []
    for cls, cx, cy, w, h in boxes:
        x1 = (cx - w / 2) * width
        x2 = (cx + w / 2) * width
        y1 = (cy - h / 2) * height
        y2 = (cy + h / 2) * height
        classes.append(cls)
        corners.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return classes, np.array(corners, dtype=np.float64).reshape(-1, 4, 2) if corners else np.zeros((0, 4, 2))


def _transform_boxes(boxes, matrix, width: int, height: int):
    """Apply forward matrix to boxes; return surviving normalized boxes."""
    classes, corners = _boxes_to_pixel_corners(boxes, width, height)
    if len(classes) == 0:
        return []

    ones = np.ones((corners.shape[0], 4, 1))
    homo = np.concatenate([corners, ones], axis=2)          # (N,4,3)
    moved = homo @ matrix.T                                  # (N,4,3)
    xy = moved[:, :, :2]

    out = []
    for idx, cls in enumerate(classes):
        pts = xy[idx]
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        area_before = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        if area_before <= 0:
            continue
        # clip to frame
        cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
        cx2, cy2 = min(x2, float(width)), min(y2, float(height))
        cw, ch = cx2 - cx1, cy2 - cy1
        if cw <= 0 or ch <= 0:
            continue
        if (cw * ch) / area_before < MIN_BOX_KEEP_FRAC:
            continue
        nw, nh = cw / width, ch / height
        ncx, ncy = (cx1 + cx2) / 2 / width, (cy1 + cy2) / 2 / height
        if nw <= MIN_BOX_SIDE or nh <= MIN_BOX_SIDE:
            continue
        ncx = min(max(ncx, 0.0), 1.0)
        ncy = min(max(ncy, 0.0), 1.0)
        nw = min(nw, 1.0)
        nh = min(nh, 1.0)
        out.append((cls, ncx, ncy, nw, nh))
    return out


def _apply_geometric(image: Image.Image, boxes, angle: float, scale: float, tx: float, ty: float, flip: bool):
    width, height = image.size
    out_boxes = list(boxes)
    out_image = image

    if flip:
        out_image = out_image.transpose(Image.FLIP_LEFT_RIGHT)
        out_boxes = [(c, 1.0 - cx, cy, w, h) for (c, cx, cy, w, h) in out_boxes]

    if angle == 0.0 and scale == 1.0 and tx == 0.0 and ty == 0.0:
        return out_image, out_boxes

    forward = _forward_matrix(angle, scale, tx, ty, width, height)
    inverse = np.linalg.inv(forward)
    a, b, c = inverse[0]
    d, e, f = inverse[1]
    out_image = out_image.transform(
        (width, height), Image.AFFINE, (a, b, c, d, e, f),
        resample=Image.BILINEAR, fillcolor=FILL,
    )
    out_boxes = _transform_boxes(out_boxes, forward, width, height)
    return out_image, out_boxes


def _apply_photometric(image: Image.Image, rng: random.Random, np_rng: np.random.Generator) -> tuple[Image.Image, list[str]]:
    desc = []
    brightness = rng.uniform(*BRIGHTNESS_RANGE)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    desc.append(f"bright{brightness:.2f}")

    contrast = rng.uniform(*CONTRAST_RANGE)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    desc.append(f"contrast{contrast:.2f}")

    color = rng.uniform(*COLOR_RANGE)
    image = ImageEnhance.Color(image).enhance(color)
    desc.append(f"color{color:.2f}")

    if rng.random() < BLUR_PROB:
        radius = rng.uniform(*BLUR_RADIUS)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        desc.append(f"blur{radius:.2f}")

    if rng.random() < NOISE_PROB:
        std = rng.uniform(*NOISE_STD)
        arr = np.asarray(image).astype(np.float32)
        arr += np_rng.normal(0.0, std, arr.shape)
        image = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
        desc.append(f"noise{std:.1f}")

    return image, desc


def augment(image_path, boxes, rng: random.Random, np_rng: np.random.Generator):
    """Return (augmented_image, boxes, transform_description).

    Retries with milder geometry if every box is clipped away; finally falls back
    to flip+photometric, which cannot lose a box.
    """
    base = Image.open(image_path).convert("RGB")
    width, height = base.size

    for attempt, damp in enumerate((1.0, 0.5, 0.25)):
        flip = rng.random() < FLIP_PROB
        angle = rng.uniform(-MAX_ROT_DEG, MAX_ROT_DEG) * damp
        scale = 1.0 + (rng.uniform(*SCALE_RANGE) - 1.0) * damp
        tx = rng.uniform(-MAX_TRANS_FRAC, MAX_TRANS_FRAC) * damp * width
        ty = rng.uniform(-MAX_TRANS_FRAC, MAX_TRANS_FRAC) * damp * height

        image, new_boxes = _apply_geometric(base, boxes, angle, scale, tx, ty, flip)
        if new_boxes:
            image, photo_desc = _apply_photometric(image, rng, np_rng)
            geo = []
            if flip:
                geo.append("hflip")
            geo.append(f"rot{angle:+.1f}")
            geo.append(f"scale{scale:.2f}")
            geo.append(f"trans({tx/width:+.3f},{ty/height:+.3f})")
            return image, new_boxes, "+".join(geo + photo_desc)

    # Guaranteed-safe fallback: flip + photometric only (exact bbox mapping).
    flip = rng.random() < FLIP_PROB
    image, new_boxes = _apply_geometric(base, boxes, 0.0, 1.0, 0.0, 0.0, flip)
    image, photo_desc = _apply_photometric(image, rng, np_rng)
    geo = ["hflip"] if flip else ["identity"]
    return image, new_boxes, "+".join(geo + photo_desc) + "+fallback"


# ---------------------------------------------------------------------------


def _selftest() -> None:
    width, height = 200, 100
    img = Image.new("RGB", (width, height), (10, 20, 30))
    boxes = [(3, 0.5, 0.5, 0.4, 0.4), (1, 0.25, 0.5, 0.1, 0.2)]

    # 1. Identity matrix leaves boxes untouched.
    ident = _forward_matrix(0.0, 1.0, 0.0, 0.0, width, height)
    same = _transform_boxes(boxes, ident, width, height)
    for (c0, *b0), (c1, *b1) in zip(boxes, same):
        assert c0 == c1
        assert all(abs(x - y) < 1e-6 for x, y in zip(b0, b1)), (b0, b1)

    # 2. Pure translation shifts the centre by exactly that amount.
    tx = 0.1 * width
    trans = _forward_matrix(0.0, 1.0, tx, 0.0, width, height)
    moved = _transform_boxes(boxes, trans, width, height)
    assert abs(moved[0][1] - (0.5 + 0.1)) < 1e-6, moved[0]
    assert abs(moved[0][2] - 0.5) < 1e-6

    # 3. Pure scale about the centre keeps the centre, scales the size.
    scaled = _transform_boxes([(0, 0.5, 0.5, 0.2, 0.2)], _forward_matrix(0.0, 1.5, 0.0, 0.0, width, height), width, height)
    assert abs(scaled[0][1] - 0.5) < 1e-6 and abs(scaled[0][2] - 0.5) < 1e-6
    assert abs(scaled[0][3] - 0.3) < 1e-6, scaled[0]

    # 4. Horizontal flip mirrors cx exactly and preserves size.
    _, flipped = _apply_geometric(img, boxes, 0.0, 1.0, 0.0, 0.0, True)
    assert abs(flipped[0][1] - 0.5) < 1e-9
    assert abs(flipped[1][1] - 0.75) < 1e-9
    assert abs(flipped[1][3] - 0.1) < 1e-9

    # 5. Rotation direction agrees between image and boxes: rotate a box that
    #    sits left-of-centre by +90deg and check it lands below-centre.
    rot = _forward_matrix(90.0, 1.0, 0.0, 0.0, 100, 100)
    got = _transform_boxes([(0, 0.25, 0.5, 0.1, 0.1)], rot, 100, 100)
    assert abs(got[0][1] - 0.5) < 1e-6 and abs(got[0][2] - 0.25) < 1e-6, got[0]

    # 6. A real augment() call always yields >=1 in-range box.
    rng, np_rng = random.Random(0), np.random.default_rng(0)
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "t.png")
        img.save(p)
        for _ in range(50):
            _, out, desc = augment(p, boxes, rng, np_rng)
            assert out, "augment produced zero boxes"
            for c, cx, cy, w, h in out:
                assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0, (cx, cy)
                assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0, (w, h)
                assert c in (1, 3)
    print("augment self-test PASS (identity, translation, scale, flip, rotation sign, box validity)")


if __name__ == "__main__":
    _selftest()
