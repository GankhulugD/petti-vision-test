"""Edge refinement and outline smoothing (numpy only)."""

import numpy as np
from PIL import Image, ImageDraw

from petti_vision.segmenter import _box, guided_filter, instances_refined, instances_v2, smooth_outline


def _scene():
    W, H = 640, 480
    im = Image.new("RGB", (W, H), (40, 70, 110))
    d = ImageDraw.Draw(im)
    g = Image.new("L", (W, H))
    dg = ImageDraw.Draw(g)
    for box in ((150, 120, 470, 380), (400, 90, 520, 200)):
        d.ellipse(box, fill=(220, 200, 180))
        dg.ellipse(box, fill=1)
    gt = np.asarray(g) > 0
    prob = np.clip(np.asarray(Image.fromarray(gt.astype(np.float32)).resize((32, 24), Image.BOX)) * 0.9 + 0.05, 0, 1)
    return im, gt, prob


def _iou(det, gt):
    m = det.mask(gt.shape[::-1])
    return (m & gt).sum() / (m | gt).sum()


def test_box_filter_matches_window_mean():
    x = np.random.default_rng(0).random((40, 50, 3)).astype(np.float32)
    assert np.allclose(_box(x, 2)[10, 10], x[8:13, 8:13].mean((0, 1)), atol=1e-5)


def test_guided_filter_preserves_constant():
    g = np.random.default_rng(1).random((30, 30, 3)).astype(np.float32)
    assert np.allclose(guided_filter(g, np.full((30, 30), 0.7, np.float32), 3, 1e-3), 0.7, atol=1e-3)


def test_refinement_beats_plain_upsampling_and_outline_is_smooth():
    im, gt, prob = _scene()
    plain = instances_v2(prob, im.size, 0.5)[0]
    ref = instances_refined(prob, im, 0.5, r=2)[0]
    assert _iou(ref, gt) > _iou(plain, gt) > 0.9
    poly = np.asarray(ref.polygon)
    assert 20 <= len(poly) <= 160
    # no long straight jumps: consecutive points are close (rounded, not blocky)
    steps = np.hypot(*np.diff(np.vstack([poly, poly[:1]]), axis=0).T)
    assert steps.max() < 0.15 * max(im.size)


def test_smooth_outline_on_closed_shape():
    m = np.zeros((192, 256), bool)
    Image.fromarray(m)
    g = Image.new("L", (256, 192))
    ImageDraw.Draw(g).ellipse((60, 48, 188, 152), fill=1)
    m = np.asarray(g) > 0
    pts = smooth_outline(m)
    pm = Image.new("L", (256, 192))
    ImageDraw.Draw(pm).polygon(pts, fill=1)
    pm = np.asarray(pm) > 0
    assert (pm & m).sum() / (pm | m).sum() > 0.97
