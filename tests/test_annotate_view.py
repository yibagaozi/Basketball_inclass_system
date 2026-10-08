"""Calibration annotate window: click ↔ image pixel mapping and aspect ratio (headless)."""

import numpy as np
import pytest

from src.calibration.annotate import _fit_to_view, _view_to_image

W, H = 1920, 1080
# window client sizes: default, 125% display scaling clamp, maximized, dragged narrow / flat
VIEWS = [(1600, 900), (1536, 834), (1920, 1017), (1200, 900), (1600, 640), (1920, 1080)]


@pytest.mark.parametrize("vw,vh", VIEWS)
def test_canvas_fills_window_and_keeps_aspect(vw, vh):
    canvas, (sx, sy, ox, oy) = _fit_to_view(np.full((H, W, 3), 128, np.uint8), vw, vh)
    assert canvas.shape == (vh, vw, 3)  # shown 1:1, nothing for HighGUI to stretch
    assert abs(sx - sy) <= 1.0 / min(W, H)
    ys, xs = np.nonzero(canvas[:, :, 0])
    assert (xs.min(), ys.min()) == (ox, oy)
    assert abs((xs.max() + 1 - ox) / (ys.max() + 1 - oy) - W / H) < 0.01


@pytest.mark.parametrize("vw,vh", VIEWS)
def test_click_maps_back_to_the_pixel_shown_under_it(vw, vh):
    _, view = _fit_to_view(np.zeros((H, W, 3), np.uint8), vw, vh)
    sx, sy, ox, oy = view
    for u in (0, 1, 333.3, 960, 1500.7, W - 1):
        for v in (0, 1, 271.9, 540, 1000.2, H - 1):
            x = int(np.floor(ox + (u + 0.5) * sx))  # canvas pixel showing image point (u, v)
            y = int(np.floor(oy + (v + 0.5) * sy))
            got = _view_to_image(x, y, view, W, H)
            assert got is not None
            assert abs(got[0] - u) <= 0.5 / sx + 1e-6 and abs(got[1] - v) <= 0.5 / sy + 1e-6


def test_backend_scaled_coords_are_not_scaled_again():
    # HighGUI already gives canvas coords; window centre must be the image centre
    _, view = _fit_to_view(np.zeros((H, W, 3), np.uint8), 1600, 900)
    u, v = _view_to_image(800, 450, view, W, H)
    assert abs(u - 960) < 1 and abs(v - 540) < 1  # 2.2.0 stored (1152, 648)


def test_clicks_on_letterbox_bars_are_ignored():
    _, view = _fit_to_view(np.zeros((H, W, 3), np.uint8), 1600, 640)  # bars left/right
    ox = view[2]
    assert ox > 0
    assert _view_to_image(ox - 2, 320, view, W, H) is None
    assert _view_to_image(1600 - ox + 1, 320, view, W, H) is None
    assert _view_to_image(ox + 1, 320, view, W, H) is not None
