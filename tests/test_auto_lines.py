"""Court line detection must work with OpenCV 4.x (N,1,4) and 5.x (N,4) HoughLinesP output."""

import cv2
import numpy as np
import pytest

from src.calibration import auto_lines
from src.calibration.auto_lines import detect_court_lines, intersection_candidates


def _court_image() -> np.ndarray:
    img = np.full((1080, 1920, 3), 60, np.uint8)
    cv2.line(img, (100, 900), (1800, 880), (255, 255, 255), 6)   # baseline
    cv2.line(img, (700, 900), (820, 400), (255, 255, 255), 6)    # paint side
    cv2.line(img, (1250, 890), (1150, 400), (255, 255, 255), 6)  # paint side
    cv2.line(img, (800, 420), (1170, 410), (255, 255, 255), 6)   # free-throw line
    return img


@pytest.mark.parametrize("layout", ["n14", "n4"])
def test_detect_court_lines_handles_both_houghlinesp_layouts(monkeypatch, layout):
    real = cv2.HoughLinesP

    def hough(*a, **k):
        out = real(*a, **k)
        if out is None:
            return None
        flat = np.asarray(out).reshape(-1, 4)
        return flat.reshape(-1, 1, 4) if layout == "n14" else flat

    monkeypatch.setattr(auto_lines.cv2, "HoughLinesP", hough)
    img = _court_image()
    segs = detect_court_lines(img)
    assert len(segs) >= 4
    assert len(intersection_candidates(segs, img.shape)) >= 2
