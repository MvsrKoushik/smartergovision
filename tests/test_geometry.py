import numpy as np

from smartergo import angle_degrees, confidence_weighted_fusion


def test_right_angle_and_fusion() -> None:
    assert angle_degrees(np.array([1, 0]), np.array([0, 0]), np.array([0, 1])) == 90.0
    fused = confidence_weighted_fusion([np.array([0, 0]), np.array([10, 0])], [0.9, 0.1])
    assert np.allclose(fused, [1, 0])
