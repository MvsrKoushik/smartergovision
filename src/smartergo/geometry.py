import numpy as np


def confidence_weighted_fusion(points: list[np.ndarray], confidences: list[float]) -> np.ndarray:
    if len(points) != len(confidences) or not points:
        raise ValueError("points and confidences must be non-empty and aligned")
    weights = np.clip(np.asarray(confidences, dtype=float), 0, None)
    if weights.sum() == 0:
        raise ValueError("at least one confidence must be positive")
    return np.average(np.asarray(points, dtype=float), axis=0, weights=weights)


def angle_degrees(first: np.ndarray, vertex: np.ndarray, third: np.ndarray) -> float:
    left, right = np.asarray(first) - np.asarray(vertex), np.asarray(third) - np.asarray(vertex)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        raise ValueError("angle points must be distinct")
    cosine = np.clip(np.dot(left, right) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))
