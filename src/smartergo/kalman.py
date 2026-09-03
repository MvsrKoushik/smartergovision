import numpy as np


class Kalman2D:
    """Constant-velocity 2D Kalman filter."""

    def __init__(self, process_noise: float = 1e-3, measurement_noise: float = 1e-2):
        self.state = np.zeros(4)
        self.covariance = np.eye(4)
        self.transition = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.process = np.eye(4) * process_noise
        self.measurement = np.eye(2) * measurement_noise
        self.initialized = False

    def update(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float)
        if not self.initialized:
            self.state[:2] = point
            self.initialized = True
        self.state = self.transition @ self.state
        self.covariance = self.transition @ self.covariance @ self.transition.T + self.process
        innovation = point - self.observation @ self.state
        innovation_covariance = self.observation @ self.covariance @ self.observation.T + self.measurement
        gain = self.covariance @ self.observation.T @ np.linalg.inv(innovation_covariance)
        self.state += gain @ innovation
        self.covariance = (np.eye(4) - gain @ self.observation) @ self.covariance
        return self.state[:2].copy()
