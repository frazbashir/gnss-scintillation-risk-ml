from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


def _sigmoid(z: float) -> float:
    if z <= -60.0:
        return 0.0
    if z >= 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class LogisticRegressionGD:
    learning_rate: float = 0.03
    epochs: int = 500
    l2: float = 0.0005

    def __post_init__(self) -> None:
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.means: List[float] = []
        self.stds: List[float] = []

    def _fit_standardization(self, x: List[List[float]]) -> List[List[float]]:
        n_features = len(x[0])
        self.means = []
        self.stds = []

        for j in range(n_features):
            col = [row[j] for row in x]
            mu = sum(col) / len(col)
            var = sum((v - mu) ** 2 for v in col) / len(col)
            std = math.sqrt(var) if var > 1e-12 else 1.0
            self.means.append(mu)
            self.stds.append(std)

        return [
            [(row[j] - self.means[j]) / self.stds[j] for j in range(n_features)]
            for row in x
        ]

    def _transform(self, x: List[List[float]]) -> List[List[float]]:
        if not self.means or not self.stds:
            raise RuntimeError("Model not fitted")

        n_features = len(self.means)
        return [
            [(row[j] - self.means[j]) / self.stds[j] for j in range(n_features)]
            for row in x
        ]

    def fit(self, x: List[List[float]], y: List[int]) -> None:
        if not x:
            raise ValueError("Empty training matrix")
        if len(x) != len(y):
            raise ValueError("x and y length mismatch")

        x_norm = self._fit_standardization(x)
        n_samples = len(x_norm)
        n_features = len(x_norm[0])

        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            grad_w = [0.0] * n_features
            grad_b = 0.0

            for i in range(n_samples):
                z = sum(self.weights[j] * x_norm[i][j] for j in range(n_features)) + self.bias
                p = _sigmoid(z)
                err = p - float(y[i])

                for j in range(n_features):
                    grad_w[j] += err * x_norm[i][j]
                grad_b += err

            for j in range(n_features):
                grad_w[j] = grad_w[j] / n_samples + self.l2 * self.weights[j]
                self.weights[j] -= self.learning_rate * grad_w[j]

            grad_b /= n_samples
            self.bias -= self.learning_rate * grad_b

    def predict_proba(self, x: List[List[float]]) -> List[float]:
        x_norm = self._transform(x)
        n_features = len(self.weights)
        probs: List[float] = []

        for row in x_norm:
            z = sum(self.weights[j] * row[j] for j in range(n_features)) + self.bias
            probs.append(_sigmoid(z))
        return probs

    def to_dict(self) -> Dict:
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
            "weights": self.weights,
            "bias": self.bias,
            "means": self.means,
            "stds": self.stds,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "LogisticRegressionGD":
        model = cls(
            learning_rate=float(payload["learning_rate"]),
            epochs=int(payload["epochs"]),
            l2=float(payload["l2"]),
        )
        model.weights = [float(v) for v in payload["weights"]]
        model.bias = float(payload["bias"])
        model.means = [float(v) for v in payload["means"]]
        model.stds = [float(v) for v in payload["stds"]]
        return model
