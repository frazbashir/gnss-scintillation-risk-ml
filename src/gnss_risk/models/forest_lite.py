from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class DecisionStump:
    feature_index: int
    threshold: float
    left_prob: float
    right_prob: float

    def predict_proba_row(self, row: Sequence[float]) -> float:
        if row[self.feature_index] <= self.threshold:
            return self.left_prob
        return self.right_prob


class RandomForestLite:
    def __init__(
        self,
        n_estimators: int = 120,
        max_features: int = 5,
        max_threshold_candidates: int = 12,
        random_seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_threshold_candidates = max_threshold_candidates
        self.random_seed = random_seed
        self.stumps: List[DecisionStump] = []

    @staticmethod
    def _gini(y: List[int]) -> float:
        if not y:
            return 0.0
        p1 = sum(y) / len(y)
        p0 = 1.0 - p1
        return 1.0 - p0 * p0 - p1 * p1

    def _candidate_thresholds(self, values: List[float]) -> List[float]:
        uniq = sorted(set(values))
        if len(uniq) <= 1:
            return []

        mids = [(uniq[i] + uniq[i + 1]) / 2.0 for i in range(len(uniq) - 1)]
        if len(mids) <= self.max_threshold_candidates:
            return mids

        step = len(mids) / float(self.max_threshold_candidates)
        out: List[float] = []
        for i in range(self.max_threshold_candidates):
            out.append(mids[int(i * step)])
        return out

    def _fit_one_stump(
        self,
        x: List[List[float]],
        y: List[int],
        feature_subset: List[int],
    ) -> DecisionStump:
        best_score = math.inf
        best_stump: DecisionStump | None = None
        parent_prob = sum(y) / len(y) if y else 0.5

        for f_idx in feature_subset:
            col = [row[f_idx] for row in x]
            thresholds = self._candidate_thresholds(col)
            for thr in thresholds:
                left_y: List[int] = []
                right_y: List[int] = []
                for i, value in enumerate(col):
                    if value <= thr:
                        left_y.append(y[i])
                    else:
                        right_y.append(y[i])

                if not left_y or not right_y:
                    continue

                score = (len(left_y) * self._gini(left_y) + len(right_y) * self._gini(right_y)) / len(y)
                if score < best_score:
                    left_prob = sum(left_y) / len(left_y)
                    right_prob = sum(right_y) / len(right_y)
                    best_score = score
                    best_stump = DecisionStump(
                        feature_index=f_idx,
                        threshold=thr,
                        left_prob=left_prob,
                        right_prob=right_prob,
                    )

        if best_stump is not None:
            return best_stump

        return DecisionStump(feature_index=0, threshold=0.0, left_prob=parent_prob, right_prob=parent_prob)

    def fit(self, x: List[List[float]], y: List[int]) -> None:
        if not x:
            raise ValueError("Empty training matrix")
        if len(x) != len(y):
            raise ValueError("x and y length mismatch")

        rng = random.Random(self.random_seed)
        n_samples = len(x)
        n_features = len(x[0])
        k_features = max(1, min(self.max_features, n_features))

        self.stumps = []
        indices = list(range(n_samples))

        for _ in range(self.n_estimators):
            boot = [rng.choice(indices) for _ in range(n_samples)]
            x_boot = [x[i] for i in boot]
            y_boot = [y[i] for i in boot]

            feature_subset = rng.sample(list(range(n_features)), k_features)
            stump = self._fit_one_stump(x_boot, y_boot, feature_subset)
            self.stumps.append(stump)

    def predict_proba(self, x: List[List[float]]) -> List[float]:
        if not self.stumps:
            raise RuntimeError("Model not fitted")

        probs: List[float] = []
        for row in x:
            p = sum(stump.predict_proba_row(row) for stump in self.stumps) / len(self.stumps)
            probs.append(p)
        return probs

    def to_dict(self) -> Dict:
        return {
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "max_threshold_candidates": self.max_threshold_candidates,
            "random_seed": self.random_seed,
            "stumps": [
                {
                    "feature_index": stump.feature_index,
                    "threshold": stump.threshold,
                    "left_prob": stump.left_prob,
                    "right_prob": stump.right_prob,
                }
                for stump in self.stumps
            ],
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "RandomForestLite":
        model = cls(
            n_estimators=int(payload["n_estimators"]),
            max_features=int(payload["max_features"]),
            max_threshold_candidates=int(payload["max_threshold_candidates"]),
            random_seed=int(payload["random_seed"]),
        )
        model.stumps = [
            DecisionStump(
                feature_index=int(s["feature_index"]),
                threshold=float(s["threshold"]),
                left_prob=float(s["left_prob"]),
                right_prob=float(s["right_prob"]),
            )
            for s in payload.get("stumps", [])
        ]
        return model
