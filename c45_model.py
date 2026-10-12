import numpy as np


def entropy(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def majority_class(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


class C45DecisionTree:
    def __init__(self, max_depth=6, min_samples_leaf=15, min_gain=1e-4):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.tree_ = None
        self.feature_names_ = None

    def _candidate_thresholds(self, x, y):
        order = np.argsort(x)
        xs = np.asarray(x, dtype=float)[order]
        ys = np.asarray(y)[order]
        candidates = []
        for i in range(1, len(xs)):
            if xs[i] == xs[i - 1]:
                continue
            if ys[i] != ys[i - 1]:
                candidates.append((xs[i] + xs[i - 1]) / 2)
        return candidates

    def _split_score(self, y, left_mask):
        n = len(y)
        n_left = int(left_mask.sum())
        n_right = n - n_left
        if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
            return None
        parent_entropy = entropy(y)
        left_y, right_y = y[left_mask], y[~left_mask]
        gain = parent_entropy - (
            len(left_y) / n * entropy(left_y)
            + len(right_y) / n * entropy(right_y)
        )
        p_left, p_right = n_left / n, n_right / n
        split_info = -(p_left * np.log2(p_left) + p_right * np.log2(p_right))
        gain_ratio = gain / split_info if split_info > 0 else 0.0
        return gain, gain_ratio

    def _best_split(self, X, y):
        candidates = []
        for feature in X.columns:
            values = X[feature].to_numpy(dtype=float)
            for threshold in self._candidate_thresholds(values, y):
                mask = values <= threshold
                score = self._split_score(y, mask)
                if score is not None:
                    gain, ratio = score
                    if gain > 0:
                        candidates.append(
                            {
                                "feature": feature,
                                "threshold": float(threshold),
                                "gain": float(gain),
                                "gain_ratio": float(ratio),
                            }
                        )

        if not candidates:
            return None

        avg_gain = np.mean([c["gain"] for c in candidates])
        eligible = [c for c in candidates if c["gain"] >= avg_gain]
        return max(eligible or candidates, key=lambda c: c["gain_ratio"])

    def _build(self, X, y, depth):
        leaf = {"type": "leaf", "prediction": majority_class(y), "samples": len(y)}
        if (
            len(np.unique(y)) == 1
            or depth >= self.max_depth
            or len(y) < 2 * self.min_samples_leaf
        ):
            return leaf

        split = self._best_split(X, y)
        if split is None or split["gain"] < self.min_gain:
            return leaf

        feature, threshold = split["feature"], split["threshold"]
        mask = X[feature].to_numpy(dtype=float) <= threshold
        if mask.sum() == 0 or (~mask).sum() == 0:
            return leaf

        return {
            "type": "split",
            "feature": feature,
            "threshold": threshold,
            "prediction": leaf["prediction"],
            "samples": len(y),
            "gain": split["gain"],
            "gain_ratio": split["gain_ratio"],
            "left": self._build(
                X.loc[mask].reset_index(drop=True), y[mask], depth + 1
            ),
            "right": self._build(
                X.loc[~mask].reset_index(drop=True), y[~mask], depth + 1
            ),
        }

    def fit(self, X, y):
        X = X.reset_index(drop=True).copy()
        self.medians_ = X.median(numeric_only=True)
        X = X.fillna(self.medians_)
        self.feature_names_ = list(X.columns)
        self.tree_ = self._build(X, np.asarray(y), 0)
        return self

    def _predict_one(self, node, row):
        while node["type"] == "split":
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["prediction"]

    def predict(self, X):
        X = X[self.feature_names_].copy().fillna(self.medians_)
        return np.array(
            [self._predict_one(self.tree_, row) for row in X.to_dict(orient="records")]
        )

    def prune(self, X_val, y_val):
        X_val = (
            X_val[self.feature_names_]
            .copy()
            .fillna(self.medians_)
            .reset_index(drop=True)
        )
        y_val = np.asarray(y_val)
        self.tree_ = self._prune_node(self.tree_, X_val, y_val)
        return self

    def _prune_node(self, node, Xv, yv):
        if node["type"] == "leaf":
            return node

        feature, threshold = node["feature"], node["threshold"]
        mask = Xv[feature].to_numpy(dtype=float) <= threshold

        node["left"] = self._prune_node(
            node["left"], Xv.loc[mask].reset_index(drop=True), yv[mask]
        )
        node["right"] = self._prune_node(
            node["right"], Xv.loc[~mask].reset_index(drop=True), yv[~mask]
        )

        if len(yv) == 0:
            return {
                "type": "leaf",
                "prediction": node["prediction"],
                "samples": node["samples"],
            }

        subtree_pred = np.array(
            [self._predict_one(node, row) for row in Xv.to_dict(orient="records")]
        )
        leaf_pred = np.repeat(node["prediction"], len(yv))
        if np.sum(leaf_pred != yv) <= np.sum(subtree_pred != yv):
            return {
                "type": "leaf",
                "prediction": node["prediction"],
                "samples": node["samples"],
            }
        return node
