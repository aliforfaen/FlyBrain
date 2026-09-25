"""Readout learning for the spike decoder.

:class:`ReadoutLearner` collects ``(features, target)`` pairs — one feature
vector per control tick, built by :meth:`flybrain.codec.SpikeDecoder.features`,
whose layout is ``[rate(action_0), ..., rate(action_{A-1}), 1.0]`` — and fits
the linear readout ``W`` with either closed-form ridge regression or an
iterative delta rule.  The learned ``W`` has shape ``(n_actions, n_features)``
and can be handed straight to :meth:`flybrain.codec.SpikeDecoder.set_weights`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flybrain.types import Episode

__all__ = ["ReadoutLearner"]


def _npz_path(path: str | Path) -> Path:
    """Return ``path`` with an explicit ``.npz`` suffix (``np.savez`` adds it)."""
    p = Path(path)
    if p.suffix != ".npz":
        p = p.with_name(p.name + ".npz")
    return p


class ReadoutLearner:
    """Collects training samples and fits the decoder's linear readout.

    The feature vector is assumed to end with a constant ``1.0`` bias term, so
    ridge regression regularizes every diagonal entry of ``X^T X`` **except the
    last one**: ``W = (X^T X + diag([l2] * (D - 1) + [0]))^{-1} X^T Y``.  The
    system is solved with :func:`numpy.linalg.solve` (never an explicit inverse)
    and falls back to least squares if it is singular.
    """

    def __init__(self, n_features: int, n_actions: int, l2: float = 1.0) -> None:
        """Configure the learner.

        Args:
            n_features: Length ``D`` of each feature vector, including the bias.
            n_actions: Number ``A`` of action keys (rows of ``W``).
            l2: Ridge penalty; ``0.0`` disables regularization.
        """
        self.n_features = int(n_features)
        self.n_actions = int(n_actions)
        self.l2 = float(l2)
        self.W: np.ndarray = np.zeros((self.n_actions, self.n_features), dtype=np.float64)
        self._X: list[np.ndarray] = []
        self._Y: list[np.ndarray] = []

    def __len__(self) -> int:
        """Number of collected samples."""
        return len(self._X)

    def _check_pair(
        self, features: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.float64).reshape(-1)
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.n_features:
            raise ValueError(f"features must have length {self.n_features}, got {x.shape[0]}")
        if y.shape[0] != self.n_actions:
            raise ValueError(f"target must have length {self.n_actions}, got {y.shape[0]}")
        return x, y

    def add(self, features: np.ndarray, target: np.ndarray) -> None:
        """Append one ``(features, target)`` sample (copies are stored)."""
        x, y = self._check_pair(features, target)
        self._X.append(x.copy())
        self._Y.append(y.copy())

    def add_episode(self, ep: Episode) -> None:
        """Append the sample carried by an :class:`~flybrain.types.Episode`."""
        self.add(ep.features, ep.target)

    def extend(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Append a batch of samples; accepts 1-D single samples too."""
        Xa = np.asarray(X, dtype=np.float64)
        Ya = np.asarray(Y, dtype=np.float64)
        if Xa.ndim == 1:
            Xa = Xa.reshape(1, -1)
        if Ya.ndim == 1:
            Ya = Ya.reshape(1, -1)
        if Xa.shape[0] != Ya.shape[0]:
            raise ValueError(f"X has {Xa.shape[0]} rows but Y has {Ya.shape[0]}")
        for i in range(Xa.shape[0]):
            self.add(Xa[i], Ya[i])

    def _stack(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._X:
            return (
                np.zeros((0, self.n_features), dtype=np.float64),
                np.zeros((0, self.n_actions), dtype=np.float64),
            )
        return np.vstack(self._X), np.vstack(self._Y)

    def fit(
        self,
        method: str = "ridge",
        lr: float = 0.05,
        epochs: int = 200,
    ) -> np.ndarray:
        """Fit ``W`` and return it (also stored as :attr:`W`).

        Args:
            method: ``"ridge"`` for closed-form regularized least squares or
                ``"delta"`` for an iterative per-sample delta rule.
            lr: Delta-rule learning rate (ignored by ``"ridge"``).
            epochs: Delta-rule passes over the collected samples.

        Returns:
            The ``(n_actions, n_features)`` weight matrix; zeros when no samples
            were collected.  Values are guaranteed finite.
        """
        if method == "ridge":
            W = self._fit_ridge()
        elif method == "delta":
            W = self._fit_delta(lr=lr, epochs=epochs)
        else:
            raise ValueError(f"unknown fit method: {method!r}")
        self.W = W
        return W

    def _fit_ridge(self) -> np.ndarray:
        """Solve ``(X^T X + diag([l2]*(D-1) + [0])) W = X^T Y``."""
        X, Y = self._stack()
        if X.shape[0] == 0:
            return np.zeros((self.n_actions, self.n_features), dtype=np.float64)

        gram = X.T @ X
        if self.l2 != 0.0 and self.n_features > 0:
            gram = gram + self.l2 * np.eye(self.n_features, dtype=np.float64)
            gram[-1, -1] -= self.l2  # bias column stays unregularized
        rhs = X.T @ Y  # (D, A)
        try:
            # Solve for W^T, shape (D, A); the caller wants W of shape (A, D).
            W_t = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            W_t, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
        W = np.nan_to_num(W_t, nan=0.0, posinf=0.0, neginf=0.0).T
        return np.ascontiguousarray(W)

    def _fit_delta(self, lr: float, epochs: int) -> np.ndarray:
        """Per-sample Widrow-Hoff delta rule with optional weight decay."""
        X, Y = self._stack()
        W = np.zeros((self.n_actions, self.n_features), dtype=np.float64)
        lr = float(lr)
        for _ in range(max(int(epochs), 0)):
            for i in range(X.shape[0]):
                x = X[i]
                error = Y[i] - W @ x
                W += lr * (np.outer(error, x) - self.l2 * W)
        return np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return raw ``X @ W.T`` scores with shape ``(n_samples, n_actions)``."""
        Xa = self._prepare_features(X)
        return Xa @ self.W.T

    def accuracy(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Fraction of correct binary decisions at threshold ``0.5``.

        Accuracy is computed elementwise over all ``(sample, action)`` pairs.
        An empty dataset scores ``0.0`` rather than NaN.
        """
        Xa, Ya = self._prepare_xy(X, Y)
        if Xa.shape[0] == 0:
            return 0.0
        correct = ((Xa @ self.W.T) > 0.5) == (Ya > 0.5)
        return float(np.mean(correct))

    def per_action_accuracy(self, X: np.ndarray, Y: np.ndarray) -> dict[int, float]:
        """Per-action-index binary accuracy at threshold ``0.5``."""
        Xa, Ya = self._prepare_xy(X, Y)
        result: dict[int, float] = {}
        if Xa.shape[0] == 0:
            return {j: 0.0 for j in range(self.n_actions)}
        correct = ((Xa @ self.W.T) > 0.5) == (Ya > 0.5)
        for j in range(self.n_actions):
            result[j] = float(np.mean(correct[:, j]))
        return result

    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim == 1:
            Xa = Xa.reshape(1, -1)
        if Xa.shape[1] != self.n_features:
            raise ValueError(f"X must have {self.n_features} columns, got {Xa.shape[1]}")
        return Xa

    def _prepare_xy(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xa = self._prepare_features(X)
        Ya = np.asarray(Y, dtype=np.float64)
        if Ya.ndim == 1:
            Ya = Ya.reshape(-1, 1)
        if Ya.shape[1] != self.n_actions:
            raise ValueError(f"Y must have {self.n_actions} columns, got {Ya.shape[1]}")
        if Xa.shape[0] != Ya.shape[0]:
            raise ValueError(f"X has {Xa.shape[0]} rows but Y has {Ya.shape[0]}")
        return Xa, Ya

    def save(self, path: str | Path) -> Path:
        """Persist weights, config and collected samples to ``.npz``."""
        p = _npz_path(path)
        X, Y = self._stack()
        np.savez(
            p,
            W=self.W,
            n_features=self.n_features,
            n_actions=self.n_actions,
            l2=self.l2,
            X=X,
            Y=Y,
        )
        return p

    def load(self, path: str | Path) -> ReadoutLearner:
        """Load state previously written by :meth:`save`; returns ``self``."""
        p = _npz_path(path)
        with np.load(p, allow_pickle=False) as data:
            self.n_features = int(data["n_features"])
            self.n_actions = int(data["n_actions"])
            self.l2 = float(data["l2"])
            self.W = np.array(data["W"], dtype=np.float64)
            X = np.asarray(data["X"], dtype=np.float64).reshape(-1, self.n_features)
            Y = np.asarray(data["Y"], dtype=np.float64).reshape(-1, self.n_actions)
            self._X = [X[i].copy() for i in range(X.shape[0])]
            self._Y = [Y[i].copy() for i in range(Y.shape[0])]
        return self

    @classmethod
    def from_npz(cls, path: str | Path) -> ReadoutLearner:
        """Convenience constructor: load a learner and its shapes from ``.npz``."""
        return cls(0, 0).load(path)
