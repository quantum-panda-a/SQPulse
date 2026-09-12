"""IQ plane discrimination and confusion matrix analysis for dispersive readout."""

from __future__ import annotations
from typing import Tuple, Dict, Optional
import numpy as np


class IQDiscriminator:
    r"""Linear threshold discriminator for binary state classification in the IQ plane.

    Classifies IQ points by projecting onto the axis connecting the cluster centers:
    .. math::
        \Delta S = S_1 - S_0
        \text{score}(z) = \text{Re}\left( (z - S_{\text{mid}}) \cdot \frac{\Delta S^*}{|\Delta S|} \right)
        \text{state} = 1 \text{ if score} \ge 0 \text{ else } 0

    Confusion matrix definition:
    .. math::
        M_{i, j} = P(\text{assigned } i \mid \text{prepared } j) = \begin{pmatrix} P(0|0) & P(0|1) \\ P(1|0) & P(1|1) \end{pmatrix}
        \mathcal{F}_{\text{ro}} = \frac{P(0|0) + P(1|1)}{2}
    """

    def __init__(
        self,
        center_0: complex = 0.0 + 0.0j,
        center_1: complex = 1.0 + 0.0j,
    ):
        self.center_0 = complex(center_0)
        self.center_1 = complex(center_1)
        self._update_geometry()

    def _update_geometry(self) -> None:
        self.midpoint = 0.5 * (self.center_0 + self.center_1)
        delta = self.center_1 - self.center_0
        dist = abs(delta)
        if dist > 1e-15:
            self.unit_vector = delta / dist
        else:
            self.unit_vector = 1.0 + 0.0j

    def fit_centers(self, center_0: complex, center_1: complex) -> IQDiscriminator:
        """Fit discriminator directly from cluster centroids."""
        self.center_0 = complex(center_0)
        self.center_1 = complex(center_1)
        self._update_geometry()
        return self

    def fit(self, iq_0: np.ndarray, iq_1: np.ndarray) -> IQDiscriminator:
        """Fit discriminator from calibration shots of state |0> and |1>."""
        c0 = np.mean(iq_0)
        c1 = np.mean(iq_1)
        return self.fit_centers(c0, c1)

    def predict(self, iq_points: np.ndarray) -> np.ndarray:
        """Assign states (0 or 1) to IQ data points based on linear decision threshold."""
        pts = np.asarray(iq_points, dtype=complex)
        scores = np.real((pts - self.midpoint) * np.conj(self.unit_vector))
        return np.where(scores >= 0.0, 1, 0)

    @staticmethod
    def compute_confusion_matrix(
        true_labels: np.ndarray,
        assigned_labels: np.ndarray,
    ) -> np.ndarray:
        """Compute the 2x2 confusion matrix: M[i, j] = P(assigned i | true j)."""
        true_labels = np.asarray(true_labels, dtype=int)
        assigned_labels = np.asarray(assigned_labels, dtype=int)

        m = np.zeros((2, 2), dtype=float)
        for true_state in (0, 1):
            mask = true_labels == true_state
            n_total = np.sum(mask)
            if n_total > 0:
                m[0, true_state] = np.sum(assigned_labels[mask] == 0) / n_total
                m[1, true_state] = np.sum(assigned_labels[mask] == 1) / n_total
            else:
                m[true_state, true_state] = 1.0

        return m

    @staticmethod
    def compute_fidelity(confusion_matrix: np.ndarray) -> float:
        """Compute readout assignment fidelity: F = (P(0|0) + P(1|1)) / 2."""
        return float(0.5 * (confusion_matrix[0, 0] + confusion_matrix[1, 1]))
