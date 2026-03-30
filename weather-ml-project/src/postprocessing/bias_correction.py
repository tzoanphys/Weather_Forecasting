"""
bias_correction.py
==================
Grid-point-wise mean bias correction for wind forecasts.

Theory (Vannitsem et al. 2021, Section "State-of-the-art", p. E682-E683)
------------------------------------------------------------------------
The simplest and most widely used postprocessing method is mean bias
correction.  For each grid point the systematic offset between the
model and the truth is estimated on a training/fitting set and then
subtracted from future predictions:

    correction[c, lat, lon] = mean( y_true - y_pred )   (over fitting set)
    y_postprocessed = y_pred + correction

This corrects the *systematic* (mean) component of the error.  Random
errors are not affected.

For wind components u10 and v10 the correction is computed independently
for each channel and each spatial grid point, so the spatial structure
of the bias is preserved.
"""

import numpy as np


class MeanBiasCorrection:
    """
    Fit and apply a grid-point-wise mean bias correction.

    Parameters
    ----------
    None

    Attributes
    ----------
    correction_ : np.ndarray, shape (C, H, W)
        Mean bias (y_true - y_pred) estimated on the fitting set.
        None before fit() is called.
    """

    def __init__(self):
        self.correction_ = None

    # ------------------------------------------------------------------
    def fit(self, y_true_list: list, y_pred_list: list) -> "MeanBiasCorrection":
        """
        Estimate the mean bias from a list of (true, pred) pairs.

        Parameters
        ----------
        y_true_list : list of np.ndarray, each shape (C, H, W)
        y_pred_list : list of np.ndarray, each shape (C, H, W)

        Returns
        -------
        self
        """
        if len(y_true_list) == 0:
            raise ValueError("y_true_list is empty — cannot fit bias correction.")

        errors = np.stack(
            [yt - yp for yt, yp in zip(y_true_list, y_pred_list)],
            axis=0
        )  # shape (N, C, H, W)

        self.correction_ = errors.mean(axis=0)  # shape (C, H, W)

        print(f"  [BiasCorrection] Fitted on {len(y_true_list)} samples.")
        print(f"  Mean correction  u10: {self.correction_[0].mean():.4f} m/s")
        print(f"  Mean correction  v10: {self.correction_[1].mean():.4f} m/s")
        return self

    # ------------------------------------------------------------------
    def apply(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply the bias correction to a single prediction.

        Parameters
        ----------
        y_pred : np.ndarray, shape (C, H, W)

        Returns
        -------
        np.ndarray, shape (C, H, W)  — bias-corrected prediction
        """
        if self.correction_ is None:
            raise RuntimeError("Call fit() before apply().")
        return y_pred + self.correction_

    # ------------------------------------------------------------------
    def save(self, path) -> None:
        """Save the fitted correction array to a .npy file."""
        if self.correction_ is None:
            raise RuntimeError("Nothing to save — model has not been fitted.")
        np.save(path, self.correction_)
        print(f"  [BiasCorrection] Saved to {path}")

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path) -> "MeanBiasCorrection":
        """Load a previously saved correction array."""
        obj = cls()
        obj.correction_ = np.load(path)
        print(f"  [BiasCorrection] Loaded from {path}")
        return obj
