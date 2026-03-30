"""
linear_calibration.py
=====================
Per-channel linear MOS-style calibration for wind forecasts.

Theory (Vannitsem et al. 2021, p. E682-E683)
--------------------------------------------
Model Output Statistics (MOS), introduced by Glahn & Lowry (1972), is
the classical linear postprocessing approach used operationally at many
National Meteorological Services.  A linear regression is fitted between
the NWP (or ML) model output and the observations/truth:

    y_true ≈ a * y_pred + b

The coefficients a (slope) and b (intercept) are estimated on a fitting
set using ordinary least squares.  At inference time the correction is:

    y_postprocessed = a * y_pred + b

Applied here *per channel* (u10, v10), fitting a single (a, b) pair
over all spatial grid points and all fitting samples pooled together.
This is the simplest MOS variant — it corrects both mean bias (offset b)
and gain error (slope a ≠ 1).

A slope a < 1 means the model overshoots → correction shrinks the output.
A slope a > 1 means the model undershoots → correction amplifies it.
b ≈ 0 with a ≈ 1 means the model is already well-calibrated.
"""

import numpy as np


class LinearCalibration:
    """
    Per-channel linear MOS calibration: y_calib = a * y_pred + b.

    Parameters
    ----------
    None

    Attributes
    ----------
    slopes_     : np.ndarray, shape (C,)
    intercepts_ : np.ndarray, shape (C,)
        Fitted per-channel linear coefficients.  None before fit().
    """

    def __init__(self):
        self.slopes_     = None
        self.intercepts_ = None

    # ------------------------------------------------------------------
    def fit(self, y_true_list: list, y_pred_list: list) -> "LinearCalibration":
        """
        Fit per-channel linear regression over all fitting samples.

        Parameters
        ----------
        y_true_list : list of np.ndarray, each shape (C, H, W)
        y_pred_list : list of np.ndarray, each shape (C, H, W)

        Returns
        -------
        self
        """
        if len(y_true_list) == 0:
            raise ValueError("y_true_list is empty — cannot fit calibration.")

        n_channels = y_true_list[0].shape[0]
        slopes     = np.zeros(n_channels)
        intercepts = np.zeros(n_channels)

        for c in range(n_channels):
            # Pool all spatial grid points and all samples into 1-D arrays
            x = np.concatenate([yp[c].ravel() for yp in y_pred_list])
            y = np.concatenate([yt[c].ravel() for yt in y_true_list])

            # Ordinary least squares: [a, b] = polyfit(x, y, deg=1)
            a, b = np.polyfit(x, y, deg=1)
            slopes[c]     = a
            intercepts[c] = b

        self.slopes_     = slopes
        self.intercepts_ = intercepts

        ch_names = ["u10", "v10"] + [f"ch{i}" for i in range(2, n_channels)]
        for c in range(n_channels):
            name = ch_names[c] if c < len(ch_names) else f"ch{c}"
            print(f"  [LinearCalib] {name}:  a={slopes[c]:.4f}  b={intercepts[c]:.4f}")
        return self

    # ------------------------------------------------------------------
    def apply(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply the linear calibration to a single prediction.

        Parameters
        ----------
        y_pred : np.ndarray, shape (C, H, W)

        Returns
        -------
        np.ndarray, shape (C, H, W)  — calibrated prediction
        """
        if self.slopes_ is None:
            raise RuntimeError("Call fit() before apply().")

        y_calib = np.empty_like(y_pred)
        for c in range(y_pred.shape[0]):
            y_calib[c] = self.slopes_[c] * y_pred[c] + self.intercepts_[c]
        return y_calib

    # ------------------------------------------------------------------
    def save(self, path) -> None:
        """Save coefficients to a .npz file."""
        if self.slopes_ is None:
            raise RuntimeError("Nothing to save — model has not been fitted.")
        np.savez(path, slopes=self.slopes_, intercepts=self.intercepts_)
        print(f"  [LinearCalib] Saved to {path}")

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path) -> "LinearCalibration":
        """Load previously saved coefficients."""
        obj = cls()
        data = np.load(path)
        obj.slopes_     = data["slopes"]
        obj.intercepts_ = data["intercepts"]
        print(f"  [LinearCalib] Loaded from {path}")
        return obj
