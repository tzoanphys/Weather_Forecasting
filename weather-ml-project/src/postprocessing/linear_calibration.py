import numpy as np


class LinearCalibration:
    """
    Simple MOS-style calibration.

    It learns a linear correction:
        y_true ≈ a * y_pred + b

    Then it applies:
        y_calib = a * y_pred + b
    """

    def __init__(self):
        # one slope per channel
        self.slopes_ = None

        # one intercept per channel
        self.intercepts_ = None

    def fit(self, y_true_list: list, y_pred_list: list) -> "LinearCalibration":
        """
        Learn one linear correction per channel.
        """
        # safety check: we need data to fit
        if len(y_true_list) == 0:
            raise ValueError("y_true_list is empty — cannot fit calibration.")

        # number of channels, for you usually 2: u10 and v10
        n_channels = y_true_list[0].shape[0]

        # arrays that will store the fitted coefficients
        slopes = np.zeros(n_channels)
        intercepts = np.zeros(n_channels)

        # fit one regression for each channel
        for c in range(n_channels):
            # collect all predicted values for this channel into one long vector
            x = np.concatenate([yp[c].ravel() for yp in y_pred_list])

            # collect all true values for this channel into one long vector
            y = np.concatenate([yt[c].ravel() for yt in y_true_list])

            # fit y ≈ a*x + b
            a, b = np.polyfit(x, y, deg=1)

            # store the fitted coefficients
            slopes[c] = a
            intercepts[c] = b

        # save learned coefficients inside the object
        self.slopes_ = slopes
        self.intercepts_ = intercepts

        # print a small summary
        ch_names = ["u10", "v10"] + [f"ch{i}" for i in range(2, n_channels)]
        for c in range(n_channels):
            name = ch_names[c] if c < len(ch_names) else f"ch{c}"
            print(f"[LinearCalib] {name}: a={slopes[c]:.4f}  b={intercepts[c]:.4f}")

        return self

    def apply(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply the learned linear correction to one prediction.
        """
        # cannot apply if fit() was never called
        if self.slopes_ is None:
            raise RuntimeError("Call fit() before apply().")

        # create output array with same shape as input
        y_calib = np.empty_like(y_pred)

        # apply one linear correction per channel
        for c in range(y_pred.shape[0]):
            y_calib[c] = self.slopes_[c] * y_pred[c] + self.intercepts_[c]

        return y_calib

    def save(self, path) -> None:
        """
        Save the fitted coefficients to disk.
        """
        if self.slopes_ is None:
            raise RuntimeError("Nothing to save — model has not been fitted.")

        np.savez(path, slopes=self.slopes_, intercepts=self.intercepts_)
        print(f"[LinearCalib] saved to {path}")

    @classmethod
    def load(cls, path) -> "LinearCalibration":
        """
        Load fitted coefficients from disk.
        """
        obj = cls()
        data = np.load(path)
        obj.slopes_ = data["slopes"]
        obj.intercepts_ = data["intercepts"]
        print(f"[LinearCalib] loaded from {path}")
        return obj