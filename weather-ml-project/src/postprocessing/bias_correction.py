import numpy as np


class MeanBiasCorrection:
    """
    Simple spatial bias correction.

    It learns the average error:
        error = y_true - y_pred

    Then it adds this average error to future predictions.
    """

    def __init__(self):
        # here we will store the learned correction
        self.correction_ = None

    def fit(self, y_true_list: list, y_pred_list: list) -> "MeanBiasCorrection":
        """
        Learn the average bias from many prediction/true pairs.
        """
        # safety check: we need data to fit
        if len(y_true_list) == 0:
            raise ValueError("y_true_list is empty — cannot fit bias correction.")

        # compute error for each sample: truth - prediction
        errors = np.stack(
            [yt - yp for yt, yp in zip(y_true_list, y_pred_list)],
            axis=0
        )  # shape: (N, C, H, W)

        # average over all samples → one correction map
        self.correction_ = errors.mean(axis=0)  # shape: (C, H, W)

        print(f"[BiasCorrection] fitted on {len(y_true_list)} samples")
        print(f"mean u10 correction: {self.correction_[0].mean():.4f}")
        print(f"mean v10 correction: {self.correction_[1].mean():.4f}")

        return self

    def apply(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Add the learned correction to one prediction.
        """
        # cannot apply if fit() was never called
        if self.correction_ is None:
            raise RuntimeError("Call fit() before apply().")

        # corrected prediction = raw prediction + learned bias
        return y_pred + self.correction_

    def save(self, path) -> None:
        """
        Save the correction to disk.
        """
        if self.correction_ is None:
            raise RuntimeError("Nothing to save — model has not been fitted.")

        np.save(path, self.correction_)
        print(f"[BiasCorrection] saved to {path}")

    @classmethod
    def load(cls, path) -> "MeanBiasCorrection":
        """
        Load a saved correction from disk.
        """
        obj = cls()
        obj.correction_ = np.load(path)
        print(f"[BiasCorrection] loaded from {path}")
        return obj