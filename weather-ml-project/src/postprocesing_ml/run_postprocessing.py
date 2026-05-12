from __future__ import annotations

"""
Main postprocessing runner.

Recommended order
-----------------
1. Put a forecast CSV in data/raw/forecast.csv
2. Run this file
3. The script will:
   - download observations,
   - match forecasts with observations,
   - evaluate correction methods,
   - save the results.
"""

from .config import PostprocessingConfig
from .evaluate_postprocessing import evaluate_all_methods
from .fetch_observations import download_hourly_observations
from .match_forecast_obs import build_training_table


def main() -> None:
    config = PostprocessingConfig()
    config.ensure_directories()

    print("Step 1: Downloading real observations...")
    download_hourly_observations(config)

    print("Step 2: Building matched forecast-observation table...")
    matched_df = build_training_table(config)

    print("Step 3: Evaluating postprocessing methods...")
    results = evaluate_all_methods(matched_df, config)

    print("\nFinal scores")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
