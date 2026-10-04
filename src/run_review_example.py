"""Run the model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.core_estimation import (
    FIGURE_S5_SCENARIO_IDS,
    MAIN_SCENARIO_IDS,
    SEED,
    fixed_multiplier_sensitivity,
    sample_all_inputs,
    summarise_global_scenarios,
    validate_analysis_ready_input,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/illustrative_analysis_ready_input.csv"),
        help="One-row-per-country analysis-ready CSV. The bundled file is synthetic.",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Directory for generated CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    data = validate_analysis_ready_input(pd.read_csv(args.input))
    rng = np.random.default_rng(SEED)
    draws, sampled_ratios = sample_all_inputs(data, rng)

    # Observed and scenario calculations use the same draws.
    summary = summarise_global_scenarios(data, draws, sampled_ratios, (*MAIN_SCENARIO_IDS, *FIGURE_S5_SCENARIO_IDS))
    sensitivity = fixed_multiplier_sensitivity(data, draws)

    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "global_scenario_summary.csv", index=False)
    sensitivity.to_csv(args.out / "fixed_multiplier_sensitivity.csv", index=False)

    observed = summary.loc[summary["scenario_id"].eq("Observed")].iloc[0]
    print(f"Countries/territories: {len(data)}")
    print(f"Observed attributable infections (synthetic example): {round(observed['attributable_infections_point']):,}")
    print(f"Wrote: {args.out.resolve()}")


if __name__ == "__main__":
    main()
