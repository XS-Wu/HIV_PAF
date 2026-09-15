"""Core calculations.

The bundled CSV is simulated and is only used to run this example.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


N_DRAWS = 10_000
N_SAMPLES = N_DRAWS + 1
SEED = 20260529

SEXES = ("male", "female")
CASCADE_STEPS = ("first95", "second95", "third95")
COMPONENTS = ("undiag", "diag_notreat", "diag_unsuppTX")

LAMBDA_U = 6.6
LAMBDA_NOTREAT = 5.3
LAMBDA_UNSUPP_TX = 1.8
LAMBDA_SUPP = 0.4
LAMBDA_INTERVALS = {
    "undiag": (6.3, LAMBDA_U, 7.0),
    "diag_notreat": (5.1, LAMBDA_NOTREAT, 5.5),
    "diag_unsuppTX": (1.6, LAMBDA_UNSUPP_TX, 2.0),
    "supp": (0.4, LAMBDA_SUPP, 0.4),
}


SCENARIO_RULES: dict[str, Mapping[str, str]] = {
    "Observed": {},
    "S1": {"first95": "plus1"},
    "S2": {"second95": "plus1"},
    "S3": {"third95": "plus1"},
    "S4": {"first95": "plus1", "second95": "plus1", "third95": "plus1"},
    "S5": {"first95": "floor95"},
    "S6": {"second95": "floor95"},
    "S7": {"third95": "floor95"},
    "S8": {"first95": "floor95", "second95": "floor95", "third95": "floor95"},
    "S9": {"first95": "floor97"},
    "S10": {"second95": "floor97"},
    "S11": {"third95": "floor97"},
    "S12": {"first95": "floor97", "second95": "floor97", "third95": "floor97"},
    "S13": {"first95": "floor99"},
    "S14": {"second95": "floor99"},
    "S15": {"third95": "floor99"},
    "S16": {"first95": "floor99", "second95": "floor99", "third95": "floor99"},
    "M1": {"first95": "minus1"},
    "M2": {"second95": "minus1"},
    "M3": {"third95": "minus1"},
    "M4": {"first95": "minus1", "second95": "minus1", "third95": "minus1"},
}

SCENARIO_LABELS = {
    "Observed": "Observed in 2023",
    "S1": "First 95 +1%",
    "S2": "Second 95 +1%",
    "S3": "Third 95 +1%",
    "S4": "All three +1%",
    "S5": "First 95 to 95%",
    "S6": "Second 95 to 95%",
    "S7": "Third 95 to 95%",
    "S8": "All three to 95%",
    "S9": "First 95 to 97%",
    "S10": "Second 95 to 97%",
    "S11": "Third 95 to 97%",
    "S12": "All three to 97%",
    "S13": "First 95 to 99%",
    "S14": "Second 95 to 99%",
    "S15": "Third 95 to 99%",
    "S16": "All three to 99%",
    "M1": "First 95 -1%",
    "M2": "Second 95 -1%",
    "M3": "Third 95 -1%",
    "M4": "All three -1%",
}

MAIN_SCENARIO_IDS = ("Observed", *(f"S{index}" for index in range(1, 17)))
FIGURE_S5_SCENARIO_IDS = ("M1", "M2", "M3", "M4")


def required_columns() -> list[str]:
    """Columns needed after source data have been selected and harmonised."""
    columns = ["ISO3", "Country", "Super region", "Pop15p_2023"]
    for prefix in ("PLHIV15p", "inc_male", "inc_female"):
        columns.extend(f"{prefix}_{stat}" for stat in ("point", "low", "high"))
    for step in CASCADE_STEPS:
        for sex in SEXES:
            columns.extend(f"{step}_{sex}_{stat}" for stat in ("point", "low", "high"))
    return columns


def validate_analysis_ready_input(data: pd.DataFrame) -> pd.DataFrame:
    """Check analysis inputs before sampling."""
    missing = [column for column in required_columns() if column not in data.columns]
    if missing:
        raise ValueError(f"Analysis-ready input is missing columns: {', '.join(missing)}")

    out = data.loc[:, required_columns()].copy()
    if out["ISO3"].duplicated().any():
        repeated = out.loc[out["ISO3"].duplicated(), "ISO3"].tolist()
        raise ValueError(f"Exactly one row per ISO3 is required; duplicates: {repeated}")
    if out[["ISO3", "Country", "Super region"]].isna().any().any():
        raise ValueError("ISO3, Country, and Super region must be present for every row.")

    numeric = [column for column in required_columns() if column not in {"ISO3", "Country", "Super region"}]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")
    if out[numeric].isna().any().any() or not np.isfinite(out[numeric].to_numpy(float)).all():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"All selected source points and bounds must be finite; check: {bad}")
    if (out["Pop15p_2023"] <= 0).any():
        raise ValueError("Adult population denominators must be positive.")

    for step in CASCADE_STEPS:
        for sex in SEXES:
            cols = [f"{step}_{sex}_{stat}" for stat in ("point", "low", "high")]
            if ((out[cols] < 0) | (out[cols] > 1)).any().any():
                raise ValueError(f"Cascade proportions must lie in [0, 1]: {step}, {sex}.")

    for prefix in ("PLHIV15p", "inc_male", "inc_female"):
        cols = [f"{prefix}_{stat}" for stat in ("point", "low", "high")]
        if (out[cols] < 0).any().any():
            raise ValueError(f"Counts cannot be negative: {prefix}.")
    return out


def _ordered_bounds(point: np.ndarray, low: np.ndarray, high: np.ndarray, *, proportion: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure bounds contain the point estimate and respect proportion limits."""
    point = np.asarray(point, dtype=float)
    low = np.minimum(np.asarray(low, dtype=float), point)
    high = np.maximum(np.asarray(high, dtype=float), point)
    if proportion:
        low = np.maximum(low, 0.0)
        high = np.minimum(high, 1.0)
    return point, low, high


def sample_source_interval(
    rng: np.random.Generator,
    point: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    proportion: bool,
) -> np.ndarray:
    """Generate triangular draws; row 0 stores the source point estimate."""
    point, low, high = _ordered_bounds(point, low, high, proportion=proportion)
    draws = np.empty((N_SAMPLES, point.size), dtype=float)
    draws[0, :] = point
    for index in range(point.size):
        if high[index] <= low[index]:
            draws[1:, index] = point[index]
        else:
            draws[1:, index] = rng.triangular(low[index], point[index], high[index], size=N_DRAWS)
    return draws


def sample_all_inputs(data: pd.DataFrame, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Generate the draws used by the observed and scenario calculations."""
    draws: dict[str, np.ndarray] = {}
    for step in CASCADE_STEPS:
        for sex in SEXES:
            prefix = f"{step}_{sex}"
            draws[prefix] = sample_source_interval(
                rng,
                data[f"{prefix}_point"].to_numpy(float),
                data[f"{prefix}_low"].to_numpy(float),
                data[f"{prefix}_high"].to_numpy(float),
                proportion=True,
            )
    for sex in SEXES:
        prefix = f"inc_{sex}"
        draws[prefix] = sample_source_interval(
            rng,
            data[f"{prefix}_point"].to_numpy(float),
            data[f"{prefix}_low"].to_numpy(float),
            data[f"{prefix}_high"].to_numpy(float),
            proportion=False,
        )
    draws["plhiv"] = sample_source_interval(
        rng,
        data["PLHIV15p_point"].to_numpy(float),
        data["PLHIV15p_low"].to_numpy(float),
        data["PLHIV15p_high"].to_numpy(float),
        proportion=False,
    )

    rates: dict[str, np.ndarray] = {}
    for state, (low, mode, high) in LAMBDA_INTERVALS.items():
        rates[state] = sample_source_interval(
            rng,
            np.array([mode]),
            np.array([low]),
            np.array([high]),
            proportion=False,
        )[:, 0]
    ratios = {
        "undiag": rates["undiag"] / rates["supp"],
        "diag_notreat": rates["diag_notreat"] / rates["supp"],
        "diag_unsuppTX": rates["diag_unsuppTX"] / rates["supp"],
    }
    return draws, ratios


def fixed_reference_ratios(multiplier: float = 1.0) -> dict[str, np.ndarray]:
    """Transmission-rate ratios for the fixed-multiplier sensitivity analysis."""
    return {
        "undiag": np.repeat((LAMBDA_U / LAMBDA_SUPP) * multiplier, N_SAMPLES),
        "diag_notreat": np.repeat((LAMBDA_NOTREAT / LAMBDA_SUPP) * multiplier, N_SAMPLES),
        "diag_unsuppTX": np.repeat((LAMBDA_UNSUPP_TX / LAMBDA_SUPP) * multiplier, N_SAMPLES),
    }


def paf_components(
    first95: np.ndarray,
    second95: np.ndarray,
    third95: np.ndarray,
    ratios: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Calculate total and state-specific PAFs."""
    q_u = 1.0 - first95
    q_n = first95 * (1.0 - second95)
    q_t = first95 * second95 * (1.0 - third95)

    w_u = q_u * (ratios["undiag"][:, None] - 1.0)
    w_n = q_n * (ratios["diag_notreat"][:, None] - 1.0)
    w_t = q_t * (ratios["diag_unsuppTX"][:, None] - 1.0)
    sum_w = w_u + w_n + w_t
    paf_total = np.divide(sum_w, 1.0 + sum_w, out=np.zeros_like(sum_w), where=sum_w > 0)

    components = {"unsupp_total": paf_total}
    for name, weight in (("undiag", w_u), ("diag_notreat", w_n), ("diag_unsuppTX", w_t)):
        components[name] = np.divide(paf_total * weight, sum_w, out=np.zeros_like(sum_w), where=sum_w > 0)
    return components


def apply_rule(values: np.ndarray, rule: str | None) -> np.ndarray:
    """Apply one cascade scenario rule."""
    if rule is None:
        return values
    if rule == "plus1":
        return np.minimum(values + 0.01, 1.0)
    if rule == "minus1":
        return np.maximum(values - 0.01, 0.0)
    if rule.startswith("floor"):
        return np.maximum(values, float(rule.removeprefix("floor")) / 100.0)
    raise ValueError(f"Unknown scenario rule: {rule}")


def compute_metrics(
    data: pd.DataFrame,
    draws: Mapping[str, np.ndarray],
    ratios: Mapping[str, np.ndarray],
    rules: Mapping[str, str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute sex-specific and combined results for one scenario."""
    rules = {} if rules is None else rules
    sex_results: dict[str, dict[str, np.ndarray]] = {}
    for sex in SEXES:
        first = apply_rule(draws[f"first95_{sex}"], rules.get("first95"))
        second = apply_rule(draws[f"second95_{sex}"], rules.get("second95"))
        third = apply_rule(draws[f"third95_{sex}"], rules.get("third95"))
        p = paf_components(first, second, third, ratios)
        incidence = draws[f"inc_{sex}"]
        sex_results[sex] = {}
        for component in (*COMPONENTS, "unsupp_total"):
            sex_results[sex][f"paf_{component}"] = p[component]
            sex_results[sex][f"attr_{component}"] = p[component] * incidence

    combined: dict[str, np.ndarray] = {}
    total_incidence = draws["inc_male"] + draws["inc_female"]
    uninfected = np.maximum(data["Pop15p_2023"].to_numpy(float)[None, :] - draws["plhiv"], 1.0)
    for component in (*COMPONENTS, "unsupp_total"):
        attributable = sex_results["male"][f"attr_{component}"] + sex_results["female"][f"attr_{component}"]
        combined[f"attr_{component}"] = attributable
        combined[f"rate_{component}"] = attributable / uninfected * 100_000.0
        combined[f"paf_{component}"] = np.divide(
            attributable, total_incidence, out=np.zeros_like(attributable), where=total_incidence > 0
        ) * 100.0
    combined["inc_total"] = total_incidence
    combined["denom"] = uninfected
    return {"male": sex_results["male"], "female": sex_results["female"], "total": combined}


def draw_summary(values: np.ndarray) -> tuple[float, float, float]:
    """Return the source-point estimate and mean-centred 95% UI."""
    values = np.asarray(values, dtype=float)
    point = float(np.nanmean(values[0]))
    random_draws = values[1:]
    centre = float(np.nanmean(random_draws))
    raw_low, raw_high = np.nanpercentile(random_draws, [2.5, 97.5])
    low = point + (float(raw_low) - centre)
    high = point + (float(raw_high) - centre)
    if np.nanmin(values) >= 0:
        low = max(0.0, low)
    return point, low, high


def difference_summary(baseline: np.ndarray, scenario: np.ndarray) -> tuple[float, float, float]:
    """Summarise a paired draw-level difference."""
    return draw_summary(np.asarray(baseline) - np.asarray(scenario))


def summarise_global_scenarios(
    data: pd.DataFrame,
    draws: Mapping[str, np.ndarray],
    ratios: Mapping[str, np.ndarray],
    scenario_ids: tuple[str, ...] | list[str] = MAIN_SCENARIO_IDS,
) -> pd.DataFrame:
    """Build the global scenario summary."""
    baseline = compute_metrics(data, draws, ratios)
    baseline_attr = baseline["total"]["attr_unsupp_total"].sum(axis=1)
    baseline_rate = baseline_attr / baseline["total"]["denom"].sum(axis=1) * 100_000.0
    observed_incidence = baseline["total"]["inc_total"].sum(axis=1)
    adult_population = float(data["Pop15p_2023"].sum())

    rows = []
    for scenario_id in scenario_ids:
        if scenario_id not in SCENARIO_RULES:
            raise KeyError(f"Unknown scenario identifier: {scenario_id}")
        if scenario_id == "Observed":
            scenario_attr = baseline_attr
            scenario_rate = baseline_rate
            total_infections = observed_incidence
            reduction = np.zeros_like(baseline_attr)
            reduction_rate = np.zeros_like(baseline_rate)
        else:
            scenario = compute_metrics(data, draws, ratios, SCENARIO_RULES[scenario_id])
            scenario_attr = scenario["total"]["attr_unsupp_total"].sum(axis=1)
            scenario_rate = scenario_attr / scenario["total"]["denom"].sum(axis=1) * 100_000.0
            reduction = baseline_attr - scenario_attr
            reduction_rate = baseline_rate - scenario_rate
            # Same-year static counterfactual quantity.
            total_infections = observed_incidence - reduction

        row = {"scenario_id": scenario_id, "scenario": SCENARIO_LABELS[scenario_id]}
        for name, values in {
            "total_infections": total_infections,
            "total_rate": total_infections / adult_population * 100_000.0,
            "attributable_infections": scenario_attr,
            "attributable_rate": scenario_rate,
            "reduction_infections": reduction,
            "reduction_rate": reduction_rate,
        }.items():
            point, low, high = draw_summary(values)
            row.update({f"{name}_point": point, f"{name}_low": low, f"{name}_high": high})
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_multiplier_sensitivity(
    data: pd.DataFrame,
    draws: Mapping[str, np.ndarray],
    scenario_ids: tuple[str, ...] = MAIN_SCENARIO_IDS,
) -> pd.DataFrame:
    """Repeat scenarios with fixed 0.50-1.50 ratio multipliers."""
    frames = []
    for label, multiplier in (("Lower 50%", 0.50), ("Lower 25%", 0.75), ("Fixed reference", 1.00), ("Higher 25%", 1.25), ("Higher 50%", 1.50)):
        summary = summarise_global_scenarios(data, draws, fixed_reference_ratios(multiplier), scenario_ids)
        summary.insert(0, "lambda_ratio_multiplier", multiplier)
        summary.insert(0, "sensitivity", label)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)
