#!/usr/bin/env python3
"""Build final parseable results package for the NAPS SBERT/TARJ paper."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import kendalltau, spearmanr


PACKAGE_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "NAPS_paper_experiments").is_dir() and (candidate / "single_node_summer_hourly_case").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {start}")


REPO_ROOT = find_repo_root(PACKAGE_DIR)
NAPS_DIR = REPO_ROOT / "NAPS_paper_experiments"
SOURCE_TABLES = NAPS_DIR / "outputs" / "sbert_direction_v2_six_treatment" / "tables"
NE_TABLE = (
    REPO_ROOT
    / "single_node_summer_hourly_case"
    / "refined_grid_equilibrium_analysis"
    / "tables"
    / "refined_ne_summary_by_hour.csv"
)

FIG_DIR = PACKAGE_DIR / "figures"
TABLE_DIR = PACKAGE_DIR / "tables"

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260628

TREATMENT_ORDER = [
    "tarj_compliance",
    "tarj_balanced_profit_compliance",
    "tarj_guided_profit",
    "tarj_default",
    "tarj_profit_max",
    "tarj_aggressive_profit_max",
]
TREATMENT_CODES = {
    "tarj_compliance": "T1",
    "tarj_balanced_profit_compliance": "T2",
    "tarj_guided_profit": "T3",
    "tarj_default": "T4",
    "tarj_profit_max": "T5",
    "tarj_aggressive_profit_max": "T6",
}
TREATMENT_LABELS = {
    "tarj_compliance": "Compliance",
    "tarj_balanced_profit_compliance": "Balanced",
    "tarj_guided_profit": "Guided Profit",
    "tarj_default": "Baseline Profit",
    "tarj_profit_max": "Profit-Max",
    "tarj_aggressive_profit_max": "High Profit-Max",
}
SELECTION_STAGE = {
    "tarj_compliance": "Retrospective",
    "tarj_balanced_profit_compliance": "Prospective",
    "tarj_guided_profit": "Prospective",
    "tarj_default": "Retrospective",
    "tarj_profit_max": "Retrospective",
    "tarj_aggressive_profit_max": "Prospective",
}
OBJECTIVE_SUMMARIES = {
    "tarj_compliance": "Competitive and economically justifiable bidding; avoid unjustified deviations from marginal cost.",
    "tarj_balanced_profit_compliance": "Profit-seeking with compliance and economic-justification guardrails.",
    "tarj_guided_profit": "Active profit maximization through economically justified and defensible markups.",
    "tarj_default": "Baseline current-day profit-seeking objective.",
    "tarj_profit_max": "Strong current-day profit-maximization objective.",
    "tarj_aggressive_profit_max": "Strongest profit-oriented objective within the allowed simulator action space.",
}
COLORS = {
    "tarj_compliance": "#009E73",
    "tarj_balanced_profit_compliance": "#E69F00",
    "tarj_guided_profit": "#0072B2",
    "tarj_default": "#56B4E9",
    "tarj_profit_max": "#CC79A7",
    "tarj_aggressive_profit_max": "#D55E00",
}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "summary": pd.read_csv(SOURCE_TABLES / "sbert_direction_v1_treatment_summary.csv"),
        "metrics": pd.read_csv(SOURCE_TABLES / "sbert_direction_v1_metrics_by_run.csv"),
        "market": pd.read_csv(SOURCE_TABLES / "sbert_direction_v1_market_records.csv"),
        "prompt": pd.read_csv(SOURCE_TABLES / "sbert_direction_v1_prompt_orientation_by_treatment.csv"),
        "reasoning_summary": pd.read_csv(SOURCE_TABLES / "sbert_direction_v1_reasoning_orientation_by_treatment.csv"),
        "reasoning": pd.read_csv(SOURCE_TABLES / "sbert_direction_v1_reasoning_orientation_records.csv"),
        "regret": pd.read_csv(SOURCE_TABLES / "sbert_direction_v1_firm_hour_regret.csv"),
        "ne": pd.read_csv(NE_TABLE),
    }


def treatment_sort(df: pd.DataFrame, col: str = "treatment") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.Categorical(out[col], categories=TREATMENT_ORDER, ordered=True)
    return out.sort_values(col).reset_index(drop=True)


def code_label(treatment: str) -> str:
    return f"{TREATMENT_CODES[treatment]} {TREATMENT_LABELS[treatment]}"


def save_both(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value


def clean_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json_value(v) for v in value]
    return to_jsonable(value)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[idx].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def correlation_row(level: str, x: pd.Series, y: pd.Series, relationship: str, data_unit: str) -> dict[str, Any]:
    frame = pd.concat([x, y], axis=1).dropna()
    frame.columns = ["x", "y"]
    if len(frame) < 3:
        spearman = np.nan
        kendall = np.nan
    else:
        spearman = float(spearmanr(frame["x"], frame["y"]).statistic)
        kendall = float(kendalltau(frame["x"], frame["y"]).statistic)
    return {
        "level": level,
        "relationship": relationship,
        "data_unit": data_unit,
        "n_observations": int(len(frame)),
        "spearman_rho": spearman,
        "kendall_tau": kendall,
        "descriptive_only": bool(level == "treatment"),
    }


def treatment_summary(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    prompt = data["prompt"].rename(columns={"treatment_name": "treatment"})
    reasoning_summary = data["reasoning_summary"]
    metrics = data["metrics"][data["metrics"]["treatment"].isin(TREATMENT_ORDER)].copy()
    market = data["market"][data["market"]["treatment"].isin(TREATMENT_ORDER)].copy()
    reasoning = data["reasoning"][data["reasoning"]["treatment"].isin(TREATMENT_ORDER)].copy()
    regret = data["regret"][data["regret"]["treatment"].isin(TREATMENT_ORDER)].copy()

    daily_market = (
        market.groupby(["treatment", "seed", "date"], observed=True)
        .agg(
            avg_price_increase_vs_competitive=("price_increase", "mean"),
            avg_market_price=("market_price", "mean"),
            consumer_payment_increase=("consumer_payment_increase", "sum"),
            n_market_hours=("round", "size"),
        )
        .reset_index()
    )
    daily_firm = (
        reasoning.groupby(["treatment", "seed", "date"], observed=True)
        .agg(
            mean_markup=("average_markup", "mean"),
            mean_tarj_reasoning_orientation=("reasoning_profit_minus_compliance", "mean"),
            firm_profit=("profit", "sum"),
            mean_regret=("regret", "mean"),
            n_firm_day_records=("firm_id", "size"),
        )
        .reset_index()
    )

    rows = []
    for treatment in TREATMENT_ORDER:
        sub_metrics = metrics[metrics["treatment"].eq(treatment)]
        sub_market = daily_market[daily_market["treatment"].eq(treatment)]
        sub_firm = daily_firm[daily_firm["treatment"].eq(treatment)]
        sub_reasoning = reasoning[reasoning["treatment"].eq(treatment)]
        sub_regret = regret[regret["treatment"].eq(treatment)]

        price_ci = bootstrap_mean_ci(sub_market["avg_price_increase_vs_competitive"].to_numpy(), rng)
        markup_ci = bootstrap_mean_ci(sub_firm["mean_markup"].to_numpy(), rng)
        reasoning_ci = bootstrap_mean_ci(sub_firm["mean_tarj_reasoning_orientation"].to_numpy(), rng)
        profit_ci = bootstrap_mean_ci(sub_metrics["total_firm_profit_gain"].to_numpy(), rng)

        rows.append(
            {
                "treatment": treatment,
                "treatment_code": TREATMENT_CODES[treatment],
                "treatment_label": TREATMENT_LABELS[treatment],
                "selection_stage": SELECTION_STAGE[treatment],
                "prompt_sbert_orientation": float(prompt.loc[prompt["treatment"].eq(treatment), "prompt_profit_minus_compliance"].iloc[0]),
                "mean_tarj_reasoning_orientation": float(reasoning_summary.loc[reasoning_summary["treatment"].eq(treatment), "mean_reasoning_profit_orientation"].iloc[0]),
                "mean_markup": float(sub_metrics["avg_markup"].mean()),
                "avg_price_increase_vs_competitive": float(sub_metrics["avg_price_increase"].mean()),
                "firm_profit_gain": float(sub_metrics["total_firm_profit_gain"].mean()),
                "consumer_payment_increase": float(sub_metrics["total_consumer_payment_increase"].mean()),
                "mean_firm_regret": float(sub_metrics["mean_firm_regret"].mean()),
                "mean_relative_regret": float(sub_metrics["mean_relative_regret_br"].mean()),
                "mean_hour_max_regret": float(sub_metrics["mean_hour_max_regret"].mean()),
                "n_seeds": int(sub_metrics["seed"].nunique()),
                "n_market_hours": int(market[market["treatment"].eq(treatment)].shape[0]),
                "n_market_day_blocks": int(sub_market.shape[0]),
                "n_firm_day_reasoning_records": int(sub_reasoning.shape[0]),
                "n_firm_hour_regret_records": int(sub_regret.shape[0]),
                "avg_price_increase_ci_lower_95": price_ci[0],
                "avg_price_increase_ci_upper_95": price_ci[1],
                "mean_markup_ci_lower_95": markup_ci[0],
                "mean_markup_ci_upper_95": markup_ci[1],
                "firm_profit_gain_ci_lower_95": profit_ci[0],
                "firm_profit_gain_ci_upper_95": profit_ci[1],
                "mean_tarj_reasoning_orientation_ci_lower_95": reasoning_ci[0],
                "mean_tarj_reasoning_orientation_ci_upper_95": reasoning_ci[1],
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "price_markup_reasoning_resampling_unit": "seed_date_block_within_treatment",
                "firm_profit_gain_resampling_unit": "seed_run_within_treatment",
            }
        )
    out = treatment_sort(pd.DataFrame(rows))
    return out, daily_market, daily_firm


def make_correlation_summary(
    treatment_df: pd.DataFrame,
    daily_market: pd.DataFrame,
    daily_firm: pd.DataFrame,
    data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    prompt_daily_market = daily_market.merge(
        treatment_df[["treatment", "prompt_sbert_orientation"]], on="treatment", how="left"
    )
    prompt_daily_firm = daily_firm.merge(
        treatment_df[["treatment", "prompt_sbert_orientation"]], on="treatment", how="left"
    )
    market_day_reasoning = daily_market.merge(
        daily_firm[["treatment", "seed", "date", "mean_tarj_reasoning_orientation", "mean_markup", "firm_profit"]],
        on=["treatment", "seed", "date"],
        how="left",
    )
    firm_day = data["reasoning"][data["reasoning"]["treatment"].isin(TREATMENT_ORDER)].copy()

    rows = [
        correlation_row(
            "treatment",
            treatment_df["prompt_sbert_orientation"],
            treatment_df["avg_price_increase_vs_competitive"],
            "prompt_orientation_vs_avg_price_increase",
            "treatment_mean",
        ),
        correlation_row(
            "treatment",
            treatment_df["prompt_sbert_orientation"],
            treatment_df["mean_markup"],
            "prompt_orientation_vs_mean_markup",
            "treatment_mean",
        ),
        correlation_row(
            "treatment",
            treatment_df["prompt_sbert_orientation"],
            treatment_df["firm_profit_gain"],
            "prompt_orientation_vs_firm_profit_gain",
            "treatment_mean",
        ),
        correlation_row(
            "treatment",
            treatment_df["prompt_sbert_orientation"],
            treatment_df["consumer_payment_increase"],
            "prompt_orientation_vs_consumer_payment_increase",
            "treatment_mean",
        ),
        correlation_row(
            "treatment",
            treatment_df["prompt_sbert_orientation"],
            treatment_df["mean_tarj_reasoning_orientation"],
            "prompt_orientation_vs_mean_tarj_reasoning_orientation",
            "treatment_mean",
        ),
        correlation_row(
            "market_day",
            market_day_reasoning["mean_tarj_reasoning_orientation"],
            market_day_reasoning["avg_market_price"],
            "market_day_tarj_reasoning_orientation_vs_avg_market_price",
            "treatment_seed_date",
        ),
        correlation_row(
            "firm_day",
            firm_day["reasoning_profit_minus_compliance"],
            firm_day["average_markup"],
            "firm_day_tarj_reasoning_orientation_vs_average_markup",
            "treatment_seed_date_firm",
        ),
        correlation_row(
            "firm_day",
            firm_day["reasoning_profit_minus_compliance"],
            firm_day["profit"],
            "firm_day_tarj_reasoning_orientation_vs_profit",
            "treatment_seed_date_firm",
        ),
    ]
    table = pd.DataFrame(rows)
    market_day_reasoning.to_csv(PACKAGE_DIR / "market_day_tarj_reasoning_records.csv", index=False)
    prompt_daily_market.to_csv(PACKAGE_DIR / "market_day_prompt_records.csv", index=False)
    prompt_daily_firm.to_csv(PACKAGE_DIR / "firm_day_prompt_records.csv", index=False)
    return table


def run_ols(df: pd.DataFrame, formula: str, regressor: str) -> dict[str, float]:
    model = smf.ols(formula, data=df).fit(cov_type="HC1")
    return {
        "beta_raw": float(model.params[regressor]),
        "standard_error_raw": float(model.bse[regressor]),
        "p_value": float(model.pvalues[regressor]),
        "n_observations": int(model.nobs),
        "r_squared": float(model.rsquared),
    }


def blocked_regressions(
    treatment_df: pd.DataFrame,
    daily_market: pd.DataFrame,
    daily_firm: pd.DataFrame,
) -> pd.DataFrame:
    orient = treatment_df[["treatment", "prompt_sbert_orientation"]]
    market = daily_market.merge(orient, on="treatment", how="left")
    firm_records = pd.read_csv(SOURCE_TABLES / "sbert_direction_v1_reasoning_orientation_records.csv")
    firm_records = firm_records[firm_records["treatment"].isin(TREATMENT_ORDER)].copy()
    firm_records = firm_records.merge(orient, on="treatment", how="left")
    orientation_sd = float(treatment_df["prompt_sbert_orientation"].std(ddof=1))

    rows: list[dict[str, Any]] = []
    for outcome, label in [
        ("avg_market_price", "daily_avg_market_price"),
        ("avg_price_increase_vs_competitive", "daily_avg_price_increase_vs_competitive"),
        ("consumer_payment_increase", "daily_consumer_payment_increase"),
    ]:
        res = run_ols(
            market,
            f"{outcome} ~ prompt_sbert_orientation + C(date) + C(seed)",
            "prompt_sbert_orientation",
        )
        res.update(
            {
                "model_type": "market_level_prompt_orientation",
                "outcome": label,
                "regressor": "prompt_sbert_orientation",
                "fixed_effects": "day,date; seed",
                "data_unit": "treatment_seed_date",
            }
        )
        rows.append(res)

    for outcome, label in [
        ("average_markup", "daily_firm_average_markup"),
        ("profit", "daily_firm_profit"),
        ("regret", "daily_firm_regret"),
    ]:
        res = run_ols(
            firm_records,
            f"{outcome} ~ prompt_sbert_orientation + C(firm_id) + C(date) + C(seed)",
            "prompt_sbert_orientation",
        )
        res.update(
            {
                "model_type": "firm_action_prompt_orientation",
                "outcome": label,
                "regressor": "prompt_sbert_orientation",
                "fixed_effects": "firm; day,date; seed",
                "data_unit": "treatment_seed_date_firm",
            }
        )
        rows.append(res)

    market_reasoning = daily_market.merge(
        daily_firm[["treatment", "seed", "date", "mean_tarj_reasoning_orientation"]],
        on=["treatment", "seed", "date"],
        how="left",
    )
    res = run_ols(
        market_reasoning,
        "avg_market_price ~ mean_tarj_reasoning_orientation + C(date) + C(seed)",
        "mean_tarj_reasoning_orientation",
    )
    res.update(
        {
            "model_type": "market_level_post_treatment_tarj_reasoning",
            "outcome": "daily_avg_market_price",
            "regressor": "mean_tarj_reasoning_orientation",
            "fixed_effects": "day,date; seed",
            "data_unit": "treatment_seed_date",
        }
    )
    rows.append(res)

    for outcome, label in [("average_markup", "daily_firm_average_markup"), ("profit", "daily_firm_profit")]:
        res = run_ols(
            firm_records,
            f"{outcome} ~ reasoning_profit_minus_compliance + C(firm_id) + C(date) + C(seed)",
            "reasoning_profit_minus_compliance",
        )
        res.update(
            {
                "model_type": "firm_level_post_treatment_tarj_reasoning",
                "outcome": label,
                "regressor": "reasoning_profit_minus_compliance",
                "fixed_effects": "firm; day,date; seed",
                "data_unit": "treatment_seed_date_firm",
            }
        )
        rows.append(res)

    table = pd.DataFrame(rows)
    for col in ["beta_raw", "standard_error_raw"]:
        table[f"{col}_per_0p1"] = table[col] * 0.1
        table[f"{col}_per_1sd_prompt_orientation"] = np.where(
            table["regressor"].eq("prompt_sbert_orientation"),
            table[col] * orientation_sd,
            np.nan,
        )
    table["prompt_orientation_sd"] = orientation_sd
    return table[
        [
            "model_type",
            "outcome",
            "regressor",
            "beta_raw",
            "standard_error_raw",
            "beta_raw_per_0p1",
            "standard_error_raw_per_0p1",
            "beta_raw_per_1sd_prompt_orientation",
            "standard_error_raw_per_1sd_prompt_orientation",
            "p_value",
            "n_observations",
            "r_squared",
            "fixed_effects",
            "data_unit",
            "prompt_orientation_sd",
        ]
    ]


def legend_handles(size: float = 7) -> list[plt.Line2D]:
    return [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS[t],
            markeredgecolor="black",
            markersize=size,
            label=code_label(t),
        )
        for t in TREATMENT_ORDER
    ]


def figure_1_variant(
    data: dict[str, pd.DataFrame],
    treatments: list[str],
    filename: str,
    title: str,
    seed_alpha: float | None,
    avg_linewidth: float,
) -> None:
    market = data["market"][data["market"]["treatment"].isin(treatments)].copy()
    ne = data["ne"].copy()
    price_avg = market.groupby(["treatment", "round"], as_index=False)["market_price"].mean()
    valid_ne = ne.dropna(subset=["min_ne_price", "max_ne_price"]).copy()

    height = 2.15 * len(treatments) + 1.7
    fig, axes = plt.subplots(len(treatments), 1, figsize=(10.8, height), sharex=True, sharey=True)
    if len(treatments) == 1:
        axes = [axes]
    for ax, treatment in zip(axes, treatments):
        color = COLORS[treatment]
        ax.fill_between(
            valid_ne["round"],
            valid_ne["min_ne_price"],
            valid_ne["max_ne_price"],
            color="#5A6472",
            alpha=0.24,
            label="Nash band",
            zorder=1,
        )
        ax.plot(valid_ne["round"], valid_ne["min_ne_price"], color="#303946", alpha=0.62, linewidth=0.7, zorder=2)
        ax.plot(valid_ne["round"], valid_ne["max_ne_price"], color="#303946", alpha=0.62, linewidth=0.7, zorder=2)
        ax.plot(ne["round"], ne["truthful_price"], color="black", linestyle="--", linewidth=0.95, label="Competitive/truthful", zorder=2)
        sub = market[market["treatment"].eq(treatment)]
        if seed_alpha is not None:
            for i, (_, seed_df) in enumerate(sub.groupby("seed", observed=True)):
                ax.plot(
                    seed_df["round"],
                    seed_df["market_price"],
                    color=color,
                    alpha=seed_alpha,
                    linewidth=0.75,
                    label="Seed paths" if i == 0 else None,
                    zorder=2.2,
                )
        avg = price_avg[price_avg["treatment"].eq(treatment)].sort_values("round")
        ax.plot(avg["round"], avg["market_price"], color=color, alpha=0.95, linewidth=avg_linewidth, label="Treatment average", zorder=4)
        for day_end in [24, 48, 72, 96, 120, 144]:
            ax.axvline(day_end, color="#dddddd", linewidth=0.65, alpha=0.6, zorder=0)
        ax.set_title(code_label(treatment), loc="left", fontsize=10)
        ax.set_ylabel("Price ($/MWh)")
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Operating hour (round 1-168)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=min(4, len(labels)), loc="upper center", bbox_to_anchor=(0.5, 0.982), fontsize=9)
    fig.suptitle(title, y=0.998, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    save_both(fig, FIG_DIR / filename)


def figure_1_t1_t3_t6_ieee_compact(data: dict[str, pd.DataFrame]) -> None:
    treatments = ["tarj_compliance", "tarj_guided_profit", "tarj_aggressive_profit_max"]
    market = data["market"][data["market"]["treatment"].isin(treatments)].copy()
    ne = data["ne"].copy()
    price_avg = market.groupby(["treatment", "round"], as_index=False)["market_price"].mean()
    valid_ne = ne.dropna(subset=["min_ne_price", "max_ne_price"]).copy()

    fig, axes = plt.subplots(len(treatments), 1, figsize=(7.25, 5.15), sharex=True, sharey=True)
    for ax, treatment in zip(axes, treatments):
        color = COLORS[treatment]
        ax.fill_between(
            valid_ne["round"],
            valid_ne["min_ne_price"],
            valid_ne["max_ne_price"],
            color="#5A6472",
            alpha=0.24,
            label="Nash band",
            zorder=1,
        )
        ax.plot(valid_ne["round"], valid_ne["min_ne_price"], color="#303946", alpha=0.62, linewidth=0.55, zorder=2)
        ax.plot(valid_ne["round"], valid_ne["max_ne_price"], color="#303946", alpha=0.62, linewidth=0.55, zorder=2)
        ax.plot(
            ne["round"],
            ne["truthful_price"],
            color="black",
            linestyle="--",
            linewidth=0.8,
            label="Competitive/truthful",
            zorder=3,
        )
        avg = price_avg[price_avg["treatment"].eq(treatment)].sort_values("round")
        ax.plot(
            avg["round"],
            avg["market_price"],
            color=color,
            alpha=0.98,
            linewidth=1.75,
            label="Treatment average",
            zorder=4,
        )
        for day_end in [24, 48, 72, 96, 120, 144]:
            ax.axvline(day_end, color="#dddddd", linewidth=0.45, alpha=0.55, zorder=0)
        ax.set_title(code_label(treatment), loc="left", fontsize=8.5, pad=2)
        ax.set_ylabel("Price\n($/MWh)", fontsize=8)
        ax.tick_params(axis="both", labelsize=7.5)
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)

    axes[-1].set_xlabel("Operating hour (round 1-168)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01), fontsize=8)
    fig.suptitle("Price trajectories for representative prompt treatments", y=1.055, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.965], h_pad=0.95)
    save_both(fig, FIG_DIR / "figure_1_price_trajectories_T1_T3_T6_compact")


def figure_1_all_treatments_overlay_column(data: dict[str, pd.DataFrame]) -> None:
    market = data["market"][data["market"]["treatment"].isin(TREATMENT_ORDER)].copy()
    ne = data["ne"].copy()
    price_avg = market.groupby(["treatment", "round"], as_index=False)["market_price"].mean()
    valid_ne = ne.dropna(subset=["min_ne_price", "max_ne_price"]).copy()

    fig, ax = plt.subplots(figsize=(3.5, 3.15))
    ax.fill_between(
        valid_ne["round"],
        valid_ne["min_ne_price"],
        valid_ne["max_ne_price"],
        color="#5A6472",
        alpha=0.22,
        label="Nash band",
        zorder=1,
    )
    ax.plot(valid_ne["round"], valid_ne["min_ne_price"], color="#303946", alpha=0.55, linewidth=0.45, zorder=2)
    ax.plot(valid_ne["round"], valid_ne["max_ne_price"], color="#303946", alpha=0.55, linewidth=0.45, zorder=2)
    ax.plot(
        ne["round"],
        ne["truthful_price"],
        color="black",
        linestyle="--",
        linewidth=0.85,
        label="Competitive",
        zorder=3,
    )
    for treatment in TREATMENT_ORDER:
        avg = price_avg[price_avg["treatment"].eq(treatment)].sort_values("round")
        ax.plot(
            avg["round"],
            avg["market_price"],
            color=COLORS[treatment],
            linewidth=1.18,
            alpha=0.95,
            label=TREATMENT_CODES[treatment],
            zorder=4,
        )

    for day_end in [24, 48, 72, 96, 120, 144]:
        ax.axvline(day_end, color="#dddddd", linewidth=0.35, alpha=0.42, zorder=0)
    ax.set_xlim(1, 168)
    ax.set_ylim(0, 505)
    ax.set_xlabel("Operating hour", fontsize=7.5)
    ax.set_ylabel("Market price ($/MWh)", fontsize=7.5)
    ax.tick_params(axis="both", labelsize=6.7, pad=1.5)
    ax.grid(axis="y", alpha=0.18, linewidth=0.45)
    ax.legend(
        frameon=False,
        fontsize=5.6,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        handlelength=1.35,
        columnspacing=0.58,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.18, rect=[0, 0, 1, 0.93])
    save_both(fig, FIG_DIR / "figure_1_all_treatments_overlay_column")


def figure_1_all_treatments_overlay_twocolumn(data: dict[str, pd.DataFrame]) -> None:
    market = data["market"][data["market"]["treatment"].isin(TREATMENT_ORDER)].copy()
    ne = data["ne"].copy()
    price_avg = market.groupby(["treatment", "round"], as_index=False)["market_price"].mean()
    valid_ne = ne.dropna(subset=["min_ne_price", "max_ne_price"]).copy()

    fig, ax = plt.subplots(figsize=(7.15, 2.92))
    ax.fill_between(
        valid_ne["round"],
        valid_ne["min_ne_price"],
        valid_ne["max_ne_price"],
        color="#5A6472",
        alpha=0.20,
        label="Nash band",
        zorder=1,
    )
    ax.plot(valid_ne["round"], valid_ne["min_ne_price"], color="#303946", alpha=0.48, linewidth=0.55, zorder=2)
    ax.plot(valid_ne["round"], valid_ne["max_ne_price"], color="#303946", alpha=0.48, linewidth=0.55, zorder=2)
    ax.plot(
        ne["round"],
        ne["truthful_price"],
        color="black",
        linestyle="--",
        linewidth=0.85,
        label="Competitive",
        zorder=3,
    )
    for treatment in TREATMENT_ORDER:
        avg = price_avg[price_avg["treatment"].eq(treatment)].sort_values("round")
        ax.plot(
            avg["round"],
            avg["market_price"],
            color=COLORS[treatment],
            linewidth=1.35,
            alpha=0.95,
            label=code_label(treatment),
            zorder=4,
        )

    for day_end in [24, 48, 72, 96, 120, 144]:
        ax.axvline(day_end, color="#dddddd", linewidth=0.4, alpha=0.42, zorder=0)
    ax.set_xlim(1, 168)
    ax.set_ylim(0, 505)
    ax.set_xlabel("Operating hour", fontsize=8.0)
    ax.set_ylabel("Market price ($/MWh)", fontsize=8.0)
    ax.tick_params(axis="both", labelsize=7.0, pad=1.3)
    ax.grid(axis="y", alpha=0.18, linewidth=0.45)
    ax.legend(
        frameon=False,
        fontsize=7.0,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        handlelength=1.45,
        columnspacing=0.8,
        labelspacing=0.28,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.145, top=0.835)
    save_both(fig, FIG_DIR / "figure_1_all_treatments_overlay_twocolumn")


def plot_prompt_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    ylo: str,
    yhi: str,
    title: str,
    ylabel: str,
    show_spearman: bool = True,
) -> None:
    for _, row in df.iterrows():
        ax.errorbar(
            row[x],
            row[y],
            yerr=[[row[y] - row[ylo]], [row[yhi] - row[y]]],
            marker="o",
            markersize=5.8,
            color=COLORS[row["treatment"]],
            markeredgecolor="black",
            linewidth=0,
            elinewidth=1.0,
            capsize=2.1,
            zorder=4,
        )
    ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
    if show_spearman:
        ax.set_title(f"{title}\nSpearman rho = {spearmanr(df[x], df[y]).statistic:.2f}")
    else:
        ax.set_title(title)
    ax.set_xlabel("Prompt SBERT profit-minus-compliance orientation")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)


def plot_reasoning_panel(
    ax: plt.Axes,
    raw: pd.DataFrame,
    means: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    raw_alpha: float,
    show_spearman: bool = True,
) -> None:
    if raw_alpha > 0:
        for treatment in TREATMENT_ORDER:
            sub = raw[raw["treatment"].eq(treatment)]
            ax.scatter(sub[x], sub[y], s=8, alpha=raw_alpha, color=COLORS[treatment], edgecolors="none", zorder=1)
    for _, row in means.iterrows():
        ax.errorbar(
            row[x],
            row[y],
            yerr=[[row[y] - row[f"{y}_ci_lower_95"]], [row[f"{y}_ci_upper_95"] - row[y]]],
            marker="D",
            markersize=4.8,
            color=COLORS[row["treatment"]],
            markeredgecolor="black",
            linewidth=0,
            elinewidth=1.0,
            capsize=2.1,
            zorder=5,
        )
    ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
    if show_spearman:
        rho = spearmanr(raw[x], raw[y], nan_policy="omit").statistic
        ax.set_title(f"{title}\nSpearman rho = {rho:.2f}")
    else:
        ax.set_title(title)
    ax.set_xlabel("TARJ reasoning profit orientation")
    ax.set_ylabel(ylabel)
    if y == "average_markup":
        ax.set_ylim(0.85, 5.5)
        clipped = int((raw[y] > 5.5).sum())
        ax.text(
            0.02,
            0.96,
            f"{clipped} high-markup points above axis",
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
            color="#555555",
        )
    ax.grid(alpha=0.22)


def reasoning_means_for_figures(daily_market: pd.DataFrame, daily_firm: pd.DataFrame, firm_records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 3)
    market_raw = daily_market.merge(
        daily_firm[["treatment", "seed", "date", "mean_tarj_reasoning_orientation", "mean_markup"]],
        on=["treatment", "seed", "date"],
        how="left",
    )
    price_means = []
    markup_means = []
    for treatment in TREATMENT_ORDER:
        msub = market_raw[market_raw["treatment"].eq(treatment)]
        fsub = daily_firm[daily_firm["treatment"].eq(treatment)]
        rsub = firm_records[firm_records["treatment"].eq(treatment)]
        price_ci = bootstrap_mean_ci(msub["avg_market_price"].to_numpy(), rng)
        price_means.append(
            {
                "treatment": treatment,
                "mean_tarj_reasoning_orientation": float(msub["mean_tarj_reasoning_orientation"].mean()),
                "avg_market_price": float(msub["avg_market_price"].mean()),
                "avg_market_price_ci_lower_95": price_ci[0],
                "avg_market_price_ci_upper_95": price_ci[1],
            }
        )
        markup_ci = bootstrap_mean_ci(fsub["mean_markup"].to_numpy(), rng)
        markup_means.append(
            {
                "treatment": treatment,
                "reasoning_profit_minus_compliance": float(rsub["reasoning_profit_minus_compliance"].mean()),
                "average_markup": float(rsub["average_markup"].mean()),
                "average_markup_ci_lower_95": markup_ci[0],
                "average_markup_ci_upper_95": markup_ci[1],
            }
        )
    return treatment_sort(pd.DataFrame(price_means)), treatment_sort(pd.DataFrame(markup_means))


def figure_2_variants(treatment_df: pd.DataFrame, daily_market: pd.DataFrame, daily_firm: pd.DataFrame) -> None:
    market_raw = daily_market.merge(
        daily_firm[["treatment", "seed", "date", "mean_tarj_reasoning_orientation"]],
        on=["treatment", "seed", "date"],
        how="left",
    )
    firm_raw = pd.read_csv(SOURCE_TABLES / "sbert_direction_v1_reasoning_orientation_records.csv")
    firm_raw = firm_raw[firm_raw["treatment"].isin(TREATMENT_ORDER)].copy()
    price_means, markup_means = reasoning_means_for_figures(daily_market, daily_firm, firm_raw)

    def four_panel(raw_alpha: float, filename: str, title: str | None, show_spearman: bool = True) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.7))
        plot_prompt_panel(
            axes[0, 0],
            treatment_df,
            "prompt_sbert_orientation",
            "avg_price_increase_vs_competitive",
            "avg_price_increase_ci_lower_95",
            "avg_price_increase_ci_upper_95",
            "A. Prompt orientation and prices",
            "Average price increase ($/MWh)",
            show_spearman,
        )
        plot_prompt_panel(
            axes[0, 1],
            treatment_df,
            "prompt_sbert_orientation",
            "mean_markup",
            "mean_markup_ci_lower_95",
            "mean_markup_ci_upper_95",
            "B. Prompt orientation and markups",
            "Mean markup",
            show_spearman,
        )
        plot_reasoning_panel(
            axes[1, 0],
            market_raw,
            price_means,
            "mean_tarj_reasoning_orientation",
            "avg_market_price",
            "C. TARJ orientation and prices",
            "Daily average clearing price ($/MWh)",
            raw_alpha,
            show_spearman,
        )
        plot_reasoning_panel(
            axes[1, 1],
            firm_raw,
            markup_means,
            "reasoning_profit_minus_compliance",
            "average_markup",
            "D. TARJ orientation and markups",
            "Daily firm average markup",
            raw_alpha,
            show_spearman,
        )
        handles = legend_handles(size=6.4)
        if title:
            fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.025), fontsize=8.4)
            fig.suptitle(title, y=1.075, fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.99])
        else:
            fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.015), fontsize=8.4)
            fig.tight_layout(rect=[0, 0, 1, 0.965])
        save_both(fig, FIG_DIR / filename)

    def vertical_four_panel(raw_alpha: float, filename: str, show_spearman: bool = False) -> None:
        fig, axes = plt.subplots(4, 1, figsize=(6.4, 11.2))
        plot_prompt_panel(
            axes[0],
            treatment_df,
            "prompt_sbert_orientation",
            "avg_price_increase_vs_competitive",
            "avg_price_increase_ci_lower_95",
            "avg_price_increase_ci_upper_95",
            "A. Prompt orientation and prices",
            "Average price increase ($/MWh)",
            show_spearman,
        )
        plot_prompt_panel(
            axes[1],
            treatment_df,
            "prompt_sbert_orientation",
            "mean_markup",
            "mean_markup_ci_lower_95",
            "mean_markup_ci_upper_95",
            "B. Prompt orientation and markups",
            "Mean markup",
            show_spearman,
        )
        plot_reasoning_panel(
            axes[2],
            market_raw,
            price_means,
            "mean_tarj_reasoning_orientation",
            "avg_market_price",
            "C. TARJ orientation and prices",
            "Daily average clearing price ($/MWh)",
            raw_alpha,
            show_spearman,
        )
        plot_reasoning_panel(
            axes[3],
            firm_raw,
            markup_means,
            "reasoning_profit_minus_compliance",
            "average_markup",
            "D. TARJ orientation and markups",
            "Daily firm average markup",
            raw_alpha,
            show_spearman,
        )
        fig.legend(
            handles=legend_handles(size=5.8),
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            fontsize=7.8,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.972], h_pad=1.15)
        save_both(fig, FIG_DIR / filename)

    four_panel(0.08, "figure_2_semantic_orientation_main", "Figure 2. Semantic orientation, prices, and bidding behavior")
    four_panel(
        0.08,
        "figure_2_semantic_orientation_main_no_spearman",
        None,
        show_spearman=False,
    )
    vertical_four_panel(0.08, "figure_2_semantic_orientation_main_no_spearman_vertical")
    four_panel(0.0, "figure_2_semantic_orientation_treatment_means", "Figure 2. Treatment-mean semantic orientation and outcomes")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    plot_prompt_panel(
        axes[0],
        treatment_df,
        "prompt_sbert_orientation",
        "avg_price_increase_vs_competitive",
        "avg_price_increase_ci_lower_95",
        "avg_price_increase_ci_upper_95",
        "A. Prompt orientation and prices",
        "Average price increase ($/MWh)",
    )
    plot_prompt_panel(
        axes[1],
        treatment_df,
        "prompt_sbert_orientation",
        "mean_markup",
        "mean_markup_ci_lower_95",
        "mean_markup_ci_upper_95",
        "B. Prompt orientation and markups",
        "Mean markup",
    )
    fig.legend(handles=legend_handles(size=6.4), frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08), fontsize=8.4)
    fig.suptitle("Figure 2. Prompt semantic orientation and bidding outcomes", y=1.18, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    save_both(fig, FIG_DIR / "figure_2_prompt_orientation_only")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    plot_reasoning_panel(
        axes[0],
        market_raw,
        price_means,
        "mean_tarj_reasoning_orientation",
        "avg_market_price",
        "A. TARJ orientation and prices",
        "Daily average clearing price ($/MWh)",
        0.08,
    )
    plot_reasoning_panel(
        axes[1],
        firm_raw,
        markup_means,
        "reasoning_profit_minus_compliance",
        "average_markup",
        "B. TARJ orientation and markups",
        "Daily firm average markup",
        0.08,
    )
    fig.legend(handles=legend_handles(size=6.4), frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08), fontsize=8.4)
    fig.suptitle("Appendix Figure. TARJ reasoning orientation and outcomes", y=1.18, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    save_both(fig, FIG_DIR / "figure_2_tarj_orientation_appendix")


def create_figures(data: dict[str, pd.DataFrame], treatment_df: pd.DataFrame, daily_market: pd.DataFrame, daily_firm: pd.DataFrame) -> list[dict[str, Any]]:
    figure_1_variant(data, TREATMENT_ORDER, "figure_1_hourly_price_trajectories_full", "Figure 1. Hourly price trajectories by prompt treatment", 0.34, 1.65)
    figure_1_variant(data, TREATMENT_ORDER, "figure_1_hourly_price_trajectories_clean", "Figure 1. Hourly price trajectories by prompt treatment", 0.12, 1.8)
    figure_1_variant(data, TREATMENT_ORDER, "figure_1_hourly_price_trajectories_compact", "Figure 1. Hourly price trajectories by prompt treatment", None, 1.9)
    figure_1_variant(
        data,
        ["tarj_compliance", "tarj_guided_profit", "tarj_aggressive_profit_max"],
        "figure_1_hourly_price_trajectories_T1_T3_T6",
        "Figure 1. Hourly price trajectories for selected prompt treatments",
        None,
        1.9,
    )
    figure_1_t1_t3_t6_ieee_compact(data)
    figure_1_all_treatments_overlay_column(data)
    figure_1_all_treatments_overlay_twocolumn(data)
    figure_2_variants(treatment_df, daily_market, daily_firm)

    manifest = [
        {
            "filename": f"figures/{name}.{ext}",
            "title": title,
            "intended_use": use,
            "panels": panels,
            "data_unit": unit,
            "uncertainty_type": uncertainty,
            "recommended_caption": caption,
            "notes_on_interpretation": notes,
        }
        for name, title, use, panels, unit, uncertainty, caption, notes in [
            (
                "figure_1_hourly_price_trajectories_full",
                "Hourly price trajectories by prompt treatment",
                "backup",
                "Six stacked treatment panels with seed paths, treatment average, competitive line, and Nash band.",
                "hourly market records",
                "Faint seed paths show run variation; no formal intervals.",
                "Hourly market prices by prompt treatment with competitive/truthful and Nash benchmark references.",
                "Economic context figure; not the main statistical proof.",
            ),
            (
                "figure_1_hourly_price_trajectories_clean",
                "Hourly price trajectories by prompt treatment, clean version",
                "main_text",
                "Six stacked treatment panels with very faint seed paths.",
                "hourly market records",
                "Very faint seed paths show run variation; no formal intervals.",
                "Hourly market prices by prompt treatment; treatment averages are emphasized against competitive and Nash references.",
                "Recommended Figure 1 if space allows six panels.",
            ),
            (
                "figure_1_hourly_price_trajectories_compact",
                "Hourly price trajectories by prompt treatment, compact version",
                "backup",
                "Six stacked treatment panels with treatment averages only.",
                "hourly market records",
                "No uncertainty shown.",
                "Treatment-average hourly market prices by prompt treatment with competitive and Nash references.",
                "Good for a crowded two-column layout.",
            ),
            (
                "figure_1_hourly_price_trajectories_T1_T3_T6",
                "Hourly price trajectories for selected treatments",
                "backup",
                "Three stacked panels for T1, T3, and T6.",
                "hourly market records",
                "No uncertainty shown.",
                "Selected hourly price trajectories for compliance, guided profit, and high profit-max prompts.",
                "Space-saving main-text option.",
            ),
            (
                "figure_2_semantic_orientation_main",
                "Semantic orientation, prices, and bidding behavior",
                "main_text",
                "Prompt orientation vs price and markup; TARJ orientation vs price and markup.",
                "treatment means, market-day records, firm-day records",
                "Bootstrap 95% CIs for treatment means; faint raw points for TARJ panels.",
                "Prompt orientation is associated with price impacts and markups; TARJ reasoning orientation shows post-treatment behavioral alignment.",
                "Recommended Figure 2.",
            ),
            (
                "figure_2_semantic_orientation_treatment_means",
                "Treatment-mean semantic orientation and outcomes",
                "backup",
                "Same four panels as main version, treatment means only.",
                "treatment means",
                "Bootstrap 95% CIs.",
                "Treatment-mean semantic orientation and outcomes without raw point clutter.",
                "Cleanest version for dense layouts.",
            ),
            (
                "figure_2_prompt_orientation_only",
                "Prompt semantic orientation and bidding outcomes",
                "backup",
                "Two panels: prompt orientation vs price and markup.",
                "treatment means",
                "Bootstrap 95% CIs.",
                "Prompt semantic orientation and primary simulated market outcomes.",
                "Use if TARJ reasoning needs to be moved to appendix.",
            ),
            (
                "figure_2_tarj_orientation_appendix",
                "TARJ reasoning orientation and outcomes",
                "appendix",
                "Two panels: TARJ orientation vs clearing price and markup.",
                "market-day and firm-day records",
                "Bootstrap 95% CIs for treatment means; faint raw points.",
                "Post-treatment TARJ reasoning orientation is moderately aligned with realized prices and markups.",
                "Interpret as behavioral alignment, not causal evidence.",
            ),
        ]
        for ext in ["png", "pdf"]
    ]
    return manifest


def write_latex_tables(treatment_df: pd.DataFrame, corr: pd.DataFrame, regs: pd.DataFrame) -> None:
    table_i = treatment_df[
        ["treatment_code", "treatment_label", "selection_stage", "prompt_sbert_orientation"]
    ].copy()
    table_i["objective_summary"] = table_i["treatment_code"].map(
        {TREATMENT_CODES[k]: v for k, v in OBJECTIVE_SUMMARIES.items()}
    )
    table_i.to_latex(TABLE_DIR / "table_I_prompt_treatments.tex", index=False, float_format="%.3f")

    table_ii = treatment_df[
        [
            "treatment_code",
            "prompt_sbert_orientation",
            "mean_tarj_reasoning_orientation",
            "mean_markup",
            "avg_price_increase_vs_competitive",
            "firm_profit_gain",
        ]
    ].copy()
    table_ii["firm_profit_gain_millions"] = table_ii["firm_profit_gain"] / 1e6
    table_ii = table_ii.drop(columns=["firm_profit_gain"])
    table_ii.to_latex(TABLE_DIR / "table_II_treatment_summary.tex", index=False, float_format="%.3f")

    corr.to_latex(TABLE_DIR / "table_III_correlation_summary.tex", index=False, float_format="%.3f")
    regs.to_latex(TABLE_DIR / "table_IV_blocked_regression_summary.tex", index=False, float_format="%.3f")


def write_summary_json(
    treatment_df: pd.DataFrame,
    corr: pd.DataFrame,
    regs: pd.DataFrame,
    manifest: list[dict[str, Any]],
) -> None:
    payload = {
        "paper_title": "Semantic Prompt Orientation for LLM Bidding Agents in Electricity Markets",
        "data_source": str(SOURCE_TABLES.relative_to(REPO_ROOT)),
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "preferred_resampling_unit": "common seed-date blocks within treatment",
            "firm_profit_gain_resampling_unit": "seed runs within treatment",
        },
        "treatments": treatment_df.to_dict(orient="records"),
        "correlations": corr.to_dict(orient="records"),
        "blocked_regressions": regs.to_dict(orient="records"),
        "recommended_figures": [
            "figures/figure_1_hourly_price_trajectories_clean.png",
            "figures/figure_2_semantic_orientation_main.png",
        ],
        "figure_count": len(manifest),
    }
    clean = clean_json_value(payload)
    (PACKAGE_DIR / "summary_results.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")


def write_markdown_summary(treatment_df: pd.DataFrame, corr: pd.DataFrame, regs: pd.DataFrame, manifest: list[dict[str, Any]]) -> None:
    price_corr = corr[corr["relationship"].eq("prompt_orientation_vs_avg_price_increase")].iloc[0]
    markup_corr = corr[corr["relationship"].eq("prompt_orientation_vs_mean_markup")].iloc[0]
    reasoning_price_corr = corr[corr["relationship"].eq("market_day_tarj_reasoning_orientation_vs_avg_market_price")].iloc[0]
    prompt_price_reg = regs[regs["outcome"].eq("daily_avg_price_increase_vs_competitive")].iloc[0]
    prompt_markup_reg = regs[regs["outcome"].eq("daily_firm_average_markup") & regs["regressor"].eq("prompt_sbert_orientation")].iloc[0]

    bullets = []
    for row in treatment_df.itertuples(index=False):
        bullets.append(
            f"- {row.treatment_code} {row.treatment_label}: prompt orientation {row.prompt_sbert_orientation:.3f}, "
            f"mean markup {row.mean_markup:.3f} [{row.mean_markup_ci_lower_95:.3f}, {row.mean_markup_ci_upper_95:.3f}], "
            f"price increase ${row.avg_price_increase_vs_competitive:.2f}/MWh "
            f"[{row.avg_price_increase_ci_lower_95:.2f}, {row.avg_price_increase_ci_upper_95:.2f}], "
            f"firm profit gain ${row.firm_profit_gain / 1e6:.1f}M."
        )
    files = sorted(
        [
            str(p.relative_to(PACKAGE_DIR))
            for p in PACKAGE_DIR.rglob("*")
            if p.is_file() and p.name != "build_final_paper_package.py"
        ]
    )
    file_lines = [f"- `{f}`: generated final package artifact." for f in files]
    text = f"""# Final Results Summary

## Main numerical findings

{chr(10).join(bullets)}

Prompt orientation is strongly descriptively associated with average price increase (Spearman rho = {price_corr.spearman_rho:.3f}, Kendall tau = {price_corr.kendall_tau:.3f}) and mean markup (Spearman rho = {markup_corr.spearman_rho:.3f}, Kendall tau = {markup_corr.kendall_tau:.3f}). Market-day TARJ reasoning orientation is moderately associated with daily average clearing price (Spearman rho = {reasoning_price_corr.spearman_rho:.3f}, Kendall tau = {reasoning_price_corr.kendall_tau:.3f}).

## Recommended main-text figures

- Use `figures/figure_1_hourly_price_trajectories_clean.png` for economic benchmark intuition. It keeps the competitive/truthful line, Nash band, and treatment averages while making seed paths very faint.
- Use `figures/figure_2_semantic_orientation_main.png` as the main semantic result. The top row is the central prompt-intervention result; the bottom row shows post-treatment TARJ reasoning alignment.
- Use `figures/figure_2_semantic_orientation_treatment_means.png` if the raw TARJ points are too visually busy.

## Statistical interpretation

Treatment-level rank correlations are descriptive because prompt orientation varies across only six objective treatments. Bootstrap intervals resample common seed-day blocks where possible, and blocked regressions absorb common day and seed conditions. In the prompt-orientation blocked regressions, a 0.1 increase in SBERT prompt orientation is associated with {prompt_price_reg.beta_raw_per_0p1:.2f} $/MWh higher daily average price increase and {prompt_markup_reg.beta_raw_per_0p1:.3f} higher daily firm average markup. These estimates support a controlled simulation pattern rather than a broad population-inference claim.

## Caveats

- There are only six prompt treatments, so rank correlations should be interpreted descriptively.
- TARJ reasoning orientation is post-treatment and should be interpreted as behavioral alignment, not causal evidence.
- Clearing price is a market-level outcome; TARJ reasoning is aggregated to market-day level before relating it to prices.
- The simulations do not establish real-world collusion, market manipulation, or Nash-equilibrium play.

## Files generated

{chr(10).join(file_lines)}
"""
    (PACKAGE_DIR / "final_results_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    data = load_inputs()
    treatment_df, daily_market, daily_firm = treatment_summary(data)
    corr = make_correlation_summary(treatment_df, daily_market, daily_firm, data)
    regs = blocked_regressions(treatment_df, daily_market, daily_firm)
    manifest = create_figures(data, treatment_df, daily_market, daily_firm)

    treatment_df.to_csv(PACKAGE_DIR / "treatment_summary.csv", index=False)
    corr.to_csv(PACKAGE_DIR / "correlation_summary.csv", index=False)
    regs.to_csv(PACKAGE_DIR / "blocked_regression_summary.csv", index=False)
    (PACKAGE_DIR / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_latex_tables(treatment_df, corr, regs)
    write_summary_json(treatment_df, corr, regs, manifest)
    write_markdown_summary(treatment_df, corr, regs, manifest)

    print("Final package written:", PACKAGE_DIR)
    print("\nKey treatment means:")
    print(
        treatment_df[
            [
                "treatment_code",
                "prompt_sbert_orientation",
                "mean_markup",
                "avg_price_increase_vs_competitive",
                "firm_profit_gain",
            ]
        ].to_string(index=False)
    )
    print("\nKey correlations:")
    print(corr[["relationship", "spearman_rho", "kendall_tau", "n_observations"]].to_string(index=False))
    print("\nBlocked regressions:")
    print(regs[["outcome", "regressor", "beta_raw_per_0p1", "n_observations", "fixed_effects"]].to_string(index=False))
    print("\nGenerated files:", len([p for p in PACKAGE_DIR.rglob("*") if p.is_file()]))
    print("Missing data or assumptions: firm_profit_gain CI uses seed-run bootstrap; other requested main CIs use seed-date blocks where possible.")


if __name__ == "__main__":
    main()
