#!/usr/bin/env python3
"""Build v3 draft figures and analysis summaries from cached six-treatment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE = REPO_ROOT / "results" / "analysis" / "six-treatment"
TABLES = SOURCE / "tables"
NE_TABLE = REPO_ROOT / "data" / "market" / "refined_ne_summary_by_hour.csv"

OUTPUT_ROOT = REPO_ROOT / "results" / "regenerated"
FIG_DIR = OUTPUT_ROOT / "figures"
APPENDIX_DIR = OUTPUT_ROOT / "figures_appendix"
TABLE_DIR = OUTPUT_ROOT / "tables"
REPORT_DIR = OUTPUT_ROOT / "reports"
LATEX_DIR = OUTPUT_ROOT / "latex_table_mockups"

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
    for path in [FIG_DIR, APPENDIX_DIR, TABLE_DIR, REPORT_DIR, LATEX_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def code_label(treatment: str) -> str:
    return f"{TREATMENT_CODES[treatment]} {TREATMENT_LABELS[treatment]}"


def treatment_sort(df: pd.DataFrame, col: str = "treatment") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.Categorical(out[col], categories=TREATMENT_ORDER, ordered=True)
    return out.sort_values(col).reset_index(drop=True)


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "summary": pd.read_csv(TABLES / "sbert_direction_v1_treatment_summary.csv"),
        "metrics": pd.read_csv(TABLES / "sbert_direction_v1_metrics_by_run.csv"),
        "prompt": pd.read_csv(TABLES / "sbert_direction_v1_prompt_orientation_by_treatment.csv"),
        "reasoning_summary": pd.read_csv(TABLES / "sbert_direction_v1_reasoning_orientation_by_treatment.csv"),
        "reasoning": pd.read_csv(TABLES / "sbert_direction_v1_reasoning_orientation_records.csv"),
        "market": pd.read_csv(TABLES / "sbert_direction_v1_market_records.csv"),
        "regret": pd.read_csv(TABLES / "sbert_direction_v1_firm_hour_regret.csv"),
        "load_quartile": pd.read_csv(TABLES / "sbert_direction_v1_load_quartile_metrics.csv"),
        "retry": pd.read_csv(TABLES / "sbert_direction_v1_retry_fallback_summary.csv"),
        "ne": pd.read_csv(NE_TABLE),
    }


def spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.concat([x, y], axis=1).dropna()
    if len(frame) < 3:
        return np.nan
    return float(frame.iloc[:, 0].rank().corr(frame.iloc[:, 1].rank()))


def pearson(x: pd.Series, y: pd.Series) -> float:
    frame = pd.concat([x, y], axis=1).dropna()
    if len(frame) < 3:
        return np.nan
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1]))


def ci95(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    return float(1.96 * series.std(ddof=1) / np.sqrt(len(series)))


def base_treatment_frame(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = data["summary"][data["summary"]["treatment"].isin(TREATMENT_ORDER)].copy()
    prompt = data["prompt"].rename(columns={"treatment_name": "treatment"})
    reasoning = data["reasoning_summary"].copy()
    out = summary.merge(prompt, on="treatment", how="left").merge(reasoning, on="treatment", how="left")
    out["code"] = out["treatment"].map(TREATMENT_CODES)
    out["label"] = out["treatment"].map(TREATMENT_LABELS)
    return treatment_sort(out)


def treatment_run_metrics(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    metrics = data["metrics"][data["metrics"]["treatment"].isin(TREATMENT_ORDER)].copy()
    prompt = data["prompt"].rename(columns={"treatment_name": "treatment"})
    out = metrics.merge(prompt[["treatment", "prompt_profit_minus_compliance"]], on="treatment", how="left")
    out["code"] = out["treatment"].map(TREATMENT_CODES)
    out["label"] = out["treatment"].map(TREATMENT_LABELS)
    return treatment_sort(out)


def save_both(fig: plt.Figure, path_no_ext: Path) -> None:
    fig.savefig(path_no_ext.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_no_ext.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def legend_handles(marker: str = "o", size: float = 7) -> list[plt.Line2D]:
    return [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            markerfacecolor=COLORS[t],
            markeredgecolor="black",
            markersize=size,
            label=code_label(t),
        )
        for t in TREATMENT_ORDER
    ]


def write_table_i(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    treatment = base_treatment_frame(data)
    table = treatment[["code", "label", "treatment", "prompt_profit_minus_compliance"]].copy()
    table["selection_stage"] = table["treatment"].map(SELECTION_STAGE)
    table["objective_summary"] = table["treatment"].map(OBJECTIVE_SUMMARIES)
    table = table[
        ["code", "label", "selection_stage", "prompt_profit_minus_compliance", "objective_summary"]
    ].rename(
        columns={
            "code": "Treatment",
            "label": "Label",
            "selection_stage": "Stage",
            "prompt_profit_minus_compliance": "SBERT orientation",
            "objective_summary": "Objective summary",
        }
    )
    table.to_csv(TABLE_DIR / "table_I_prompt_treatments.csv", index=False)
    table.to_latex(LATEX_DIR / "table_I_prompt_treatments.tex", index=False, float_format="%.3f")
    return table


def write_treatment_summary_with_ci(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    run = treatment_run_metrics(data)
    reasoning = data["reasoning_summary"][["treatment", "mean_reasoning_profit_orientation"]].copy()
    metrics = [
        ("avg_price_increase", "Avg price increase ($/MWh)"),
        ("avg_markup", "Mean markup"),
        ("total_firm_profit_gain", "Firm profit gain ($)"),
        ("total_consumer_payment_increase", "Consumer payment increase ($)"),
        ("mean_firm_regret", "Mean firm regret"),
        ("mean_relative_regret_br", "Mean relative regret"),
        ("mean_hour_max_regret", "Mean hourly max regret"),
    ]
    rows = []
    for treatment, sub in run.groupby("treatment", observed=True):
        row = {
            "treatment": treatment,
            "code": TREATMENT_CODES[treatment],
            "label": TREATMENT_LABELS[treatment],
            "n_seeds": int(sub["seed"].nunique()),
            "prompt_profit_minus_compliance": float(sub["prompt_profit_minus_compliance"].iloc[0]),
        }
        for col, label in metrics:
            row[f"{label} mean"] = float(sub[col].mean())
            row[f"{label} se"] = float(sub[col].std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else np.nan
            row[f"{label} ci95"] = ci95(sub[col])
        rows.append(row)
    table = pd.DataFrame(rows).merge(reasoning, on="treatment", how="left")
    table = treatment_sort(table)
    table.to_csv(TABLE_DIR / "treatment_summary_with_seed_ci.csv", index=False)

    compact = table[
        [
            "code",
            "label",
            "prompt_profit_minus_compliance",
            "mean_reasoning_profit_orientation",
            "Mean markup mean",
            "Mean markup ci95",
            "Avg price increase ($/MWh) mean",
            "Avg price increase ($/MWh) ci95",
            "Firm profit gain ($) mean",
            "Firm profit gain ($) ci95",
            "Consumer payment increase ($) mean",
            "Consumer payment increase ($) ci95",
        ]
    ].copy()
    compact["Firm profit gain ($M) mean"] = compact.pop("Firm profit gain ($) mean") / 1e6
    compact["Firm profit gain ($M) ci95"] = compact.pop("Firm profit gain ($) ci95") / 1e6
    compact["Consumer payment increase ($M) mean"] = compact.pop("Consumer payment increase ($) mean") / 1e6
    compact["Consumer payment increase ($M) ci95"] = compact.pop("Consumer payment increase ($) ci95") / 1e6
    compact.to_csv(TABLE_DIR / "table_II_treatment_summary_for_overleaf.csv", index=False)
    compact.to_latex(LATEX_DIR / "table_II_treatment_summary.tex", index=False, float_format="%.3f")
    return table


def daily_market_frame(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    market = data["market"][data["market"]["treatment"].isin(TREATMENT_ORDER)].copy()
    daily = (
        market.groupby(["treatment", "seed", "date"], observed=True)
        .agg(
            avg_market_price=("market_price", "mean"),
            avg_price_increase=("price_increase", "mean"),
            total_consumer_payment_increase=("consumer_payment_increase", "sum"),
            avg_load=("load_mw", "mean"),
        )
        .reset_index()
    )
    prompt = data["prompt"].rename(columns={"treatment_name": "treatment"})
    daily = daily.merge(prompt[["treatment", "prompt_profit_minus_compliance"]], on="treatment", how="left")
    return treatment_sort(daily)


def daily_firm_frame(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    reasoning = data["reasoning"][data["reasoning"]["treatment"].isin(TREATMENT_ORDER)].copy()
    prompt = data["prompt"].rename(columns={"treatment_name": "treatment"})
    out = reasoning.merge(prompt[["treatment", "prompt_profit_minus_compliance"]], on="treatment", how="left")
    return treatment_sort(out)


def write_correlation_summary(data: dict[str, pd.DataFrame], treatment_ci: pd.DataFrame) -> pd.DataFrame:
    treatment = treatment_ci.merge(
        data["reasoning_summary"][["treatment", "mean_reasoning_profit_orientation"]],
        on="treatment",
        how="left",
        suffixes=("", "_from_summary"),
    )
    rows = []
    treatment_outcomes = [
        ("Average price increase", "Avg price increase ($/MWh) mean"),
        ("Mean markup", "Mean markup mean"),
        ("Consumer payment increase", "Consumer payment increase ($) mean"),
        ("Firm profit gain", "Firm profit gain ($) mean"),
        ("Mean TARJ reasoning orientation", "mean_reasoning_profit_orientation"),
    ]
    for label, col in treatment_outcomes:
        rows.append(
            {
                "level": "Treatment",
                "n": len(treatment),
                "relationship": f"Prompt orientation vs. {label}",
                "spearman_rho": spearman(treatment["prompt_profit_minus_compliance"], treatment[col]),
                "pearson_r": pearson(treatment["prompt_profit_minus_compliance"], treatment[col]),
            }
        )

    firm_day = daily_firm_frame(data)
    firm_day_outcomes = [
        ("Daily firm average markup", "average_markup"),
        ("Daily firm profit", "profit"),
        ("Daily firm regret", "regret"),
    ]
    for label, col in firm_day_outcomes:
        rows.append(
            {
                "level": "Firm-day",
                "n": int(firm_day[[col, "reasoning_profit_minus_compliance"]].dropna().shape[0]),
                "relationship": f"TARJ reasoning orientation vs. {label}",
                "spearman_rho": spearman(firm_day["reasoning_profit_minus_compliance"], firm_day[col]),
                "pearson_r": pearson(firm_day["reasoning_profit_minus_compliance"], firm_day[col]),
            }
        )

    market_day = daily_market_frame(data)
    reasoning_day = (
        firm_day.groupby(["treatment", "seed", "date"], observed=True)
        .agg(mean_reasoning_profit_minus_compliance=("reasoning_profit_minus_compliance", "mean"))
        .reset_index()
    )
    market_reasoning = market_day.merge(reasoning_day, on=["treatment", "seed", "date"], how="left")
    rows.append(
        {
            "level": "Market-day",
            "n": int(
                market_reasoning[["avg_market_price", "mean_reasoning_profit_minus_compliance"]]
                .dropna()
                .shape[0]
            ),
            "relationship": "Mean TARJ reasoning orientation vs. daily average clearing price",
            "spearman_rho": spearman(
                market_reasoning["mean_reasoning_profit_minus_compliance"],
                market_reasoning["avg_market_price"],
            ),
            "pearson_r": pearson(
                market_reasoning["mean_reasoning_profit_minus_compliance"],
                market_reasoning["avg_market_price"],
            ),
        }
    )

    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "correlation_summary.csv", index=False)
    table.round(3).to_latex(LATEX_DIR / "table_III_correlation_summary.tex", index=False)
    market_reasoning.to_csv(TABLE_DIR / "market_day_reasoning_price_records.csv", index=False)
    return table


def run_fe_regression(df: pd.DataFrame, formula: str, coef_name: str) -> dict[str, float | str]:
    model = smf.ols(formula, data=df).fit(cov_type="HC1")
    return {
        "coef": float(model.params.get(coef_name, np.nan)),
        "std_error": float(model.bse.get(coef_name, np.nan)),
        "p_value": float(model.pvalues.get(coef_name, np.nan)),
        "n": int(model.nobs),
        "r_squared": float(model.rsquared),
    }


def write_blocked_regressions(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    market_day = daily_market_frame(data)
    firm_day = daily_firm_frame(data)
    rows = []
    for outcome, label in [
        ("avg_market_price", "Daily average market price"),
        ("avg_price_increase", "Daily average price increase"),
        ("total_consumer_payment_increase", "Daily consumer payment increase"),
    ]:
        result = run_fe_regression(
            market_day,
            f"{outcome} ~ prompt_profit_minus_compliance + C(date) + C(seed)",
            "prompt_profit_minus_compliance",
        )
        result.update(
            {
                "level": "Market-day",
                "outcome": label,
                "fixed_effects": "date, seed",
                "regressor": "prompt_profit_minus_compliance",
            }
        )
        rows.append(result)

    for outcome, label in [
        ("average_markup", "Daily firm average markup"),
        ("profit", "Daily firm profit"),
        ("regret", "Daily firm regret"),
    ]:
        result = run_fe_regression(
            firm_day,
            f"{outcome} ~ prompt_profit_minus_compliance + C(firm_id) + C(date) + C(seed)",
            "prompt_profit_minus_compliance",
        )
        result.update(
            {
                "level": "Firm-day",
                "outcome": label,
                "fixed_effects": "firm, date, seed",
                "regressor": "prompt_profit_minus_compliance",
            }
        )
        rows.append(result)

    table = pd.DataFrame(rows)[
        ["level", "outcome", "regressor", "coef", "std_error", "p_value", "n", "r_squared", "fixed_effects"]
    ]
    table.to_csv(TABLE_DIR / "blocked_regression_summary.csv", index=False)
    table.round(3).to_latex(LATEX_DIR / "table_IV_blocked_regression_summary.tex", index=False)
    return table


def write_reasoning_alignment(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    treatment = base_treatment_frame(data)
    table = treatment[
        [
            "code",
            "label",
            "treatment",
            "prompt_profit_minus_compliance",
            "mean_reasoning_profit_orientation",
            "sd_reasoning_profit_orientation",
            "n_reasoning_records",
        ]
    ].copy()
    table["prompt_orientation_rank"] = table["prompt_profit_minus_compliance"].rank()
    table["reasoning_orientation_rank"] = table["mean_reasoning_profit_orientation"].rank()
    table["rank_difference"] = table["reasoning_orientation_rank"] - table["prompt_orientation_rank"]
    table.to_csv(TABLE_DIR / "reasoning_alignment_by_treatment.csv", index=False)
    return table


def binned_overall(df: pd.DataFrame, x: str, y: str, bins: int = 12) -> pd.DataFrame:
    frame = df[[x, y]].dropna().copy()
    frame["bin"] = pd.qcut(frame[x], q=min(bins, len(frame)), labels=False, duplicates="drop")
    return (
        frame.groupby("bin", observed=True)
        .agg(x_mean=(x, "mean"), y_mean=(y, "mean"), y_sem=(y, "sem"), n=(y, "size"))
        .reset_index(drop=True)
    )


def treatment_mean_for_reasoning(df: pd.DataFrame, y: str) -> pd.DataFrame:
    return (
        df.groupby("treatment", observed=True)
        .agg(
            x=("reasoning_profit_minus_compliance", "mean"),
            y=(y, "mean"),
            y_sem=(y, "sem"),
            n=(y, "size"),
        )
        .reset_index()
        .pipe(treatment_sort)
    )


def plot_treatment_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    yerr_col: str | None = None,
) -> None:
    for _, row in df.iterrows():
        yerr = row[yerr_col] if yerr_col else None
        ax.errorbar(
            row[x],
            row[y],
            yerr=yerr,
            marker="o",
            markersize=6,
            color=COLORS[row["treatment"]],
            markeredgecolor="black",
            linewidth=0,
            elinewidth=1.0,
            capsize=2.3,
            zorder=3,
        )
    ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nSpearman rho = {spearman(df[x], df[y]):.2f}")
    ax.grid(alpha=0.22)


def figure_1_hourly_price_trajectories(data: dict[str, pd.DataFrame]) -> None:
    market = data["market"][data["market"]["treatment"].isin(TREATMENT_ORDER)].copy()
    ne = data["ne"].copy()
    price_avg = market.groupby(["treatment", "round"], as_index=False)["market_price"].mean()
    valid_ne = ne.dropna(subset=["min_ne_price", "max_ne_price"]).copy()

    fig, axes = plt.subplots(len(TREATMENT_ORDER), 1, figsize=(11.6, 14.0), sharex=True, sharey=True)
    for ax, treatment in zip(axes, TREATMENT_ORDER):
        color = COLORS[treatment]
        ax.fill_between(
            valid_ne["round"],
            valid_ne["min_ne_price"],
            valid_ne["max_ne_price"],
            color="#5A6472",
            alpha=0.24,
            label="Nash band",
            zorder=1.5,
        )
        ax.plot(valid_ne["round"], valid_ne["min_ne_price"], color="#303946", alpha=0.62, linewidth=0.75, zorder=3)
        ax.plot(valid_ne["round"], valid_ne["max_ne_price"], color="#303946", alpha=0.62, linewidth=0.75, zorder=3)
        ax.plot(ne["round"], ne["truthful_price"], color="black", linestyle="--", linewidth=0.95, label="Competitive/truthful", zorder=2)
        sub = market[market["treatment"].eq(treatment)]
        for i, (_, seed_df) in enumerate(sub.groupby("seed", observed=True)):
            ax.plot(
                seed_df["round"],
                seed_df["market_price"],
                color=color,
                alpha=0.34,
                linewidth=0.85,
                label="Seed paths" if i == 0 else None,
                zorder=2.2,
            )
        avg = price_avg[price_avg["treatment"].eq(treatment)].sort_values("round")
        ax.plot(avg["round"], avg["market_price"], color=color, alpha=0.92, linewidth=1.45, label="Treatment average", zorder=4)
        for day_end in [24, 48, 72, 96, 120, 144]:
            ax.axvline(day_end, color="#dddddd", linewidth=0.75, alpha=0.72, zorder=0)
        ax.set_title(code_label(treatment), loc="left", fontsize=10.5)
        ax.set_ylabel("Price ($/MWh)")
        ax.grid(axis="y", alpha=0.22)

    axes[-1].set_xlabel("Operating hour (round 1-168)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.976))
    fig.suptitle("Figure 1. Hourly price trajectories by prompt treatment", y=0.998, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.947])
    save_both(fig, FIG_DIR / "figure_1_hourly_price_trajectories_by_treatment")


def figure_2_semantic_four_panel(data: dict[str, pd.DataFrame], treatment_ci: pd.DataFrame) -> None:
    run_table = treatment_ci.copy()
    run_table["firm_profit_gain_millions"] = run_table["Firm profit gain ($) mean"] / 1e6
    run_table["firm_profit_gain_millions_ci95"] = run_table["Firm profit gain ($) ci95"] / 1e6
    reasoning = daily_firm_frame(data)

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.4))
    plot_treatment_scatter(
        axes[0, 0],
        run_table,
        "prompt_profit_minus_compliance",
        "Avg price increase ($/MWh) mean",
        "Prompt SBERT profit-minus-compliance orientation",
        "Average price increase ($/MWh)",
        "A. Prompt orientation and prices",
        "Avg price increase ($/MWh) ci95",
    )
    plot_treatment_scatter(
        axes[0, 1],
        run_table,
        "prompt_profit_minus_compliance",
        "firm_profit_gain_millions",
        "Prompt SBERT profit-minus-compliance orientation",
        "Firm profit gain ($M)",
        "B. Prompt orientation and firm profit",
        "firm_profit_gain_millions_ci95",
    )

    for ax, y, ylabel, title in [
        (axes[1, 0], "average_markup", "Daily firm average markup", "C. Reasoning orientation and markups"),
        (axes[1, 1], "profit", "Daily firm profit ($)", "D. Reasoning orientation and profit"),
    ]:
        for treatment in TREATMENT_ORDER:
            sub = reasoning[reasoning["treatment"].eq(treatment)]
            ax.scatter(
                sub["reasoning_profit_minus_compliance"],
                sub[y],
                s=9,
                alpha=0.08,
                color=COLORS[treatment],
                edgecolors="none",
                zorder=1,
            )
        means = treatment_mean_for_reasoning(reasoning, y)
        for row in means.itertuples(index=False):
            ax.errorbar(
                row.x,
                row.y,
                yerr=1.96 * row.y_sem if pd.notna(row.y_sem) else None,
                marker="D",
                markersize=4.8,
                color=COLORS[row.treatment],
                markeredgecolor="black",
                linewidth=0,
                elinewidth=1.0,
                capsize=2.2,
                zorder=5,
            )
        ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
        ax.set_xlabel("TARJ reasoning profit orientation")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nSpearman rho = {spearman(reasoning['reasoning_profit_minus_compliance'], reasoning[y]):.2f}")
        if y == "average_markup":
            ax.set_ylim(0.85, 5.5)
            clipped = int((reasoning[y] > 5.5).sum())
            ax.text(
                0.02,
                0.96,
                f"{clipped} high-markup points above axis",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color="#555555",
            )
        ax.grid(alpha=0.22)

    handles = legend_handles(marker="o", size=6.5)
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    fig.suptitle("Figure 2. Semantic orientation and market outcomes", y=1.075, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    save_both(fig, FIG_DIR / "figure_2_semantic_orientation_four_panel")


def figure_2_semantic_six_panel(data: dict[str, pd.DataFrame], treatment_ci: pd.DataFrame) -> None:
    run_table = treatment_ci.copy()
    run_table["firm_profit_gain_millions"] = run_table["Firm profit gain ($) mean"] / 1e6
    run_table["firm_profit_gain_millions_ci95"] = run_table["Firm profit gain ($) ci95"] / 1e6
    reasoning = daily_firm_frame(data)

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.8))

    prompt_panels = [
        (
            "Avg price increase ($/MWh) mean",
            "Avg price increase ($/MWh) ci95",
            "Average price increase ($/MWh)",
            "A. Prompt orientation and prices",
        ),
        (
            "Mean markup mean",
            "Mean markup ci95",
            "Mean markup",
            "B. Prompt orientation and markups",
        ),
        (
            "firm_profit_gain_millions",
            "firm_profit_gain_millions_ci95",
            "Firm profit gain ($M)",
            "C. Prompt orientation and profit",
        ),
    ]
    for ax, (y, yerr, ylabel, title) in zip(axes[0], prompt_panels):
        plot_treatment_scatter(
            ax,
            run_table,
            "prompt_profit_minus_compliance",
            y,
            "Prompt SBERT profit-minus-compliance orientation",
            ylabel,
            title,
            yerr,
        )

    reasoning_panels = [
        ("clearing_price", "Daily average clearing price ($/MWh)", "D. Reasoning orientation and prices"),
        ("average_markup", "Daily firm average markup", "E. Reasoning orientation and markups"),
        ("profit", "Daily firm profit ($)", "F. Reasoning orientation and profit"),
    ]
    for ax, (y, ylabel, title) in zip(axes[1], reasoning_panels):
        for treatment in TREATMENT_ORDER:
            sub = reasoning[reasoning["treatment"].eq(treatment)]
            ax.scatter(
                sub["reasoning_profit_minus_compliance"],
                sub[y],
                s=8,
                alpha=0.075,
                color=COLORS[treatment],
                edgecolors="none",
                zorder=1,
            )
        means = treatment_mean_for_reasoning(reasoning, y)
        for row in means.itertuples(index=False):
            ax.errorbar(
                row.x,
                row.y,
                yerr=1.96 * row.y_sem if pd.notna(row.y_sem) else None,
                marker="D",
                markersize=4.6,
                color=COLORS[row.treatment],
                markeredgecolor="black",
                linewidth=0,
                elinewidth=1.0,
                capsize=2.1,
                zorder=5,
            )
        ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
        ax.set_xlabel("TARJ reasoning profit orientation")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nSpearman rho = {spearman(reasoning['reasoning_profit_minus_compliance'], reasoning[y]):.2f}")
        if y == "average_markup":
            ax.set_ylim(0.85, 5.5)
            clipped = int((reasoning[y] > 5.5).sum())
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

    handles = legend_handles(marker="o", size=6.2)
    fig.legend(handles=handles, frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.015), fontsize=8.2)
    fig.suptitle("Figure 2. Prompt and reasoning semantic orientation across market outcomes", y=1.06, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    save_both(fig, FIG_DIR / "figure_2_semantic_orientation_six_panel")


def figure_2_semantic_compact_v3(data: dict[str, pd.DataFrame], treatment_ci: pd.DataFrame) -> None:
    run_table = treatment_ci.copy()
    reasoning = daily_firm_frame(data)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.7))

    prompt_panels = [
        (
            "Avg price increase ($/MWh) mean",
            "Avg price increase ($/MWh) ci95",
            "Average price increase ($/MWh)",
            "A. Prompt orientation and prices",
        ),
        (
            "Mean markup mean",
            "Mean markup ci95",
            "Mean markup",
            "B. Prompt orientation and markups",
        ),
    ]
    for ax, (y, yerr, ylabel, title) in zip(axes[0], prompt_panels):
        plot_treatment_scatter(
            ax,
            run_table,
            "prompt_profit_minus_compliance",
            y,
            "Prompt SBERT profit-minus-compliance orientation",
            ylabel,
            title,
            yerr,
        )

    reasoning_panels = [
        ("clearing_price", "Daily average clearing price ($/MWh)", "C. Reasoning orientation and prices"),
        ("average_markup", "Daily firm average markup", "D. Reasoning orientation and markups"),
    ]
    for ax, (y, ylabel, title) in zip(axes[1], reasoning_panels):
        for treatment in TREATMENT_ORDER:
            sub = reasoning[reasoning["treatment"].eq(treatment)]
            ax.scatter(
                sub["reasoning_profit_minus_compliance"],
                sub[y],
                s=9,
                alpha=0.08,
                color=COLORS[treatment],
                edgecolors="none",
                zorder=1,
            )
        means = treatment_mean_for_reasoning(reasoning, y)
        for row in means.itertuples(index=False):
            ax.errorbar(
                row.x,
                row.y,
                yerr=1.96 * row.y_sem if pd.notna(row.y_sem) else None,
                marker="D",
                markersize=4.8,
                color=COLORS[row.treatment],
                markeredgecolor="black",
                linewidth=0,
                elinewidth=1.0,
                capsize=2.2,
                zorder=5,
            )
        ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
        ax.set_xlabel("TARJ reasoning profit orientation")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nSpearman rho = {spearman(reasoning['reasoning_profit_minus_compliance'], reasoning[y]):.2f}")
        if y == "average_markup":
            ax.set_ylim(0.85, 5.5)
            clipped = int((reasoning[y] > 5.5).sum())
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

    handles = legend_handles(marker="o", size=6.4)
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.025), fontsize=8.4)
    fig.suptitle("Figure 2. Semantic orientation, prices, and bidding behavior", y=1.075, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    save_both(fig, FIG_DIR / "figure_2_semantic_orientation_compact_v3")


def figure_3_reasoning_behavior(data: dict[str, pd.DataFrame]) -> None:
    reasoning = daily_firm_frame(data)
    panels = [
        ("average_markup", "Daily firm average markup", "A. Bidding behavior"),
        ("clearing_price", "Daily average clearing price ($/MWh)", "B. Market price"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    for ax, (y, ylabel, title) in zip(axes, panels):
        for treatment in TREATMENT_ORDER:
            sub = reasoning[reasoning["treatment"].eq(treatment)]
            ax.scatter(
                sub["reasoning_profit_minus_compliance"],
                sub[y],
                s=11,
                alpha=0.15,
                color=COLORS[treatment],
                edgecolors="none",
            )
        means = treatment_mean_for_reasoning(reasoning, y)
        for row in means.itertuples(index=False):
            ax.errorbar(
                row.x,
                row.y,
                yerr=1.96 * row.y_sem if pd.notna(row.y_sem) else None,
                marker="D",
                markersize=4.8,
                color=COLORS[row.treatment],
                markeredgecolor="black",
                linewidth=0,
                elinewidth=1.0,
                capsize=2.2,
                zorder=6,
            )
        ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
        ax.set_title(f"{title}\nSpearman rho = {spearman(reasoning['reasoning_profit_minus_compliance'], reasoning[y]):.2f}")
        ax.set_xlabel("TARJ reasoning profit orientation")
        ax.set_ylabel(ylabel)
        if y == "average_markup":
            ax.set_ylim(0.85, 5.5)
            clipped = int((reasoning[y] > 5.5).sum())
            ax.text(
                0.02,
                0.96,
                f"{clipped} high-markup points above axis",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color="#555555",
            )
        ax.grid(alpha=0.22)

    handles = legend_handles(marker="o", size=7)
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.05), fontsize=8.5)
    fig.suptitle("Figure 3. TARJ reasoning orientation and realized behavior", y=1.17, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_both(fig, FIG_DIR / "figure_3_reasoning_orientation_behavior")


def simple_bar_plot(df: pd.DataFrame, value: str, ylabel: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(df))
    ax.bar(x, df[value], color=[COLORS[t] for t in df["treatment"]], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([df.loc[i, "code"] for i in range(len(df))])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(handles=legend_handles(size=6.5), frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.16), fontsize=8)
    fig.tight_layout()
    save_both(fig, APPENDIX_DIR / filename)


def appendix_figures(data: dict[str, pd.DataFrame], treatment_ci: pd.DataFrame) -> None:
    df = base_treatment_frame(data)
    simple_bar_plot(df, "avg_markup", "Mean markup", "Average markup by treatment", "appendix_average_markup_by_treatment")
    simple_bar_plot(df, "mean_hour_max_regret", "Mean hourly max regret ($)", "Mean hourly max regret by treatment", "appendix_regret_by_treatment")

    for y, ylabel, filename in [
        ("avg_markup", "Mean markup", "appendix_prompt_orientation_vs_average_markup"),
        ("mean_reasoning_profit_orientation", "Mean TARJ reasoning orientation", "appendix_prompt_orientation_vs_reasoning_orientation"),
    ]:
        fig, ax = plt.subplots(figsize=(6.7, 4.5))
        plot_treatment_scatter(
            ax,
            df,
            "prompt_profit_minus_compliance",
            y,
            "Prompt SBERT profit-minus-compliance orientation",
            ylabel,
            ylabel,
        )
        ax.legend(handles=legend_handles(size=6.5), frameon=False, fontsize=8, loc="best")
        fig.tight_layout()
        save_both(fig, APPENDIX_DIR / filename)

    loadq = data["load_quartile"][data["load_quartile"]["treatment"].isin(TREATMENT_ORDER)].copy()
    for value, ylabel, title, filename in [
        ("avg_markup", "Mean markup", "Load-quartile markup by treatment", "appendix_load_quartile_markup_by_treatment"),
        ("avg_price_increase", "Avg price increase ($/MWh)", "Load-quartile price impact by treatment", "appendix_load_quartile_price_impact_by_treatment"),
    ]:
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        for treatment in TREATMENT_ORDER:
            sub = loadq[loadq["treatment"].eq(treatment)].sort_values("load_quartile")
            ax.plot(
                sub["load_quartile"],
                sub[value],
                marker="o",
                linewidth=1.4,
                color=COLORS[treatment],
                label=code_label(treatment),
            )
        ax.set_xlabel("Load quartile")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        save_both(fig, APPENDIX_DIR / filename)

    retry = data["retry"][data["retry"]["treatment"].isin(TREATMENT_ORDER)].copy()
    retry.to_csv(TABLE_DIR / "retry_fallback_summary.csv", index=False)


def write_report(
    treatment_ci: pd.DataFrame,
    correlations: pd.DataFrame,
    regressions: pd.DataFrame,
    alignment: pd.DataFrame,
) -> None:
    price_corr = correlations.loc[
        correlations["relationship"].eq("Prompt orientation vs. Average price increase")
    ].iloc[0]
    markup_corr = correlations.loc[correlations["relationship"].eq("Prompt orientation vs. Mean markup")].iloc[0]
    reasoning_corr = correlations.loc[
        correlations["relationship"].eq("Prompt orientation vs. Mean TARJ reasoning orientation")
    ].iloc[0]
    firm_markup_corr = correlations.loc[
        correlations["relationship"].eq("TARJ reasoning orientation vs. Daily firm average markup")
    ].iloc[0]
    price_reg = regressions.loc[regressions["outcome"].eq("Daily average price increase")].iloc[0]
    markup_reg = regressions.loc[regressions["outcome"].eq("Daily firm average markup")].iloc[0]

    ordered = alignment[
        [
            "code",
            "label",
            "prompt_profit_minus_compliance",
            "mean_reasoning_profit_orientation",
            "rank_difference",
        ]
    ].copy()
    ordered_text = ordered.to_markdown(index=False, floatfmt=".3f")

    report = f"""# V3 Results Interpretation

This v3 package uses cached six-treatment outputs only. No simulations were rerun.

## Main descriptive results

- Prompt orientation is strongly associated with average price impact across treatments: Spearman rho = {price_corr.spearman_rho:.3f}, Pearson r = {price_corr.pearson_r:.3f}.
- Prompt orientation is also strongly associated with average markup: Spearman rho = {markup_corr.spearman_rho:.3f}, Pearson r = {markup_corr.pearson_r:.3f}.
- Prompt orientation tracks mean TARJ reasoning orientation closely: Spearman rho = {reasoning_corr.spearman_rho:.3f}, Pearson r = {reasoning_corr.pearson_r:.3f}.
- At the firm-day level, TARJ reasoning orientation is positively associated with average markup: Spearman rho = {firm_markup_corr.spearman_rho:.3f}, Pearson r = {firm_markup_corr.pearson_r:.3f}.

## Blocked comparisons

- With day and seed fixed effects, the prompt-orientation coefficient for daily average price increase is {price_reg.coef:.3f} with robust SE {price_reg.std_error:.3f} (N = {int(price_reg.n)}).
- With firm, day, and seed fixed effects, the prompt-orientation coefficient for daily firm average markup is {markup_reg.coef:.3f} with robust SE {markup_reg.std_error:.3f} (N = {int(markup_reg.n)}).

## Reasoning alignment by treatment

{ordered_text}

## Interpretation

The final six-treatment results support the paper's directional story: more profit-oriented objective blocks are associated with more profit-oriented generated TARJ rationales and more aggressive bidding outcomes. The relationship is not perfectly monotonic at every adjacent treatment, especially around Guided Profit and Baseline Profit, but the broader ranking is strong. The blocked regressions preserve the same positive direction after common day and seed conditions are absorbed, which is useful evidence that the result is not just driven by a particular week/day pattern.

The reasoning-language results should remain framed as semantic-behavioral alignment rather than causation. TARJ rationales are generated after the prompt intervention and alongside the bid, so they are informative artifacts of agent posture, not independent causal mechanisms.
"""
    (REPORT_DIR / "v3_results_interpretation.md").write_text(report, encoding="utf-8")


def write_readme() -> None:
    readme = """# Draft Figures and Tables V3

This folder contains final-candidate figures and analysis outputs for the six-treatment TARJ/SBERT draft.

Source data: `NAPS_paper_experiments/outputs/sbert_direction_v2_six_treatment/tables`.
No simulations are rerun by this script.

## Main figures

- `figures/figure_1_hourly_price_trajectories_by_treatment.{png,pdf}`
- `figures/figure_2_semantic_orientation_compact_v3.{png,pdf}`
- `figures/figure_2_semantic_orientation_six_panel.{png,pdf}`
- `figures/figure_2_semantic_orientation_four_panel.{png,pdf}`
- `figures/figure_3_reasoning_orientation_behavior.{png,pdf}`

## Tables and source data

- `tables/treatment_summary_with_seed_ci.csv`
- `tables/table_II_treatment_summary_for_overleaf.csv`
- `tables/correlation_summary.csv`
- `tables/blocked_regression_summary.csv`
- `tables/reasoning_alignment_by_treatment.csv`

## Report

- `reports/v3_results_interpretation.md`
"""
    (SCRIPT_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    data = load_inputs()
    table_i = write_table_i(data)
    treatment_ci = write_treatment_summary_with_ci(data)
    correlations = write_correlation_summary(data, treatment_ci)
    regressions = write_blocked_regressions(data)
    alignment = write_reasoning_alignment(data)

    figure_1_hourly_price_trajectories(data)
    figure_2_semantic_four_panel(data, treatment_ci)
    figure_2_semantic_six_panel(data, treatment_ci)
    figure_2_semantic_compact_v3(data, treatment_ci)
    figure_3_reasoning_behavior(data)
    appendix_figures(data, treatment_ci)

    write_report(treatment_ci, correlations, regressions, alignment)
    write_readme()
    print(f"[write] {FIG_DIR}")
    print(f"[write] {APPENDIX_DIR}")
    print(f"[write] {TABLE_DIR}")
    print(f"[write] {REPORT_DIR}")
    print(f"[write] {LATEX_DIR}")
    print(f"[info] wrote {len(table_i)} prompt-treatment rows")


if __name__ == "__main__":
    main()
