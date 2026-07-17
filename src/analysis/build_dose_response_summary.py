#!/usr/bin/env python3
"""Build treatment-level dose-response summaries for the NAPS paper."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from scipy import stats


PACKAGE_DIR = Path(__file__).resolve().parent
TREATMENT_SUMMARY = PACKAGE_DIR / "treatment_summary.csv"
OUT_CSV = PACKAGE_DIR / "dose_response_summary.csv"
OUT_TEX = PACKAGE_DIR / "tables" / "table_dose_response_summary.tex"
OUT_MD = PACKAGE_DIR / "dose_response_summary.md"

# 95% two-sided t critical value for df = 4, used because there are six
# treatment-level observations and a two-parameter linear model.
T_CRIT_95_DF4 = 2.7764451051977987

OUTCOMES = [
    (
        "Average price increase vs competitive",
        "avg_price_increase_vs_competitive",
        "$/MWh",
        1.0,
    ),
    ("Mean markup", "mean_markup", "markup factor", 1.0),
    ("Firm profit gain", "firm_profit_gain", "million dollars", 1_000_000.0),
    ("Consumer payment increase", "consumer_payment_increase", "million dollars", 1_000_000.0),
    ("Mean TARJ reasoning orientation", "mean_tarj_reasoning_orientation", "orientation score", 1.0),
]


def fit_simple_ols(x: pd.Series, y: pd.Series) -> dict[str, float]:
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    sxy = float(((x - x_mean) * (y - y_mean)).sum())
    beta = sxy / sxx
    alpha = y_mean - beta * x_mean
    fitted = alpha + beta * x
    resid = y - fitted
    sse = float((resid**2).sum())
    sst = float(((y - y_mean) ** 2).sum())
    df = n - 2
    sigma2 = sse / df
    beta_se = math.sqrt(sigma2 / sxx)
    r2 = 1.0 - sse / sst if sst else float("nan")
    return {
        "alpha": alpha,
        "beta": beta,
        "beta_se": beta_se,
        "r2": r2,
        "n": n,
        "df": df,
    }


def write_latex_table(results: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Treatment-level dose-response estimates}",
        r"\label{tab:dose_response_summary}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Outcome & $\hat{\beta}_{0.1}$ & SE & 95\% CI lower & 95\% CI upper & $R^2$ & $p$ & $N$ \\",
        r"\midrule",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"{row.outcome} & "
            f"{row.beta_per_0p1_orientation:.3f} & "
            f"{row.standard_error:.3f} & "
            f"{row.ci_lower:.3f} & "
            f"{row.ci_upper:.3f} & "
            f"{row.r2:.3f} & "
            f"{row.p_value:.3f} & "
            f"{row.n_treatments} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.5em}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize Notes: Estimates come from treatment-level regressions $Y_t=\alpha+\beta S_t+\epsilon_t$, where $S_t$ is prompt SBERT profit-minus-compliance orientation. Coefficients are scaled to a 0.1 increase in prompt orientation. Standard errors, confidence intervals, and $p$-values use the conventional treatment-level OLS slope standard error across the six treatment means with four residual degrees of freedom.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(results: pd.DataFrame) -> None:
    price = results.loc[
        results["outcome"] == "Average price increase vs competitive"
    ].iloc[0]
    markup = results.loc[results["outcome"] == "Mean markup"].iloc[0]
    profit = results.loc[results["outcome"] == "Firm profit gain"].iloc[0]
    reasoning = results.loc[
        results["outcome"] == "Mean TARJ reasoning orientation"
    ].iloc[0]

    md = f"""# Treatment-Level Dose-Response Summary

Model:

```text
Y_t = alpha + beta S_t + error_t
```

where `S_t` is prompt SBERT profit-minus-compliance orientation for treatment `t`. Coefficients are reported per 0.1 increase in prompt orientation. Confidence intervals use conventional OLS standard errors across the six treatment means with four residual degrees of freedom.

## Results

{results.to_markdown(index=False)}

## Paper-Ready Interpretation

A 0.1 increase in prompt SBERT profit-minus-compliance orientation is associated with a {price.beta_per_0p1_orientation:.2f} $/MWh increase in average price relative to the competitive benchmark (95% CI: {price.ci_lower:.2f}, {price.ci_upper:.2f}) and a {markup.beta_per_0p1_orientation:.3f} increase in mean markup (95% CI: {markup.ci_lower:.3f}, {markup.ci_upper:.3f}). The same treatment-level dose-response relationship is associated with a ${profit.beta_per_0p1_orientation:.2f} million increase in firm profit gain and a {reasoning.beta_per_0p1_orientation:.3f} increase in mean TARJ reasoning orientation per 0.1 orientation units. SBERT measures the semantic intensity of the objective-language intervention; it is not part of the agents' decision process and should not be interpreted as a behavioral mechanism by itself.

## Interpretation Notes

- This is a simple treatment-level descriptive model with six observations.
- The coefficient describes how outcomes vary with prompt semantic orientation across treatments.
- The result supports the paper's dose-response framing, but it should not be presented as broad population inference.
- TARJ reasoning orientation is post-treatment and should be interpreted as behavioral alignment rather than causal evidence.
"""
    OUT_MD.write_text(md, encoding="utf-8")


def main() -> None:
    df = pd.read_csv(TREATMENT_SUMMARY)
    x = df["prompt_sbert_orientation"]
    rows = []
    for outcome_name, column, unit, scale in OUTCOMES:
        fit = fit_simple_ols(x, df[column] / scale)
        beta_per_0p1 = fit["beta"] * 0.1
        se_per_0p1 = fit["beta_se"] * 0.1
        t_stat = beta_per_0p1 / se_per_0p1
        p_value = 2 * stats.t.sf(abs(t_stat), df=fit["df"])
        rows.append(
            {
                "outcome": outcome_name,
                "outcome_column": column,
                "outcome_unit": unit,
                "beta_per_0p1_orientation": beta_per_0p1,
                "standard_error": se_per_0p1,
                "t_statistic": t_stat,
                "p_value": p_value,
                "ci_lower": beta_per_0p1 - T_CRIT_95_DF4 * se_per_0p1,
                "ci_upper": beta_per_0p1 + T_CRIT_95_DF4 * se_per_0p1,
                "r2": fit["r2"],
                "n_treatments": fit["n"],
                "df_residual": fit["df"],
                "ci_method": "ols_t_interval_treatment_means_df4",
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(OUT_CSV, index=False)
    write_latex_table(results)
    write_markdown(results)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_TEX}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
