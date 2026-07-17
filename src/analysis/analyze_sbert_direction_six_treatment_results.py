#!/usr/bin/env python3
"""Analyze the six-treatment SBERT-directed TARJ experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import analyze_sbert_direction_five_treatment_results as base


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NAPS_DIR = PROJECT_ROOT / "NAPS_paper_experiments"

base.OUT_ROOT = NAPS_DIR / "outputs" / "sbert_direction_v2_six_treatment"
base.TABLES_DIR = base.OUT_ROOT / "tables"
base.PLOTS_DIR = base.OUT_ROOT / "plots"
base.REPORTS_DIR = base.OUT_ROOT / "reports"
base.REFERENCE_ROW_PATH = base.TABLES_DIR / "sbert_direction_v2_perfect_competition_reference.csv"

base.TREATMENT_ORDER = [
    "tarj_compliance",
    "tarj_balanced_profit_compliance",
    "tarj_guided_profit",
    "tarj_default",
    "tarj_profit_max",
    "tarj_aggressive_profit_max",
]
base.SUMMARY_ORDER = ["perfect_competition"] + base.TREATMENT_ORDER
base.TREATMENT_LABELS.update(
    {
        "tarj_guided_profit": "Guided Profit",
    }
)
base.TREATMENT_COLORS.update(
    {
        "tarj_guided_profit": "#0072B2",
        "tarj_default": "#56B4E9",
    }
)
base.RUN_TAG_BY_TREATMENT.update(
    {
        "tarj_guided_profit": "sbert_direction_v2",
    }
)

GUIDED_SCREEN_DIR = NAPS_DIR / "SBERT" / "outputs" / "SBERT_prompt_screening_experiment_2_guided_profit"


def prompt_orientation() -> pd.DataFrame:
    exp1_scores = pd.read_csv(base.SBERT_SCREEN_DIR / "SBERT_prompt_screening_experiment_1_objective_scores.csv")
    exp1_selected = pd.read_csv(base.SBERT_SCREEN_DIR / "SBERT_prompt_screening_experiment_1_selected_objectives.csv")
    guided_scores = pd.read_csv(GUIDED_SCREEN_DIR / "SBERT_prompt_screening_experiment_2_guided_profit_objective_scores.csv")
    guided_selected = pd.read_csv(GUIDED_SCREEN_DIR / "SBERT_prompt_screening_experiment_2_guided_profit_selected_objective.csv")

    baseline_names = ["tarj_compliance", "tarj_default", "tarj_profit_max"]
    baseline = exp1_scores[
        exp1_scores["target_category"].eq("existing_baseline")
        & exp1_scores["treatment_name"].isin(baseline_names)
    ]
    exp1_selected_scores = exp1_scores.merge(
        exp1_selected[["treatment_name", "candidate_id"]],
        on=["treatment_name", "candidate_id"],
        how="inner",
    )
    guided_selected_scores = guided_scores.merge(
        guided_selected[["treatment_name", "candidate_id"]],
        on=["treatment_name", "candidate_id"],
        how="inner",
    )
    out = pd.concat([baseline, exp1_selected_scores, guided_selected_scores], ignore_index=True)
    out = out.rename(columns={"profit_minus_compliance": "prompt_profit_minus_compliance"})
    return base.treatment_sort(
        out[
            [
                "treatment_name",
                "candidate_id",
                "profit_similarity",
                "compliance_similarity",
                "prompt_profit_minus_compliance",
            ]
        ],
        "treatment_name",
    )


base.prompt_orientation = prompt_orientation


if __name__ == "__main__":
    base.main()
