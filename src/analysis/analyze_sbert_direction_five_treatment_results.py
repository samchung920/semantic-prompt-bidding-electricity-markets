#!/usr/bin/env python3
"""Analyze the five-treatment SBERT-directed TARJ experiment.

This combines:
  - revised_v2 baseline runs for the original three TARJ treatments
  - sbert_direction_v1 runs for the two SBERT-selected treatments

Outputs are written under NAPS_paper_experiments/outputs/sbert_direction_v1_five_treatment
so baseline tables/plots are not overwritten.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NAPS_DIR = PROJECT_ROOT / "NAPS_paper_experiments"
RUNS_DIR = NAPS_DIR / "runs" / "raw_outputs"
PROMPT_DIR = NAPS_DIR / "prompts"
OUT_ROOT = NAPS_DIR / "outputs" / "sbert_direction_v1_five_treatment"
TABLES_DIR = OUT_ROOT / "tables"
PLOTS_DIR = OUT_ROOT / "plots"
REPORTS_DIR = OUT_ROOT / "reports"
SBERT_SCREEN_DIR = NAPS_DIR / "SBERT" / "outputs" / "SBERT_prompt_screening_experiment_1"
REFERENCE_ROW_PATH = TABLES_DIR / "sbert_direction_v1_perfect_competition_reference.csv"

CASE_DIR = PROJECT_ROOT / "single_node_summer_hourly_case"
GEN_FILE = CASE_DIR / "generators_thermal_5firm.csv"
NE_TABLE = CASE_DIR / "refined_grid_equilibrium_analysis" / "tables" / "refined_ne_summary_by_hour.csv"
TF_MARKET = CASE_DIR / "truthful_ed" / "truthful_market_summary_hourly_summer_high.csv"
TF_FIRM = CASE_DIR / "truthful_ed" / "truthful_firm_summary_hourly_summer_high.csv"

MODEL = "gpt-4.1"
TEMP = "0p2"
FIRMS = ["Firm_Nuclear", "Firm_Coal", "Firm_CC", "Firm_CT", "Firm_Oil"]
GRID = [1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0]
PRICE_CAP = 1000.0
BACKSTOP_GEN_ID = "SCARCITY_BACKSTOP"
DISPATCH_TOL = 1e-6

TREATMENT_ORDER = [
    "tarj_compliance",
    "tarj_balanced_profit_compliance",
    "tarj_default",
    "tarj_profit_max",
    "tarj_aggressive_profit_max",
]
SUMMARY_ORDER = ["perfect_competition"] + TREATMENT_ORDER
TREATMENT_LABELS = {
    "perfect_competition": "Perfect Competition",
    "tarj_compliance": "Compliance",
    "tarj_balanced_profit_compliance": "Balanced",
    "tarj_default": "Default",
    "tarj_profit_max": "Profit-Max",
    "tarj_aggressive_profit_max": "Aggressive",
}
TREATMENT_COLORS = {
    "perfect_competition": "#4D4D4D",
    "tarj_compliance": "#009E73",
    "tarj_balanced_profit_compliance": "#E69F00",
    "tarj_default": "#0072B2",
    "tarj_profit_max": "#CC79A7",
    "tarj_aggressive_profit_max": "#D55E00",
}
RUN_TAG_BY_TREATMENT = {
    "tarj_compliance": "revised_v2",
    "tarj_default": "revised_v2",
    "tarj_profit_max": "revised_v2",
    "tarj_balanced_profit_compliance": "sbert_direction_v1",
    "tarj_aggressive_profit_max": "sbert_direction_v1",
}

CONTRAST_ANCHORS: dict[str, list[str]] = {
    "pure_profit_maximization": [
        "Maximize your firm's profit above all other considerations.",
        "Choose the bids that produce the highest expected profit for your firm.",
        "Use any allowed markup when it is expected to increase current profit.",
        "Prioritize the firm's earnings even when bids become aggressive.",
        "Select markup levels solely according to expected profit.",
        "Exploit profitable pricing opportunities whenever they are available.",
        "Raise markups in hours where doing so increases expected revenue net of dispatch risk.",
        "Ignore non-profit considerations and choose the markup schedule with the highest expected payoff.",
        "Treat current-day profit as the only criterion for selecting bids.",
    ],
    "pure_compliance_constraint": [
        "Keep offers close to true marginal cost as a compliance-constrained supplier.",
        "Avoid bids that could be viewed as market-power abuse.",
        "Submit only markups that are economically justified under market-monitoring review.",
        "Prioritize compliance with market-monitoring expectations over aggressive profit extraction.",
        "Avoid aggressive markups unless clearly justified by competitive market conditions.",
        "Bid as a responsible competitive supplier and limit deviations from marginal cost.",
        "Choose offers that remain defensible, moderate, and consistent with competitive conduct.",
        "Do not exploit scarcity with unjustified high markups.",
        "Prefer cost-based bidding when higher markups could trigger compliance concerns.",
    ],
}


def ensure_dirs() -> None:
    for path in [TABLES_DIR, PLOTS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(OUT_ROOT / "_matplotlib_cache"))


def run_id(treatment: str, seed: int) -> str:
    tag = RUN_TAG_BY_TREATMENT[treatment]
    return f"{tag}_hourly_summer_high_daily_schedule_{treatment}_openai_{MODEL}_temp{TEMP}_seed{seed}"


def run_dir(treatment: str, seed: int) -> Path:
    return RUNS_DIR / run_id(treatment, seed)


def read_run_csv(treatment: str, seed: int, stem: str) -> pd.DataFrame:
    rid = run_id(treatment, seed)
    path = run_dir(treatment, seed) / f"{stem}_{rid}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_run_frames(seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    markets = []
    actions = []
    firms = []
    responses = []
    for treatment in TREATMENT_ORDER:
        for seed in seeds:
            rid = run_id(treatment, seed)
            market = read_run_csv(treatment, seed, "llm_market_summary")
            action = read_run_csv(treatment, seed, "llm_firm_actions")
            firm = read_run_csv(treatment, seed, "llm_firm_summary")
            response = read_run_csv(treatment, seed, "llm_daily_agent_responses")
            for df in [market, action, firm, response]:
                df["treatment"] = treatment
                df["seed"] = seed
                df["run_tag"] = RUN_TAG_BY_TREATMENT[treatment]
                df["run_id"] = rid
            markets.append(market)
            actions.append(action)
            firms.append(firm)
            responses.append(response)
    return (
        pd.concat(markets, ignore_index=True),
        pd.concat(actions, ignore_index=True),
        pd.concat(firms, ignore_index=True),
        pd.concat(responses, ignore_index=True),
    )


def dispatch_with_offers(gen_df: pd.DataFrame, offer_prices: np.ndarray, load_mw: float) -> tuple[pd.DataFrame, float]:
    offer_arr = np.asarray(offer_prices, dtype=float)
    sort_idx = np.argsort(offer_arr, kind="stable")
    stack = gen_df.iloc[sort_idx].reset_index(drop=True).copy()
    stack_offs = offer_arr[sort_idx]
    dispatch_mw = np.zeros(len(stack), dtype=float)
    remaining = float(load_mw)
    marginal_i = 0
    for i in range(len(stack)):
        if remaining <= DISPATCH_TOL:
            break
        dispatch = min(float(stack.at[i, "pmax_mw"]), remaining)
        dispatch_mw[i] = dispatch
        remaining -= dispatch
        marginal_i = i
    market_price = float(stack_offs[marginal_i])
    stack["dispatch_mw"] = dispatch_mw
    stack["offer_price"] = stack_offs
    stack["market_price"] = market_price
    stack["revenue"] = market_price * dispatch_mw
    stack["production_cost"] = stack["true_mc_per_mwh"].to_numpy(dtype=float) * dispatch_mw
    stack["profit"] = stack["revenue"] - stack["production_cost"]
    if abs(dispatch_mw.sum() - load_mw) > DISPATCH_TOL * 100:
        raise RuntimeError(f"Balance error at load {load_mw}")
    return stack, market_price


def compute_regret(market: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    gen_df = pd.read_csv(GEN_FILE)
    true_mc = gen_df["true_mc_per_mwh"].to_numpy(dtype=float)
    firm_ids = gen_df["firm_id"].to_numpy()
    is_backstop = gen_df["gen_id"].eq(BACKSTOP_GEN_ID).to_numpy()
    rows = []

    action_pivot = (
        actions[actions["firm_id"].isin(FIRMS)]
        .pivot_table(index=["run_id", "treatment", "seed", "round", "date", "hour"], columns="firm_id", values="markup_factor")
        .reset_index()
    )
    for mrow in market.itertuples(index=False):
        actual = action_pivot[action_pivot["run_id"].eq(mrow.run_id) & action_pivot["round"].eq(mrow.round)]
        if actual.empty:
            continue
        actual = actual.iloc[0]
        actual_by_firm = {firm: float(actual[firm]) for firm in FIRMS}
        base_offer = np.minimum(np.array([actual_by_firm.get(f, 1.0) for f in firm_ids]) * true_mc, PRICE_CAP)
        base_offer[is_backstop] = PRICE_CAP
        base_stack, _ = dispatch_with_offers(gen_df, base_offer, float(mrow.load_mw))
        base_profit = base_stack[base_stack["firm_id"].isin(FIRMS)].groupby("firm_id")["profit"].sum().to_dict()

        for firm in FIRMS:
            current_profit = float(base_profit.get(firm, 0.0))
            best_profit = -np.inf
            best_alpha = np.nan
            for alpha in GRID:
                offer = base_offer.copy()
                mask = firm_ids == firm
                offer[mask] = np.minimum(alpha * true_mc[mask], PRICE_CAP)
                offer[is_backstop] = PRICE_CAP
                stack, _ = dispatch_with_offers(gen_df, offer, float(mrow.load_mw))
                profit = float(stack.loc[stack["firm_id"] == firm, "profit"].sum())
                if profit > best_profit + 1e-12:
                    best_profit = profit
                    best_alpha = alpha
            regret = max(best_profit - current_profit, 0.0)
            rows.append(
                {
                    "run_id": mrow.run_id,
                    "run_tag": mrow.run_tag,
                    "treatment": mrow.treatment,
                    "seed": mrow.seed,
                    "round": int(mrow.round),
                    "date": mrow.date,
                    "hour": int(mrow.hour),
                    "firm_id": firm,
                    "current_profit": current_profit,
                    "best_response_profit": best_profit,
                    "best_response_alpha": best_alpha,
                    "regret": regret,
                    "relative_regret_br": regret / max(abs(best_profit), abs(current_profit), 1.0),
                }
            )
    return pd.DataFrame(rows)


def treatment_sort(df: pd.DataFrame, col: str = "treatment") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.Categorical(out[col], categories=TREATMENT_ORDER, ordered=True)
    return out.sort_values(col)


def summary_sort(df: pd.DataFrame, col: str = "treatment") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.Categorical(out[col], categories=SUMMARY_ORDER, ordered=True)
    return out.sort_values(col)


def compute_metrics(market: pd.DataFrame, actions: pd.DataFrame, firm: pd.DataFrame, regret: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    for (treatment, seed, rid), _ in market.groupby(["treatment", "seed", "run_id"], sort=False):
        m = market[market["run_id"] == rid]
        a = actions[actions["run_id"] == rid]
        f = firm[(firm["run_id"] == rid) & (firm["firm_id"].isin(FIRMS))]
        r = regret[regret["run_id"] == rid]
        hourly_max_regret = r.groupby("round")["regret"].max().mean()
        run_rows.append(
            {
                "run_id": rid,
                "run_tag": RUN_TAG_BY_TREATMENT[treatment],
                "treatment": treatment,
                "seed": seed,
                "avg_price_increase": m["price_increase"].mean(),
                "avg_market_price": m["market_price"].mean(),
                "avg_truthful_price": m["truthful_market_price"].mean(),
                "total_consumer_payment_increase": m["consumer_payment_increase"].sum(),
                "avg_markup": a[a["firm_id"].isin(FIRMS)]["markup_factor"].mean(),
                "total_firm_profit": f["firm_profit"].sum(),
                "total_firm_profit_gain": f["firm_profit_gain"].sum(),
                "mean_firm_regret": r["regret"].mean(),
                "mean_relative_regret_br": r["relative_regret_br"].mean(),
                "mean_hour_max_regret": hourly_max_regret,
                "retry_used_rows": int(a["retry_used"].sum()),
                "fallback_used_rows": int(a["fallback_used"].sum()),
                "valid_response_rows": int(a["valid_response"].sum()),
            }
        )
    by_run = treatment_sort(pd.DataFrame(run_rows))
    by_treatment = (
        by_run.groupby("treatment", observed=True, as_index=False)
        .agg(
            n_runs=("run_id", "nunique"),
            avg_price_increase=("avg_price_increase", "mean"),
            avg_market_price=("avg_market_price", "mean"),
            avg_truthful_price=("avg_truthful_price", "mean"),
            avg_consumer_payment_increase=("total_consumer_payment_increase", "mean"),
            avg_markup=("avg_markup", "mean"),
            avg_total_firm_profit=("total_firm_profit", "mean"),
            avg_total_firm_profit_gain=("total_firm_profit_gain", "mean"),
            mean_firm_regret=("mean_firm_regret", "mean"),
            mean_relative_regret_br=("mean_relative_regret_br", "mean"),
            mean_hour_max_regret=("mean_hour_max_regret", "mean"),
            total_retry_used_rows=("retry_used_rows", "sum"),
            total_fallback_used_rows=("fallback_used_rows", "sum"),
            total_valid_response_rows=("valid_response_rows", "sum"),
        )
    )
    hourly_max_regret = (
        regret.groupby(["treatment", "round"], observed=True)["regret"]
        .max()
        .groupby("treatment", observed=True)
        .mean()
        .reset_index(name="mean_hour_max_regret")
    )
    by_treatment = by_treatment.drop(columns=["mean_hour_max_regret"]).merge(hourly_max_regret, on="treatment", how="left")
    return by_run, treatment_sort(by_treatment)


def compute_perfect_competition_reference() -> pd.DataFrame:
    tf_market = pd.read_csv(TF_MARKET).copy()
    tf_firm = pd.read_csv(TF_FIRM).copy()
    if "stress_case" in tf_market.columns:
        tf_market = tf_market[tf_market["stress_case"].eq("high")].copy()
    if "stress_case" in tf_firm.columns:
        tf_firm = tf_firm[tf_firm["stress_case"].eq("high")].copy()
    cached_market = cached_csv(TABLES_DIR / "sbert_direction_v1_market_records.csv")
    if cached_market is not None and not cached_market.empty:
        sim_rounds = set(pd.to_numeric(cached_market["round"], errors="coerce").dropna().astype(int))
        tf_market = tf_market[tf_market["round"].astype(int).isin(sim_rounds)].copy()
        tf_firm = tf_firm[tf_firm["round"].astype(int).isin(sim_rounds)].copy()
    else:
        tf_market = tf_market[tf_market["round"].astype(int).between(1, 168)].copy()
        tf_firm = tf_firm[tf_firm["round"].astype(int).between(1, 168)].copy()
    tf_market["run_id"] = "perfect_competition_truthful_ed"
    tf_market["run_tag"] = "reference"
    tf_market["treatment"] = "perfect_competition"
    tf_market["seed"] = 0
    tf_market["truthful_market_price"] = tf_market["market_price"]
    tf_market["price_increase"] = 0.0
    tf_market["consumer_payment_increase"] = 0.0

    action_rows = []
    for row in tf_market.itertuples(index=False):
        for firm_id in FIRMS:
            action_rows.append(
                {
                    "run_id": "perfect_competition_truthful_ed",
                    "run_tag": "reference",
                    "treatment": "perfect_competition",
                    "seed": 0,
                    "round": int(row.round),
                    "date": row.date,
                    "hour": int(row.hour),
                    "firm_id": firm_id,
                    "markup_factor": 1.0,
                    "valid_response": True,
                    "retry_used": False,
                    "fallback_used": False,
                }
            )
    tf_actions = pd.DataFrame(action_rows)
    tf_regret = compute_regret(tf_market, tf_actions)
    strategic_tf_firm = tf_firm[tf_firm["firm_id"].isin(FIRMS)]
    return pd.DataFrame(
        [
            {
                "treatment": "perfect_competition",
                "n_runs": 0,
                "avg_price_increase": 0.0,
                "avg_market_price": tf_market["market_price"].mean(),
                "avg_truthful_price": tf_market["market_price"].mean(),
                "avg_consumer_payment_increase": 0.0,
                "avg_markup": 1.0,
                "avg_total_firm_profit": strategic_tf_firm["firm_profit"].sum(),
                "avg_total_firm_profit_gain": 0.0,
                "mean_firm_regret": tf_regret["regret"].mean(),
                "mean_relative_regret_br": tf_regret["relative_regret_br"].mean(),
                "total_retry_used_rows": 0,
                "total_fallback_used_rows": 0,
                "total_valid_response_rows": 0,
                "mean_hour_max_regret": tf_regret.groupby("round")["regret"].max().mean(),
            }
        ]
    )


def get_perfect_competition_reference(recompute: bool = False) -> pd.DataFrame:
    if REFERENCE_ROW_PATH.exists() and not recompute:
        return pd.read_csv(REFERENCE_ROW_PATH)

    ref = compute_perfect_competition_reference()
    ref.to_csv(REFERENCE_ROW_PATH, index=False)
    return ref


def add_perfect_competition_reference(treatment_summary: pd.DataFrame, recompute_reference: bool = False) -> pd.DataFrame:
    ref = get_perfect_competition_reference(recompute_reference)
    no_ref = treatment_summary[treatment_summary["treatment"].ne("perfect_competition")].copy()
    return summary_sort(pd.concat([ref, no_ref], ignore_index=True))


def load_quartile_metrics(market: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    market = market.copy()
    market["load_quartile"] = pd.qcut(market["load_mw"], q=4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    hourly_markup = (
        actions[actions["firm_id"].isin(FIRMS)]
        .groupby(["run_id", "round"], as_index=False)["markup_factor"]
        .mean()
        .rename(columns={"markup_factor": "mean_markup"})
    )
    merged = market.merge(hourly_markup, on=["run_id", "round"], how="left")
    out = (
        merged.groupby(["treatment", "load_quartile"], observed=True, as_index=False)
        .agg(
            avg_markup=("mean_markup", "mean"),
            avg_price_increase=("price_increase", "mean"),
            avg_market_price=("market_price", "mean"),
            avg_load_mw=("load_mw", "mean"),
        )
    )
    return treatment_sort(out)


def normalize_text(text: Any) -> str:
    text = "" if text is None or pd.isna(text) else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines).strip())


def safe_parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or pd.isna(value):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            continue
    return {}


def build_reasoning_records(responses: pd.DataFrame, firm: pd.DataFrame, market: pd.DataFrame, regret: pd.DataFrame) -> pd.DataFrame:
    firm_daily = (
        firm[firm["firm_id"].isin(FIRMS)]
        .groupby(["run_id", "date", "firm_id"], as_index=False)
        .agg(profit=("firm_profit", "sum"), average_markup=("markup_factor", "mean"))
    )
    market_daily = (
        market.groupby(["run_id", "date"], as_index=False)
        .agg(clearing_price=("market_price", "mean"), load=("load_mw", "mean"))
    )
    regret_daily = regret.groupby(["run_id", "date", "firm_id"], as_index=False).agg(regret=("regret", "sum"))
    rows = []
    for row in responses.itertuples(index=False):
        payload = safe_parse_jsonish(getattr(row, "parsed_response_json", ""))
        parts = []
        for label, key in [
            ("Thought", "thought"),
            ("Action", "action_summary"),
            ("Reflection", "reflection"),
            ("Journal", "journal_update"),
        ]:
            val = normalize_text(payload.get(key, ""))
            if val:
                parts.append(f"{label}: {val}")
        reasoning_text = normalize_text("\n".join(parts))
        if len(reasoning_text.split()) < 12:
            continue
        rows.append(
            {
                "run_id": row.run_id,
                "run_tag": row.run_tag,
                "treatment": row.treatment,
                "seed": row.seed,
                "date": row.date,
                "firm_id": row.firm_id,
                "valid_response": row.valid_response,
                "retry_used": row.retry_used,
                "fallback_used": row.fallback_used,
                "tarj_reasoning_text": reasoning_text,
            }
        )
    out = pd.DataFrame(rows)
    out = out.merge(firm_daily, on=["run_id", "date", "firm_id"], how="left")
    out = out.merge(market_daily, on=["run_id", "date"], how="left")
    out = out.merge(regret_daily, on=["run_id", "date", "firm_id"], how="left")
    return out


def require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: sentence-transformers") from exc
    return SentenceTransformer


def load_sbert_model(model_name: str, local_files_only: bool):
    SentenceTransformer = require_sentence_transformers()
    if local_files_only:
        return SentenceTransformer(model_name, local_files_only=True)
    try:
        return SentenceTransformer(model_name)
    except Exception as online_exc:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            print(f"[warn] Online model load failed; retrying local cache for {model_name}.")
            return SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            raise online_exc


def encode_texts(texts: list[str], model) -> np.ndarray:
    return np.asarray(model.encode(texts, convert_to_numpy=True, normalize_embeddings=True), dtype=float)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def anchor_centroids(model) -> tuple[list[str], np.ndarray]:
    labels = list(CONTRAST_ANCHORS)
    centroids = []
    for label in labels:
        embeddings = encode_texts(CONTRAST_ANCHORS[label], model)
        centroids.append(normalize_rows(embeddings.mean(axis=0, keepdims=True))[0])
    return labels, np.vstack(centroids)


def prompt_orientation() -> pd.DataFrame:
    scores = pd.read_csv(SBERT_SCREEN_DIR / "SBERT_prompt_screening_experiment_1_objective_scores.csv")
    selected = pd.read_csv(SBERT_SCREEN_DIR / "SBERT_prompt_screening_experiment_1_selected_objectives.csv")
    baseline = scores[scores["target_category"].eq("existing_baseline") & scores["treatment_name"].isin(TREATMENT_ORDER)]
    selected_keys = selected[["treatment_name", "candidate_id"]]
    selected_scores = scores.merge(selected_keys, on=["treatment_name", "candidate_id"], how="inner")
    out = pd.concat([baseline, selected_scores], ignore_index=True)
    out = out.rename(columns={"profit_minus_compliance": "prompt_profit_minus_compliance"})
    return treatment_sort(out[["treatment_name", "candidate_id", "profit_similarity", "compliance_similarity", "prompt_profit_minus_compliance"]], "treatment_name")


def add_reasoning_orientation(reasoning: pd.DataFrame, model_name: str, local_files_only: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = load_sbert_model(model_name, local_files_only)
    labels, anchors = anchor_centroids(model)
    embeddings = encode_texts(reasoning["tarj_reasoning_text"].astype(str).tolist(), model)
    sims = embeddings @ anchors.T
    sim_df = pd.DataFrame(sims, columns=labels)
    scored = pd.concat([reasoning.reset_index(drop=True), sim_df], axis=1)
    scored["reasoning_profit_similarity"] = scored["pure_profit_maximization"]
    scored["reasoning_compliance_similarity"] = scored["pure_compliance_constraint"]
    scored["reasoning_profit_minus_compliance"] = scored["reasoning_profit_similarity"] - scored["reasoning_compliance_similarity"]
    summary = (
        scored.groupby("treatment", as_index=False)
        .agg(
            n_reasoning_records=("reasoning_profit_minus_compliance", "size"),
            mean_reasoning_profit_orientation=("reasoning_profit_minus_compliance", "mean"),
            sd_reasoning_profit_orientation=("reasoning_profit_minus_compliance", "std"),
            mean_average_markup=("average_markup", "mean"),
            mean_clearing_price=("clearing_price", "mean"),
            mean_profit=("profit", "mean"),
            mean_regret=("regret", "mean"),
        )
    )
    return scored, treatment_sort(summary)


def bar_plot(df: pd.DataFrame, value: str, title: str, ylabel: str, filename: str) -> None:
    plot_df = treatment_sort(df)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(
        [TREATMENT_LABELS[t] for t in plot_df["treatment"]],
        plot_df[value],
        color=[TREATMENT_COLORS[t] for t in plot_df["treatment"]],
        alpha=0.88,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def grouped_quartile_plot(quartile: pd.DataFrame, value: str, title: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    labels = ["Q1 low", "Q2", "Q3", "Q4 high"]
    x = np.arange(len(labels))
    width = 0.15
    for i, treatment in enumerate(TREATMENT_ORDER):
        sub = quartile[quartile["treatment"].eq(treatment)].set_index("load_quartile").reindex(labels)
        ax.bar(x + (i - 2) * width, sub[value], width=width, label=TREATMENT_LABELS[treatment], color=TREATMENT_COLORS[treatment], alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def scatter_records(scored: pd.DataFrame, y: str, ylabel: str, filename: str) -> None:
    frame = scored[["reasoning_profit_minus_compliance", y, "treatment"]].copy()
    frame[y] = pd.to_numeric(frame[y], errors="coerce")
    frame = frame.dropna()
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for treatment in TREATMENT_ORDER:
        sub = frame[frame["treatment"].eq(treatment)]
        ax.scatter(sub["reasoning_profit_minus_compliance"], sub[y], s=20, alpha=0.78, color=TREATMENT_COLORS[treatment], label=TREATMENT_LABELS[treatment], edgecolors="none")
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Reasoning profit orientation")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def binned_scatter_records(scored: pd.DataFrame, y: str, ylabel: str, filename: str, bins: int = 8) -> None:
    frame = scored[["reasoning_profit_minus_compliance", y, "treatment"]].copy()
    frame[y] = pd.to_numeric(frame[y], errors="coerce")
    frame = frame.dropna()
    rows = []
    for treatment in TREATMENT_ORDER:
        sub = frame[frame["treatment"].eq(treatment)].copy()
        if len(sub) < 2:
            continue
        try:
            sub["bin"] = pd.qcut(
                sub["reasoning_profit_minus_compliance"],
                q=min(bins, len(sub)),
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            continue
        for bin_id, group in sub.groupby("bin", observed=True):
            if group.empty:
                continue
            rows.append(
                {
                    "treatment": treatment,
                    "bin": int(bin_id),
                    "n": len(group),
                    "mean_reasoning_profit_orientation": group["reasoning_profit_minus_compliance"].mean(),
                    f"mean_{y}": group[y].mean(),
                    f"sem_{y}": group[y].sem() if len(group) > 1 else 0.0,
                    "x_min": group["reasoning_profit_minus_compliance"].min(),
                    "x_max": group["reasoning_profit_minus_compliance"].max(),
                }
            )

    binned = pd.DataFrame(rows)
    binned.to_csv(TABLES_DIR / filename.replace(".png", ".csv"), index=False)
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    if not binned.empty:
        for treatment in TREATMENT_ORDER:
            sub = binned[binned["treatment"].eq(treatment)].sort_values("mean_reasoning_profit_orientation")
            if sub.empty:
                continue
            ax.errorbar(
                sub["mean_reasoning_profit_orientation"],
                sub[f"mean_{y}"],
                yerr=sub[f"sem_{y}"],
                marker="o",
                markersize=5.5,
                linewidth=1.7,
                capsize=2.5,
                alpha=0.86,
                color=TREATMENT_COLORS[treatment],
                label=TREATMENT_LABELS[treatment],
            )
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Reasoning profit orientation, binned within treatment")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def scatter_treatment(df: pd.DataFrame, y: str, ylabel: str, filename: str) -> None:
    plot_df = treatment_sort(df)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for row in plot_df.itertuples(index=False):
        ax.scatter(row.prompt_profit_minus_compliance, getattr(row, y), s=72, alpha=0.85, color=TREATMENT_COLORS[row.treatment], label=TREATMENT_LABELS[row.treatment])
        ax.text(row.prompt_profit_minus_compliance, getattr(row, y), f" {TREATMENT_LABELS[row.treatment]}", va="center", fontsize=9)
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Prompt SBERT profit orientation")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_profile(value: Any) -> dict[str, float]:
    if value is None or pd.isna(value):
        return {}
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): float(v) for k, v in parsed.items()}


def profile_mean(value: Any) -> float | None:
    profile = parse_profile(value)
    vals = [profile[firm] for firm in FIRMS if firm in profile]
    return sum(vals) / len(vals) if vals else None


def load_ne_table() -> pd.DataFrame:
    ne = pd.read_csv(NE_TABLE).copy()
    ne["nash_low_mean_markup"] = ne["lowest_price_ne_profile"].map(profile_mean)
    ne["nash_high_mean_markup"] = ne["highest_price_ne_profile"].map(profile_mean)
    ne["nash_band_min"] = ne[["nash_low_mean_markup", "nash_high_mean_markup"]].min(axis=1)
    ne["nash_band_max"] = ne[["nash_low_mean_markup", "nash_high_mean_markup"]].max(axis=1)
    return ne


def save_naps_summary_table(treatment_summary: pd.DataFrame) -> None:
    cols = [
        "avg_price_increase",
        "avg_total_firm_profit",
        "avg_total_firm_profit_gain",
        "mean_firm_regret",
        "mean_hour_max_regret",
        "mean_relative_regret_br",
    ]
    fig, ax = plt.subplots(figsize=(15.5, 3.2))
    ax.axis("off")
    table_df = summary_sort(treatment_summary)[["treatment"] + cols].copy()
    table_df["treatment"] = table_df["treatment"].map(TREATMENT_LABELS)
    for col in cols:
        table_df[col] = table_df[col].map(lambda x: f"{x:,.2f}")
    tbl = ax.table(
        cellText=table_df.values.tolist(),
        colLabels=[
            "Treatment",
            "Avg price\nincrease",
            "Avg total\nfirm profit",
            "Avg profit\ngain",
            "Mean firm\nregret",
            "Mean hourly\nmax regret",
            "Mean rel.\nregret",
        ],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.45)
    ax.set_title("NAPS TARJ treatment summary table: SBERT direction v1 five-treatment set", pad=18)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sbert_direction_v1_naps_tarj_metrics_summary_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_paper_main_results_table(
    treatment_summary: pd.DataFrame,
    prompt_scores: pd.DataFrame,
    reasoning_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    prompt = prompt_scores.rename(columns={"treatment_name": "treatment"})
    cols = ["treatment", "prompt_profit_minus_compliance"]
    paper = treatment_summary.merge(prompt[cols], on="treatment", how="left")

    if reasoning_summary is not None and not reasoning_summary.empty:
        reasoning = reasoning_summary[["treatment", "mean_reasoning_profit_orientation"]].copy()
        paper = paper.merge(reasoning, on="treatment", how="left")
    else:
        paper["mean_reasoning_profit_orientation"] = np.nan

    paper = summary_sort(paper)
    paper = paper[
        [
            "treatment",
            "n_runs",
            "prompt_profit_minus_compliance",
            "mean_reasoning_profit_orientation",
            "avg_markup",
            "avg_price_increase",
            "avg_consumer_payment_increase",
            "avg_total_firm_profit_gain",
        ]
    ].copy()
    paper["avg_consumer_payment_increase_millions"] = paper["avg_consumer_payment_increase"] / 1e6
    paper["avg_total_firm_profit_gain_millions"] = paper["avg_total_firm_profit_gain"] / 1e6
    paper = paper.drop(columns=["avg_consumer_payment_increase", "avg_total_firm_profit_gain"])
    paper.to_csv(TABLES_DIR / "sbert_direction_v1_paper_main_results_table.csv", index=False)

    display = paper.copy()
    display["treatment"] = display["treatment"].map(TREATMENT_LABELS)
    display = display.rename(
        columns={
            "treatment": "Treatment",
            "n_runs": "Runs",
            "prompt_profit_minus_compliance": "Prompt\norientation",
            "mean_reasoning_profit_orientation": "Reasoning\norientation",
            "avg_markup": "Mean\nmarkup",
            "avg_price_increase": "Price\nincrease ($/MWh)",
            "avg_consumer_payment_increase_millions": "Consumer payment\nincrease ($M)",
            "avg_total_firm_profit_gain_millions": "Firm profit\ngain ($M)",
        }
    )
    for col in ["Prompt\norientation", "Reasoning\norientation"]:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    display["Mean\nmarkup"] = display["Mean\nmarkup"].map(lambda x: f"{x:.2f}")
    display["Price\nincrease ($/MWh)"] = display["Price\nincrease ($/MWh)"].map(lambda x: f"{x:.2f}")
    display["Consumer payment\nincrease ($M)"] = display["Consumer payment\nincrease ($M)"].map(lambda x: f"{x:.1f}")
    display["Firm profit\ngain ($M)"] = display["Firm profit\ngain ($M)"].map(lambda x: f"{x:.1f}")

    fig, ax = plt.subplots(figsize=(14.5, 3.35))
    ax.axis("off")
    tbl = ax.table(
        cellText=display.values.tolist(),
        colLabels=list(display.columns),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.7)
    tbl.scale(1, 1.5)
    ax.set_title("Paper Table: Prompt Orientation and Market Outcomes in the TARJ Bidding Experiment", pad=18)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sbert_direction_v1_paper_main_results_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return paper


def save_naps_price_increase_by_run(by_run: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    positions = {t: i for i, t in enumerate(TREATMENT_ORDER)}
    for treatment in TREATMENT_ORDER:
        grp = by_run[by_run["treatment"].eq(treatment)]
        xs = np.full(len(grp), positions[treatment], dtype=float)
        offsets = np.linspace(-0.12, 0.12, len(grp)) if len(grp) else []
        ax.scatter(
            xs + offsets,
            grp["avg_price_increase"],
            s=58,
            color=TREATMENT_COLORS[treatment],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.84,
        )
        ax.hlines(
            grp["avg_price_increase"].mean(),
            positions[treatment] - 0.24,
            positions[treatment] + 0.24,
            color="black",
            linewidth=2,
        )
    ax.set_xticks(list(positions.values()), [TREATMENT_LABELS[t] for t in TREATMENT_ORDER], rotation=20)
    ax.set_ylabel("Average price increase ($/MWh)")
    ax.set_title("NAPS TARJ runs: price increase by treatment and seed")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sbert_direction_v1_naps_tarj_price_increase_by_run.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_naps_consumer_payment(treatment_summary: pd.DataFrame) -> None:
    plot_df = summary_sort(treatment_summary)
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    ax.bar(
        [TREATMENT_LABELS[t] for t in plot_df["treatment"]],
        plot_df["avg_consumer_payment_increase"] / 1e6,
        color=[TREATMENT_COLORS[t] for t in plot_df["treatment"]],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.88,
    )
    ax.set_ylabel("Average consumer payment increase ($M)")
    ax.set_title("NAPS TARJ runs: consumer payment increase by treatment")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sbert_direction_v1_naps_tarj_consumer_payment_by_treatment.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_naps_daily_price_increase(market: pd.DataFrame) -> None:
    daily = (
        market.assign(date=pd.to_datetime(market["date"]).dt.date)
        .groupby(["treatment", "seed", "date"], as_index=False)["price_increase"]
        .mean()
        .groupby(["treatment", "date"], as_index=False)["price_increase"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    for treatment in TREATMENT_ORDER:
        sub = daily[daily["treatment"].eq(treatment)].sort_values("date")
        ax.plot(
            sub["date"],
            sub["price_increase"],
            marker="o",
            linewidth=1.8,
            color=TREATMENT_COLORS[treatment],
            label=TREATMENT_LABELS[treatment],
        )
    ax.set_ylabel("Daily average price increase ($/MWh)")
    ax.set_title("NAPS TARJ runs: daily price increase averaged across seeds")
    ax.legend(frameon=False, ncol=3)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sbert_direction_v1_naps_tarj_daily_price_increase.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_naps_weekly_price_trajectories(market: pd.DataFrame, ne: pd.DataFrame) -> None:
    daily = (
        market.assign(date=pd.to_datetime(market["date"]).dt.date)
        .groupby(["treatment", "seed", "date"], as_index=False)["market_price"]
        .mean()
        .groupby(["treatment", "date"], as_index=False)["market_price"]
        .mean()
    )
    daily_truthful = (
        ne.assign(date=pd.to_datetime(ne["datetime"]).dt.date)
        .groupby("date", as_index=False)["truthful_price"]
        .mean()
    )
    valid_ne = ne.dropna(subset=["min_ne_price", "max_ne_price"]).copy()
    daily_ne = (
        valid_ne.assign(date=pd.to_datetime(valid_ne["datetime"]).dt.date)
        .groupby("date", as_index=False)
        .agg(min_ne_price=("min_ne_price", "mean"), max_ne_price=("max_ne_price", "mean"))
    )

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    if not daily_ne.empty:
        ax.fill_between(daily_ne["date"], daily_ne["min_ne_price"], daily_ne["max_ne_price"], color="#5A6472", alpha=0.22, label="Nash band")
    ax.plot(daily_truthful["date"], daily_truthful["truthful_price"], color="black", linestyle="--", linewidth=1.0, label="Competitive/truthful")
    for treatment in TREATMENT_ORDER:
        sub = daily[daily["treatment"].eq(treatment)].sort_values("date")
        ax.plot(sub["date"], sub["market_price"], marker="o", linewidth=1.8, color=TREATMENT_COLORS[treatment], label=TREATMENT_LABELS[treatment])
    ax.set_ylabel("Daily average market price ($/MWh)")
    ax.set_title("NAPS TARJ runs: weekly price trajectories")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sbert_direction_v1_naps_tarj_weekly_price_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_naps_hourly_regret_profit(regret: pd.DataFrame) -> None:
    hourly = (
        regret.groupby(["treatment", "round"], observed=True)
        .agg(
            max_firm_regret=("regret", "max"),
            total_current_profit=("current_profit", "sum"),
        )
        .reset_index()
    )
    hourly.to_csv(TABLES_DIR / "sbert_direction_v1_naps_tarj_metrics_by_treatment_hour.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    for treatment in TREATMENT_ORDER:
        sub = hourly[hourly["treatment"].eq(treatment)].sort_values("round")
        ax.plot(
            sub["round"],
            sub["max_firm_regret"],
            linewidth=1.8,
            marker="o",
            markersize=2.5,
            color=TREATMENT_COLORS[treatment],
            label=TREATMENT_LABELS[treatment],
        )
    ax.set_xlabel("Operating hour (round 1-168)")
    ax.set_ylabel("Max firm regret ($/hr)")
    ax.set_title("NAPS TARJ runs: hourly max regret by treatment")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sbert_direction_v1_naps_tarj_hourly_max_regret_by_treatment.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    for treatment in TREATMENT_ORDER:
        sub = hourly[hourly["treatment"].eq(treatment)].sort_values("round")
        ax.plot(
            sub["round"],
            sub["total_current_profit"] / 1e6,
            linewidth=1.8,
            marker="o",
            markersize=2.5,
            color=TREATMENT_COLORS[treatment],
            label=TREATMENT_LABELS[treatment],
        )
    ax.set_xlabel("Operating hour (round 1-168)")
    ax.set_ylabel("Total firm profit ($M/hr)")
    ax.set_title("NAPS TARJ runs: hourly total firm profit by treatment")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sbert_direction_v1_naps_tarj_hourly_total_profit_by_treatment.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_naps_markup_trajectories(actions: pd.DataFrame, ne: pd.DataFrame) -> None:
    daily = (
        actions[actions["firm_id"].isin(FIRMS)]
        .assign(date=pd.to_datetime(actions["date"]).dt.date)
        .groupby(["treatment", "seed", "date", "firm_id"], as_index=False)["markup_factor"]
        .mean()
        .groupby(["treatment", "date", "firm_id"], as_index=False)["markup_factor"]
        .mean()
    )
    ne_rows = []
    for row in ne.itertuples(index=False):
        date = pd.to_datetime(row.datetime).date()
        low = parse_profile(row.lowest_price_ne_profile)
        high = parse_profile(row.highest_price_ne_profile)
        for firm in FIRMS:
            if firm in low and firm in high:
                ne_rows.append({"date": date, "round": row.round, "firm_id": firm, "low": low[firm], "high": high[firm]})
    bench = pd.DataFrame(ne_rows)
    daily_bench = bench.groupby(["date", "firm_id"], as_index=False).agg(low=("low", "mean"), high=("high", "mean"))

    fig, axes = plt.subplots(len(FIRMS), 1, figsize=(10.8, 13.0), sharex=True)
    for ax, firm in zip(axes, FIRMS):
        for treatment in TREATMENT_ORDER:
            sub = daily[daily["treatment"].eq(treatment) & daily["firm_id"].eq(firm)].sort_values("date")
            ax.plot(sub["date"], sub["markup_factor"], marker="o", linewidth=1.7, color=TREATMENT_COLORS[treatment], label=TREATMENT_LABELS[treatment])
        b = daily_bench[daily_bench["firm_id"].eq(firm)].sort_values("date")
        if not b.empty:
            ax.fill_between(b["date"], b["low"], b["high"], color="#5A6472", alpha=0.22, label="Nash band" if firm == FIRMS[0] else None)
            ax.plot(b["date"], b["low"], color="#303946", alpha=0.7, linewidth=0.8)
            ax.plot(b["date"], b["high"], color="#303946", alpha=0.7, linewidth=0.8)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Competitive" if firm == FIRMS[0] else None)
        ax.set_ylabel(firm.replace("Firm_", ""))
        ax.grid(alpha=0.25)
    axes[0].set_title("NAPS TARJ runs: weekly bidding trajectories with competitive and Nash benchmarks")
    axes[-1].set_xlabel("Operating date")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(PLOTS_DIR / "sbert_direction_v1_naps_tarj_weekly_markup_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    hourly = (
        actions[actions["firm_id"].isin(FIRMS)]
        .groupby(["treatment", "seed", "round", "firm_id"], as_index=False)["markup_factor"]
        .mean()
        .groupby(["treatment", "round", "firm_id"], as_index=False)["markup_factor"]
        .mean()
    )
    fig, axes = plt.subplots(len(FIRMS), 1, figsize=(12, 13.5), sharex=True)
    for ax, firm in zip(axes, FIRMS):
        for treatment in TREATMENT_ORDER:
            sub = hourly[hourly["treatment"].eq(treatment) & hourly["firm_id"].eq(firm)].sort_values("round")
            ax.plot(sub["round"], sub["markup_factor"], marker="o", markersize=2.5, linewidth=1.45, color=TREATMENT_COLORS[treatment], label=TREATMENT_LABELS[treatment])
        b = bench[bench["firm_id"].eq(firm)].sort_values("round")
        if not b.empty:
            ax.fill_between(b["round"], b["low"], b["high"], color="#5A6472", alpha=0.22, label="Nash band" if firm == FIRMS[0] else None)
            ax.plot(b["round"], b["low"], color="#303946", alpha=0.7, linewidth=0.8)
            ax.plot(b["round"], b["high"], color="#303946", alpha=0.7, linewidth=0.8)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Competitive" if firm == FIRMS[0] else None)
        for day_end in [24, 48, 72, 96, 120, 144]:
            ax.axvline(day_end, color="#dddddd", linewidth=0.8, alpha=0.7)
        ax.set_ylabel(firm.replace("Firm_", ""))
        ax.grid(alpha=0.25)
    axes[0].set_title("NAPS TARJ runs: hourly bidding trajectories with competitive and Nash benchmarks")
    axes[-1].set_xlabel("Operating hour (round 1-168)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(PLOTS_DIR / "sbert_direction_v1_naps_tarj_hourly_markup_trajectories.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_naps_market_price_vs_ne(market: pd.DataFrame, ne: pd.DataFrame) -> None:
    avg = market.groupby(["treatment", "round"], as_index=False)["market_price"].mean()
    fig, axes = plt.subplots(len(TREATMENT_ORDER), 1, figsize=(11.5, 14.0), sharex=True, sharey=True)
    for ax, treatment in zip(axes, TREATMENT_ORDER):
        sub = avg[avg["treatment"].eq(treatment)].sort_values("round")
        valid_ne = ne.dropna(subset=["min_ne_price", "max_ne_price"])
        ax.fill_between(valid_ne["round"], valid_ne["min_ne_price"], valid_ne["max_ne_price"], color="#5A6472", alpha=0.25, label="Nash band")
        ax.plot(valid_ne["round"], valid_ne["min_ne_price"], color="#303946", alpha=0.68, linewidth=0.85)
        ax.plot(valid_ne["round"], valid_ne["max_ne_price"], color="#303946", alpha=0.68, linewidth=0.85)
        ax.plot(ne["round"], ne["truthful_price"], color="black", linestyle="--", linewidth=1.0, label="Competitive truthful")
        ax.plot(sub["round"], sub["market_price"], color=TREATMENT_COLORS[treatment], linewidth=1.7, label="TARJ price")
        ax.set_title(TREATMENT_LABELS[treatment], loc="left")
        ax.set_ylabel("Price ($/MWh)")
        ax.grid(alpha=0.24)
    axes[-1].set_xlabel("Operating hour (round 1-168)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("NAPS TARJ runs: market price vs competitive and Nash benchmarks", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(PLOTS_DIR / "sbert_direction_v1_naps_tarj_market_price_vs_ne_band.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_naps_seed_trajectories(market: pd.DataFrame, actions: pd.DataFrame, ne: pd.DataFrame) -> None:
    price_avg = market.groupby(["treatment", "round"], as_index=False)["market_price"].mean()
    fig, axes = plt.subplots(len(TREATMENT_ORDER), 1, figsize=(12, 14.0), sharex=True, sharey=True)
    valid_ne = ne.dropna(subset=["min_ne_price", "max_ne_price"])
    for ax, treatment in zip(axes, TREATMENT_ORDER):
        color = TREATMENT_COLORS[treatment]
        ax.fill_between(valid_ne["round"], valid_ne["min_ne_price"], valid_ne["max_ne_price"], color="#5A6472", alpha=0.25, label="Nash band", zorder=1.8)
        ax.plot(valid_ne["round"], valid_ne["min_ne_price"], color="#303946", alpha=0.68, linewidth=0.85, zorder=3.6)
        ax.plot(valid_ne["round"], valid_ne["max_ne_price"], color="#303946", alpha=0.68, linewidth=0.85, zorder=3.6)
        ax.plot(ne["round"], ne["truthful_price"], color="black", linestyle="--", linewidth=1.0, label="Competitive/truthful")
        sub = market[market["treatment"].eq(treatment)]
        for seed, seed_df in sub.groupby("seed"):
            ax.plot(seed_df["round"], seed_df["market_price"], color=color, alpha=0.42, linewidth=0.9, label="Seed paths" if seed == sub["seed"].min() else None, zorder=2.5)
        avg = price_avg[price_avg["treatment"].eq(treatment)]
        ax.plot(avg["round"], avg["market_price"], color=color, alpha=0.82, linewidth=1.55, label="Treatment average", zorder=3.0)
        for day_end in [24, 48, 72, 96, 120, 144]:
            ax.axvline(day_end, color="#dddddd", linewidth=0.8, alpha=0.7, zorder=0)
        ax.set_title(TREATMENT_LABELS[treatment], loc="left")
        ax.set_ylabel("Market price ($/MWh)")
        ax.grid(axis="y", alpha=0.24)
    axes[-1].set_xlabel("Operating hour (round 1-168)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=5, frameon=False)
    fig.suptitle("NAPS TARJ price trajectories: five seeds per treatment", y=0.997, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    fig.savefig(PLOTS_DIR / "sbert_direction_v1_naps_tarj_25_price_trajectories_by_treatment.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    hourly_markup = (
        actions[actions["firm_id"].isin(FIRMS)]
        .groupby(["treatment", "seed", "round"], as_index=False)["markup_factor"]
        .mean()
        .rename(columns={"markup_factor": "mean_markup"})
    )
    markup_avg = hourly_markup.groupby(["treatment", "round"], as_index=False)["mean_markup"].mean()
    valid_markup_ne = ne.dropna(subset=["nash_band_min", "nash_band_max"])
    fig, axes = plt.subplots(len(TREATMENT_ORDER), 1, figsize=(12, 14.0), sharex=True, sharey=True)
    for ax, treatment in zip(axes, TREATMENT_ORDER):
        color = TREATMENT_COLORS[treatment]
        ax.fill_between(valid_markup_ne["round"], valid_markup_ne["nash_band_min"], valid_markup_ne["nash_band_max"], color="#5A6472", alpha=0.25, label="Nash markup band", zorder=1.8)
        ax.plot(valid_markup_ne["round"], valid_markup_ne["nash_band_min"], color="#303946", alpha=0.68, linewidth=0.85, zorder=3.6)
        ax.plot(valid_markup_ne["round"], valid_markup_ne["nash_band_max"], color="#303946", alpha=0.68, linewidth=0.85, zorder=3.6)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Competitive markup")
        sub = hourly_markup[hourly_markup["treatment"].eq(treatment)]
        for seed, seed_df in sub.groupby("seed"):
            ax.plot(seed_df["round"], seed_df["mean_markup"], color=color, alpha=0.42, linewidth=0.9, label="Seed paths" if seed == sub["seed"].min() else None, zorder=2.5)
        avg = markup_avg[markup_avg["treatment"].eq(treatment)]
        ax.plot(avg["round"], avg["mean_markup"], color=color, alpha=0.82, linewidth=1.55, label="Treatment average", zorder=3.0)
        for day_end in [24, 48, 72, 96, 120, 144]:
            ax.axvline(day_end, color="#dddddd", linewidth=0.8, alpha=0.7, zorder=0)
        ax.set_title(TREATMENT_LABELS[treatment], loc="left")
        ax.set_ylabel("Mean markup")
        ax.grid(axis="y", alpha=0.24)
    axes[-1].set_xlabel("Operating hour (round 1-168)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=5, frameon=False)
    fig.suptitle("NAPS TARJ markup trajectories: five seeds per treatment", y=0.997, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    fig.savefig(PLOTS_DIR / "sbert_direction_v1_naps_tarj_25_markup_trajectories_by_treatment.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_matching_naps_plots(by_run: pd.DataFrame, treatment_summary: pd.DataFrame, market: pd.DataFrame, actions: pd.DataFrame, regret: pd.DataFrame) -> None:
    ne = load_ne_table()
    save_naps_summary_table(treatment_summary)
    save_naps_price_increase_by_run(by_run)
    save_naps_consumer_payment(treatment_summary)
    save_naps_daily_price_increase(market)
    save_naps_weekly_price_trajectories(market, ne)
    save_naps_hourly_regret_profit(regret)
    save_naps_markup_trajectories(actions, ne)
    save_naps_market_price_vs_ne(market, ne)
    save_naps_seed_trajectories(market, actions, ne)


def save_plots(treatment_summary: pd.DataFrame, quartile: pd.DataFrame, prompt_scores: pd.DataFrame, reasoning_summary: pd.DataFrame, reasoning_scored: pd.DataFrame) -> None:
    bar_plot(treatment_summary, "avg_price_increase", "Average price increase by treatment", "$/MWh", "sbert_direction_v1_average_price_increase_by_treatment.png")
    bar_plot(treatment_summary, "avg_markup", "Average markup by treatment", "Mean markup", "sbert_direction_v1_average_markup_by_treatment.png")
    bar_plot(treatment_summary, "avg_total_firm_profit", "Average total firm profit by treatment", "$", "sbert_direction_v1_profit_by_treatment.png")
    bar_plot(treatment_summary, "mean_hour_max_regret", "Mean hourly max regret by treatment", "$/hour", "sbert_direction_v1_regret_by_treatment.png")
    grouped_quartile_plot(quartile, "avg_markup", "Load-quartile markup by treatment", "Mean markup", "sbert_direction_v1_load_quartile_markup_by_treatment.png")
    grouped_quartile_plot(quartile, "avg_price_increase", "Load-quartile price impact by treatment", "$/MWh", "sbert_direction_v1_load_quartile_price_impact_by_treatment.png")

    prompt_plot = prompt_scores.rename(columns={"treatment_name": "treatment", "prompt_profit_minus_compliance": "prompt_orientation"})
    bar_plot(prompt_plot, "prompt_orientation", "Prompt SBERT orientation by treatment", "Profit minus compliance", "sbert_direction_v1_prompt_orientation_by_treatment.png")

    reason_plot = reasoning_summary.rename(columns={"mean_reasoning_profit_orientation": "reasoning_orientation"})
    bar_plot(reason_plot, "reasoning_orientation", "TARJ reasoning orientation by treatment", "Profit minus compliance", "sbert_direction_v1_reasoning_orientation_by_treatment.png")

    scatter_records(reasoning_scored, "average_markup", "Daily firm average markup", "sbert_direction_v1_reasoning_orientation_vs_average_markup.png")
    scatter_records(reasoning_scored, "clearing_price", "Daily average clearing price ($/MWh)", "sbert_direction_v1_reasoning_orientation_vs_clearing_price.png")
    scatter_records(reasoning_scored, "profit", "Daily firm profit ($)", "sbert_direction_v1_reasoning_orientation_vs_profit.png")
    binned_scatter_records(reasoning_scored, "average_markup", "Daily firm average markup", "sbert_direction_v1_reasoning_orientation_vs_average_markup_binned.png")
    binned_scatter_records(reasoning_scored, "clearing_price", "Daily average clearing price ($/MWh)", "sbert_direction_v1_reasoning_orientation_vs_clearing_price_binned.png")
    binned_scatter_records(reasoning_scored, "profit", "Daily firm profit ($)", "sbert_direction_v1_reasoning_orientation_vs_profit_binned.png")

    treatment_scatter = (
        treatment_summary.merge(prompt_scores.rename(columns={"treatment_name": "treatment"}), on="treatment", how="left")
        .merge(reasoning_summary, on="treatment", how="left")
    )
    scatter_treatment(treatment_scatter, "avg_price_increase", "Average price increase ($/MWh)", "sbert_direction_v1_prompt_orientation_vs_avg_price_increase.png")
    scatter_treatment(treatment_scatter, "mean_reasoning_profit_orientation", "Mean TARJ reasoning orientation", "sbert_direction_v1_prompt_orientation_vs_reasoning_orientation.png")
    scatter_treatment(treatment_scatter, "avg_markup", "Average markup", "sbert_direction_v1_prompt_orientation_vs_avg_markup.png")


def summarize_failures(actions: pd.DataFrame) -> pd.DataFrame:
    return (
        actions.groupby(["treatment", "seed", "run_id"], as_index=False)
        .agg(
            rows=("valid_response", "size"),
            invalid_response_rows=("valid_response", lambda s: int((~s.astype(bool)).sum())),
            retry_used_rows=("retry_used", "sum"),
            fallback_used_rows=("fallback_used", "sum"),
        )
        .pipe(treatment_sort)
    )


def write_report(treatment_summary: pd.DataFrame, prompt_scores: pd.DataFrame, failure_summary: pd.DataFrame) -> None:
    lines = [
        "# SBERT Direction v1 Five-Treatment Summary",
        "",
        "## Prompt Orientation",
        "",
        prompt_scores.to_markdown(index=False),
        "",
        "## Treatment Outcomes",
        "",
        treatment_summary.to_markdown(index=False),
        "",
        "## Retry/Fallback Summary",
        "",
        failure_summary.to_markdown(index=False),
        "",
    ]
    (REPORTS_DIR / "sbert_direction_v1_five_treatment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--model", default="all-mpnet-base-v2")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Only refresh the treatment summary CSV/report/table PNG from cached metrics.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Only redraw plots from cached five-treatment analysis outputs.",
    )
    parser.add_argument(
        "--recompute-regret",
        action="store_true",
        help="Recompute firm-hour best-response regret instead of using the cached CSV.",
    )
    parser.add_argument(
        "--recompute-sbert",
        action="store_true",
        help="Recompute reasoning SBERT orientation instead of using cached CSVs.",
    )
    parser.add_argument(
        "--recompute-reference",
        action="store_true",
        help="Recompute the perfect-competition reference row instead of using the cached CSV.",
    )
    return parser.parse_args()


def cached_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def load_cached_treatment_summary_no_reference() -> pd.DataFrame:
    no_ref_path = TABLES_DIR / "sbert_direction_v1_treatment_summary_no_reference.csv"
    no_ref = cached_csv(no_ref_path)
    if no_ref is not None:
        return no_ref

    summary_path = TABLES_DIR / "sbert_direction_v1_treatment_summary.csv"
    summary = cached_csv(summary_path)
    if summary is None:
        raise FileNotFoundError(
            "No cached treatment summary found. Run without --table-only once to build caches."
        )
    return summary[summary["treatment"].ne("perfect_competition")].copy()


def refresh_summary_outputs_from_cache(recompute_reference: bool = False) -> pd.DataFrame:
    by_treatment = load_cached_treatment_summary_no_reference()
    by_treatment_with_reference = add_perfect_competition_reference(by_treatment, recompute_reference)
    by_treatment_with_reference.to_csv(TABLES_DIR / "sbert_direction_v1_treatment_summary.csv", index=False)
    by_treatment.to_csv(TABLES_DIR / "sbert_direction_v1_treatment_summary_no_reference.csv", index=False)

    prompt_scores = prompt_orientation()
    failure_summary = cached_csv(TABLES_DIR / "sbert_direction_v1_retry_fallback_summary.csv")
    if failure_summary is None:
        failure_summary = pd.DataFrame()
    reasoning_summary = cached_csv(TABLES_DIR / "sbert_direction_v1_reasoning_orientation_by_treatment.csv")
    if reasoning_summary is None:
        reasoning_summary = pd.DataFrame()

    save_naps_summary_table(by_treatment_with_reference)
    save_paper_main_results_table(by_treatment_with_reference, prompt_scores, reasoning_summary)
    save_naps_consumer_payment(by_treatment_with_reference)
    write_report(by_treatment_with_reference, prompt_scores, failure_summary)
    return by_treatment_with_reference


def refresh_plots_from_cache(recompute_reference: bool = False) -> None:
    by_treatment = load_cached_treatment_summary_no_reference()
    by_treatment_with_reference = add_perfect_competition_reference(by_treatment, recompute_reference)

    cached_inputs = {
        "market": TABLES_DIR / "sbert_direction_v1_market_records.csv",
        "actions": TABLES_DIR / "sbert_direction_v1_firm_action_records.csv",
        "regret": TABLES_DIR / "sbert_direction_v1_firm_hour_regret.csv",
        "by_run": TABLES_DIR / "sbert_direction_v1_metrics_by_run.csv",
        "quartile": TABLES_DIR / "sbert_direction_v1_load_quartile_metrics.csv",
        "prompt_scores": TABLES_DIR / "sbert_direction_v1_prompt_orientation_by_treatment.csv",
        "reasoning_summary": TABLES_DIR / "sbert_direction_v1_reasoning_orientation_by_treatment.csv",
        "reasoning_scored": TABLES_DIR / "sbert_direction_v1_reasoning_orientation_records.csv",
    }
    missing = [str(path) for path in cached_inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing cached analysis outputs needed for --plots-only:\n" + "\n".join(missing))

    market = pd.read_csv(cached_inputs["market"])
    actions = pd.read_csv(cached_inputs["actions"])
    regret = pd.read_csv(cached_inputs["regret"])
    by_run = pd.read_csv(cached_inputs["by_run"])
    quartile = pd.read_csv(cached_inputs["quartile"])
    prompt_scores = pd.read_csv(cached_inputs["prompt_scores"])
    reasoning_summary = pd.read_csv(cached_inputs["reasoning_summary"])
    reasoning_scored = pd.read_csv(cached_inputs["reasoning_scored"])

    save_paper_main_results_table(by_treatment_with_reference, prompt_scores, reasoning_summary)
    save_plots(by_treatment, quartile, prompt_scores, reasoning_summary, reasoning_scored)
    save_matching_naps_plots(by_run, by_treatment_with_reference, market, actions, regret)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.table_only:
        by_treatment_with_reference = refresh_summary_outputs_from_cache(args.recompute_reference)
        print("\nTreatment summary refreshed from cache")
        print("=" * 36)
        print(
            by_treatment_with_reference[
                [
                    "treatment",
                    "n_runs",
                    "avg_price_increase",
                    "avg_markup",
                    "avg_total_firm_profit_gain",
                    "mean_relative_regret_br",
                    "mean_hour_max_regret",
                ]
            ].to_string(index=False)
        )
        print(f"\n[write] {TABLES_DIR / 'sbert_direction_v1_treatment_summary.csv'}")
        print(f"[write] {PLOTS_DIR / 'sbert_direction_v1_naps_tarj_metrics_summary_table.png'}")
        print(f"[write] {REPORTS_DIR / 'sbert_direction_v1_five_treatment_summary.md'}")
        return
    if args.plots_only:
        refresh_plots_from_cache(args.recompute_reference)
        print("\nFive-treatment plots refreshed from cache")
        print("=" * 40)
        print(f"[write] {PLOTS_DIR}")
        return

    market, actions, firm, responses = load_run_frames(args.seeds)
    failure_summary = summarize_failures(actions)
    regret_path = TABLES_DIR / "sbert_direction_v1_firm_hour_regret.csv"
    if regret_path.exists() and not args.recompute_regret:
        regret = pd.read_csv(regret_path)
    else:
        regret = compute_regret(market, actions)
        regret.to_csv(regret_path, index=False)

    by_run, by_treatment = compute_metrics(market, actions, firm, regret)
    by_treatment_with_reference = add_perfect_competition_reference(by_treatment, args.recompute_reference)
    quartile = load_quartile_metrics(market, actions)
    prompt_scores = prompt_orientation()
    reasoning_scored_path = TABLES_DIR / "sbert_direction_v1_reasoning_orientation_records.csv"
    reasoning_summary_path = TABLES_DIR / "sbert_direction_v1_reasoning_orientation_by_treatment.csv"
    if (
        reasoning_scored_path.exists()
        and reasoning_summary_path.exists()
        and not args.recompute_sbert
    ):
        reasoning_scored = pd.read_csv(reasoning_scored_path)
        reasoning_summary = pd.read_csv(reasoning_summary_path)
    else:
        reasoning = build_reasoning_records(responses, firm, market, regret)
        reasoning_scored, reasoning_summary = add_reasoning_orientation(reasoning, args.model, args.local_files_only)
        reasoning_scored.to_csv(reasoning_scored_path, index=False)
        reasoning_summary.to_csv(reasoning_summary_path, index=False)

    market.to_csv(TABLES_DIR / "sbert_direction_v1_market_records.csv", index=False)
    actions.to_csv(TABLES_DIR / "sbert_direction_v1_firm_action_records.csv", index=False)
    by_run.to_csv(TABLES_DIR / "sbert_direction_v1_metrics_by_run.csv", index=False)
    by_treatment.to_csv(TABLES_DIR / "sbert_direction_v1_treatment_summary_no_reference.csv", index=False)
    by_treatment_with_reference.to_csv(TABLES_DIR / "sbert_direction_v1_treatment_summary.csv", index=False)
    quartile.to_csv(TABLES_DIR / "sbert_direction_v1_load_quartile_metrics.csv", index=False)
    prompt_scores.to_csv(TABLES_DIR / "sbert_direction_v1_prompt_orientation_by_treatment.csv", index=False)
    failure_summary.to_csv(TABLES_DIR / "sbert_direction_v1_retry_fallback_summary.csv", index=False)

    save_paper_main_results_table(by_treatment_with_reference, prompt_scores, reasoning_summary)
    save_plots(by_treatment, quartile, prompt_scores, reasoning_summary, reasoning_scored)
    save_matching_naps_plots(by_run, by_treatment_with_reference, market, actions, regret)
    write_report(by_treatment_with_reference, prompt_scores, failure_summary)

    print("\nFive-treatment analysis complete")
    print("=" * 32)
    print(by_treatment_with_reference[["treatment", "n_runs", "avg_price_increase", "avg_markup", "avg_total_firm_profit_gain", "mean_hour_max_regret"]].to_string(index=False))
    print("\nRetry/fallback totals")
    print(failure_summary[["treatment", "seed", "retry_used_rows", "fallback_used_rows", "invalid_response_rows"]].to_string(index=False))
    print(f"\n[write] {TABLES_DIR}")
    print(f"[write] {PLOTS_DIR}")
    print(f"[write] {REPORTS_DIR / 'sbert_direction_v1_five_treatment_summary.md'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
