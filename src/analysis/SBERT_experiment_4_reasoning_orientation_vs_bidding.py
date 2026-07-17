#!/usr/bin/env python3
"""SBERT Experiment 4: reasoning orientation versus bidding aggressiveness.

This script applies the Experiment 2 profit-vs-compliance contrast to each
generated TARJ reasoning record from Experiment 3, then tests whether the
resulting reasoning orientation relates to bidding and outcome variables.

It embeds only generated TARJ reasoning text:
    Thought + Action summary + Reflection + Journal update

It does not embed full prompts or objective text for the reasoning scores.

Run from repository root:
    python3 NAPS_paper_experiments/SBERT/SBERT_experiment_4_reasoning_orientation_vs_bidding.py --local-files-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


SBERT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = SBERT_DIR / "outputs"
EXPERIMENT_TAG = "SBERT_experiment_4"
OUT_DIR = OUT_ROOT / EXPERIMENT_TAG
EXP3_DIR = OUT_ROOT / "SBERT_experiment_3"

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

OUTCOME_COLUMNS = [
    "average_markup",
    "selected_markup",
    "profit",
    "regret",
    "clearing_price",
    "load",
]


def out_path(stem: str, suffix: str) -> Path:
    return OUT_DIR / f"{EXPERIMENT_TAG}_{stem}{suffix}"


def require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers.\n"
            "Install it, then rerun:\n"
            "  python3 -m pip install sentence-transformers\n"
        ) from exc
    return SentenceTransformer


def load_sbert_model(model_name: str, local_files_only: bool = False):
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
    return np.asarray(
        model.encode(texts, convert_to_numpy=True, normalize_embeddings=True),
        dtype=float,
    )


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def load_experiment_3_inputs() -> tuple[pd.DataFrame, np.ndarray]:
    metadata_path = EXP3_DIR / "SBERT_experiment_3_tarj_reasoning_embedding_metadata.csv"
    embeddings_path = EXP3_DIR / "SBERT_experiment_3_tarj_reasoning_embeddings.npy"
    if not metadata_path.exists() or not embeddings_path.exists():
        raise SystemExit(
            "Experiment 3 metadata/embeddings not found. Run SBERT_experiment_3 first."
        )
    metadata = pd.read_csv(metadata_path)
    embeddings = np.load(embeddings_path)
    if len(metadata) != embeddings.shape[0]:
        raise ValueError("Experiment 3 metadata row count does not match embedding rows.")
    return metadata, embeddings


def compute_anchor_centroids(model_name: str, local_files_only: bool) -> tuple[list[str], np.ndarray]:
    model = load_sbert_model(model_name, local_files_only=local_files_only)
    labels = list(CONTRAST_ANCHORS)
    centroids = []
    for label in labels:
        embeddings = encode_texts(CONTRAST_ANCHORS[label], model)
        centroids.append(normalize_rows(embeddings.mean(axis=0, keepdims=True))[0])
    anchors = pd.DataFrame(
        [
            {"anchor_category": category, "anchor_sentence": sentence}
            for category, sentences in CONTRAST_ANCHORS.items()
            for sentence in sentences
        ]
    )
    anchors.to_csv(out_path("profit_vs_compliance_anchor_sentences", ".csv"), index=False)
    return labels, np.vstack(centroids)


def add_reasoning_orientation(metadata: pd.DataFrame, embeddings: np.ndarray, anchor_labels: list[str], anchor_embeddings: np.ndarray) -> pd.DataFrame:
    sims = embeddings @ anchor_embeddings.T
    sim_df = pd.DataFrame(sims, columns=anchor_labels)
    out = pd.concat([metadata.reset_index(drop=True), sim_df], axis=1)
    out["reasoning_profit_similarity"] = out["pure_profit_maximization"]
    out["reasoning_compliance_similarity"] = out["pure_compliance_constraint"]
    out["reasoning_profit_minus_compliance"] = (
        out["reasoning_profit_similarity"] - out["reasoning_compliance_similarity"]
    )
    out["reasoning_orientation"] = np.where(
        out["reasoning_profit_minus_compliance"] >= 0,
        "profit_oriented",
        "compliance_oriented",
    )
    return out


def pearson_spearman(x: pd.Series, y: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return {"n": len(frame), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    pr = stats.pearsonr(frame["x"], frame["y"])
    sr = stats.spearmanr(frame["x"], frame["y"])
    return {
        "n": len(frame),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def compute_correlations(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOME_COLUMNS:
        if outcome not in scored:
            continue
        vals = pearson_spearman(scored["reasoning_profit_minus_compliance"], scored[outcome])
        rows.append({"outcome": outcome, **vals})
    corr = pd.DataFrame(rows)
    corr.to_csv(out_path("reasoning_orientation_outcome_correlations", ".csv"), index=False)
    return corr


def treatment_adjusted_regressions(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOME_COLUMNS:
        if outcome not in scored:
            continue
        frame = scored[["reasoning_profit_minus_compliance", "treatment_name", outcome]].copy()
        frame[outcome] = pd.to_numeric(frame[outcome], errors="coerce")
        frame = frame.dropna()
        if len(frame) < 10 or frame[outcome].nunique() < 2:
            continue
        treatment_dummies = pd.get_dummies(frame["treatment_name"], drop_first=True, dtype=float)
        x_full = pd.concat([frame[["reasoning_profit_minus_compliance"]].astype(float), treatment_dummies], axis=1)
        y = frame[outcome].astype(float)
        full = LinearRegression().fit(x_full, y)
        pred = full.predict(x_full)
        coef_orientation = float(full.coef_[0])

        x_treatment = treatment_dummies
        if x_treatment.shape[1] == 0:
            r2_treatment = np.nan
        else:
            treatment_model = LinearRegression().fit(x_treatment, y)
            r2_treatment = float(r2_score(y, treatment_model.predict(x_treatment)))

        x_orientation = frame[["reasoning_profit_minus_compliance"]].astype(float)
        orientation_model = LinearRegression().fit(x_orientation, y)
        r2_orientation = float(r2_score(y, orientation_model.predict(x_orientation)))

        rows.append(
            {
                "outcome": outcome,
                "n": len(frame),
                "coef_reasoning_orientation_with_treatment_controls": coef_orientation,
                "r2_orientation_only": r2_orientation,
                "r2_treatment_only": r2_treatment,
                "r2_orientation_plus_treatment": float(r2_score(y, pred)),
            }
        )
    regs = pd.DataFrame(rows)
    regs.to_csv(out_path("reasoning_orientation_treatment_adjusted_regressions", ".csv"), index=False)
    return regs


def summarize_by_treatment(scored: pd.DataFrame) -> pd.DataFrame:
    agg = (
        scored.groupby("treatment_name", as_index=False)
        .agg(
            n=("reasoning_profit_minus_compliance", "size"),
            mean_reasoning_profit_orientation=("reasoning_profit_minus_compliance", "mean"),
            sd_reasoning_profit_orientation=("reasoning_profit_minus_compliance", "std"),
            mean_average_markup=("average_markup", "mean"),
            mean_selected_markup=("selected_markup", "mean"),
            mean_profit=("profit", "mean"),
            mean_regret=("regret", "mean"),
            mean_clearing_price=("clearing_price", "mean"),
            mean_load=("load", "mean"),
        )
        .sort_values("mean_reasoning_profit_orientation", ascending=False)
    )
    agg.to_csv(out_path("reasoning_orientation_by_treatment", ".csv"), index=False)
    return agg


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / "_matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_scatter_plots(scored: pd.DataFrame) -> None:
    plt = setup_matplotlib()
    for outcome in ["average_markup", "selected_markup", "profit", "regret", "clearing_price"]:
        if outcome not in scored:
            continue
        frame = scored[["reasoning_profit_minus_compliance", outcome, "treatment_name"]].copy()
        frame[outcome] = pd.to_numeric(frame[outcome], errors="coerce")
        frame = frame.dropna()
        if frame.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        for treatment, sub in frame.groupby("treatment_name"):
            ax.scatter(
                sub["reasoning_profit_minus_compliance"],
                sub[outcome],
                s=18,
                alpha=0.55,
                label=treatment,
                edgecolors="none",
            )
        ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Reasoning profit orientation")
        ax.set_ylabel(outcome)
        ax.set_title(f"SBERT Experiment 4: Reasoning Orientation vs {outcome}")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_path(f"reasoning_orientation_vs_{outcome}", ".png"), dpi=220, bbox_inches="tight")
        plt.close(fig)


def save_treatment_barplot(summary: pd.DataFrame) -> None:
    plt = setup_matplotlib()
    plot_df = summary.sort_values("mean_reasoning_profit_orientation")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.barh(plot_df["treatment_name"], plot_df["mean_reasoning_profit_orientation"], color="#0B3D91", alpha=0.85)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Mean reasoning profit orientation")
    ax.set_title("SBERT Experiment 4: Reasoning Orientation by Treatment")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path("reasoning_orientation_by_treatment_barplot", ".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary: pd.DataFrame, corr: pd.DataFrame, regs: pd.DataFrame) -> None:
    print("\nSBERT Experiment 4 summary")
    print("=" * 27)
    print("\nReasoning orientation by treatment")
    print(summary.to_string(index=False))

    print("\nCorrelations with reasoning profit orientation")
    cols = ["outcome", "n", "pearson_r", "pearson_p", "spearman_r", "spearman_p"]
    print(corr[cols].to_string(index=False))

    print("\nTreatment-adjusted regressions")
    if regs.empty:
        print("No regressions estimated.")
    else:
        print(regs.to_string(index=False))

    if not corr.empty:
        strongest = corr.assign(abs_spearman=corr["spearman_r"].abs()).sort_values("abs_spearman", ascending=False).iloc[0]
        print(
            f"\nStrongest monotonic association: {strongest['outcome']} "
            f"(Spearman r={strongest['spearman_r']:.3f}, p={strongest['spearman_p']:.3g})."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all-mpnet-base-v2", help="Sentence-transformers model name.")
    parser.add_argument("--local-files-only", action="store_true", help="Load SBERT model only from local cache.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata, embeddings = load_experiment_3_inputs()
    anchor_labels, anchor_embeddings = compute_anchor_centroids(args.model, args.local_files_only)
    scored = add_reasoning_orientation(metadata, embeddings, anchor_labels, anchor_embeddings)
    scored.to_csv(out_path("reasoning_orientation_scored_records", ".csv"), index=False)
    print(f"[write] {out_path('reasoning_orientation_scored_records', '.csv')}")

    summary = summarize_by_treatment(scored)
    corr = compute_correlations(scored)
    regs = treatment_adjusted_regressions(scored)
    save_scatter_plots(scored)
    save_treatment_barplot(summary)

    for path in sorted(OUT_DIR.glob(f"{EXPERIMENT_TAG}_*")):
        print(f"[write] {path}")
    print_summary(summary, corr, regs)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
