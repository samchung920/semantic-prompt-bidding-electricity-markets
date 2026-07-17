#!/usr/bin/env python3
"""
run_llm_bidding_simulation.py
==============================
LLM-agent bidding simulation for the thermal-only 5-firm electricity market.

Supports two agent types:
  mock   — rule-based agents (truthful / random / heuristic). No API calls.
  openai — real GPT agents via the OpenAI Python SDK.

Market:  single-node merit-order ED with perfect load realization.
Agents:  5 strategic firms + 1 non-strategic scarcity backstop.
Pricing: uniform clearing price = offer price of marginal dispatched unit.
Cost:    always based on true_mc_per_mwh, not offer price.

Usage examples
--------------
# Original test case — mock truthful baseline (backward-compatible)
python run_llm_bidding_simulation.py --agent-type mock --mock-agent-mode truthful --seed 1

# Summer hourly case — mock truthful baseline
python run_llm_bidding_simulation.py \\
    --case-dir single_node_summer_hourly_case \\
    --stress-case high --agent-type mock --mock-agent-mode truthful \\
    --prompt-mode myopic_profit --history-window 24 --n-rounds 1488 --seed 1

# Summer hourly case — real GPT run (requires OPENAI_API_KEY)
python run_llm_bidding_simulation.py \\
    --case-dir single_node_summer_hourly_case \\
    --stress-case high --agent-type openai --model gpt-4.1 \\
    --temperature 0.2 --prompt-mode myopic_profit --history-window 24 --n-rounds 1488 --seed 1
"""

# ============================================================
# CONFIGURATION — defaults (all overridable by argparse)
# ============================================================
PROMPT_MODE          = "myopic_profit"    # legacy prompt-mode support
PROMPT_FILE_OVERRIDE = None
PROMPT_ID            = None
RUN_TAG              = None
SCHEMA_PROFILE       = "decision_rationale"  # decision_rationale | summary | flexible
TREATMENT_NAME       = None
MOCK_AGENT_MODE      = "heuristic"        # "truthful" | "random" | "heuristic"
AGENT_TYPE           = "mock"             # "mock" | "openai"
DECISION_FREQUENCY   = "hourly"           # "hourly" | "daily_schedule"
MODEL                = "gpt-4.1"          # OpenAI model name
TEMPERATURE          = 0.2
MAX_TOKENS           = 2000
API_TIMEOUT          = 60
SLEEP_SECONDS        = 0.2
SEED                 = 1
HISTORY_WINDOW       = 5
N_ROUNDS             = 52
PRICE_CAP            = 1000.0
ALLOWED_MARKUP_FACTORS = [1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0]
_DEFAULT_MARKUP_GRID = list(ALLOWED_MARKUP_FACTORS)   # snapshot; used to detect CLI overrides
STRESS_CASE          = "high"
USE_RESPONSES_API    = False   # True → use Responses API instead of Chat Completions
REASONING_EFFORT     = "medium"  # low | medium | high (Responses API only)

# ============================================================
# IMPORTS AND CONSTANTS
# ============================================================
import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import openai as _openai_lib
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT / "NAPS_paper_experiments"
WORKSPACE_PROMPTS_DIR = WORKSPACE_ROOT / "prompts"
LEGACY_PROMPTS_DIR = PROJECT_ROOT / "prompts"
RUNS_ROOT = WORKSPACE_ROOT / "runs" / "raw_outputs"

# Maps case-directory name → short tag used in output file names
_CASE_TAG_MAP = {
    "single_node_test_case"          : "thermal",
    "single_node_summer_hourly_case" : "hourly_summer",
    "single_node_chron_4mo_case"     : "chron_4mo",
}

# Optional datetime fields added by the chronological hourly dataset
_DATETIME_COLS = ["datetime", "year", "month", "day", "day_of_week", "season"]

_MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

BACKSTOP_GEN_ID    = "SCARCITY_BACKSTOP"
BACKSTOP_FIRM_ID   = "Firm_Backstop"
DISPATCH_TOLERANCE = 1e-4
STRATEGIC_FIRMS    = ["Firm_Nuclear", "Firm_Coal", "Firm_CC", "Firm_CT", "Firm_Oil"]

FIRM_COLORS = {
    "Firm_Nuclear":  "#7b68ee",
    "Firm_Coal":     "#555555",
    "Firm_CC":       "#e07b39",
    "Firm_CT":       "#f4a460",
    "Firm_Oil":      "#c0392b",
    "Firm_Backstop": "#cccccc",
}

OPENAI_SYSTEM_MSG = (
    "You are a careful electricity-market bidding agent. "
    "Return only valid JSON matching the requested schema."
)

# ── Responses API structured-output schemas (daily-schedule mode) ────────────
# FISH schema: PLANS / INSIGHTS memory + hourly schedule
_DAILY_SCHEDULE_FISH_RESPONSES = {
    "type": "object",
    "required": [
        "firm_id",
        "date",
        "prompt_architecture",
        "objective_treatment",
        "observations_and_thoughts",
        "new_plans",
        "new_insights",
        "hourly_markup_schedule",
        "confidence",
    ],
    "additionalProperties": False,
    "properties": {
        "firm_id": {"type": "string"},
        "date": {"type": "string"},
        "prompt_architecture": {"type": "string", "enum": ["fish"]},
        "objective_treatment": {"type": "string"},
        "observations_and_thoughts": {"type": "string"},
        "new_plans": {"type": "string"},
        "new_insights": {"type": "string"},
        "hourly_markup_schedule": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["hour", "markup_factor", "rationale"],
                "additionalProperties": False,
                "properties": {
                    "hour": {"type": "integer"},
                    "markup_factor": {"type": "number"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
}

# TARJ schema: structured reasoning + journal + hourly schedule
_DAILY_SCHEDULE_TARJ_RESPONSES = {
    "type": "object",
    "required": [
        "firm_id",
        "date",
        "prompt_architecture",
        "objective_treatment",
        "thought",
        "action_summary",
        "reflection",
        "journal_update",
        "hourly_markup_schedule",
        "confidence",
    ],
    "additionalProperties": False,
    "properties": {
        "firm_id": {"type": "string"},
        "date": {"type": "string"},
        "prompt_architecture": {"type": "string", "enum": ["tarj"]},
        "objective_treatment": {"type": "string"},
        "thought": {"type": "string"},
        "action_summary": {"type": "string"},
        "reflection": {"type": "string"},
        "journal_update": {"type": "string"},
        "hourly_markup_schedule": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["hour", "markup_factor", "rationale"],
                "additionalProperties": False,
                "properties": {
                    "hour": {"type": "integer"},
                    "markup_factor": {"type": "number"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
}

# SUMMARY schema: compact strategy summary + hourly schedule
_DAILY_SCHEDULE_SUMMARY_RESPONSES = {
    "type": "object",
    "required": ["firm_id", "date", "strategy_summary", "hourly_markup_schedule", "confidence"],
    "additionalProperties": False,
    "properties": {
        "firm_id":          {"type": "string"},
        "date":             {"type": "string"},
        "strategy_summary": {"type": "string"},
        "hourly_markup_schedule": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["hour", "markup_factor", "rationale"],
                "additionalProperties": False,
                "properties": {
                    "hour":          {"type": "integer"},
                    "markup_factor": {"type": "number"},
                    "rationale":     {"type": "string"},
                },
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
}

# FLEXIBLE schema: base fields only, extra prompt-specific fields allowed
_DAILY_SCHEDULE_FLEXIBLE_RESPONSES = {
    "type": "object",
    "required": ["firm_id", "date", "hourly_markup_schedule", "confidence"],
    "additionalProperties": True,
    "properties": {
        "firm_id": {"type": "string"},
        "date":    {"type": "string"},
        "hourly_markup_schedule": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["hour", "markup_factor"],
                "additionalProperties": True,
                "properties": {
                    "hour":          {"type": "integer"},
                    "markup_factor": {"type": "number"},
                    "rationale":     {"type": "string"},
                },
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
}

# OLD_SCHEMA mode (decision_rationale): myopic_profit
_DAILY_SCHEDULE_OLD_SCHEMA_RESPONSES = {
    "type": "object",
    "required": ["firm_id", "date", "decision_rationale", "hourly_markup_schedule", "confidence"],
    "additionalProperties": False,
    "properties": {
        "firm_id": {"type": "string"},
        "date":    {"type": "string"},
        "decision_rationale": {
            "type": "object",
            "required": ["market_assessment", "profit_tradeoff",
                         "risk_assessment", "final_justification"],
            "additionalProperties": False,
            "properties": {
                "market_assessment":   {"type": "string"},
                "profit_tradeoff":     {"type": "string"},
                "risk_assessment":     {"type": "string"},
                "final_justification": {"type": "string"},
            },
        },
        "hourly_markup_schedule": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["hour", "markup_factor", "rationale"],
                "additionalProperties": False,
                "properties": {
                    "hour":          {"type": "integer"},
                    "markup_factor": {"type": "number"},
                    "rationale":     {"type": "string"},
                },
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
}

PROMPT_FILES = {
    "myopic_profit":                                 "firm_agent_myopic_v1.txt",
    "cumulative_profit":                             "firm_agent_cumulative_v1.txt",
    "hourly_myopic_profit":                          "firm_agent_hourly_myopic_v1.txt",
    "hourly_cumulative_profit":                      "firm_agent_hourly_cumulative_v1.txt",
    "daily_schedule_myopic_profit":                  "firm_agent_daily_schedule_myopic_v1.txt",
    "daily_schedule_cumulative_profit":              "firm_agent_daily_schedule_cumulative_v1.txt",
    "daily_schedule_myopic_diagnostics":             "firm_agent_daily_schedule_myopic_diagnostics_v1.txt",
    "daily_schedule_profit_max_no_caution":          "firm_agent_daily_schedule_profit_max_no_caution_v1.txt",
    "daily_schedule_br_reasoning_scaffold":          "firm_agent_daily_schedule_br_reasoning_scaffold_v1.txt",
    "daily_schedule_clean_market_power_diagnostics": "firm_agent_daily_schedule_clean_market_power_diagnostics_v1.txt",
    "daily_schedule_competitive_compliance":         "firm_agent_daily_schedule_competitive_compliance_v1.txt",
}

_CUMULATIVE_MODES = frozenset([
    "cumulative_profit",
    "hourly_cumulative_profit",
    "daily_schedule_cumulative_profit",
])

_DAILY_SCHEDULE_MODES = frozenset([
    "daily_schedule_myopic_profit",
    "daily_schedule_cumulative_profit",
    "daily_schedule_myopic_diagnostics",
    "daily_schedule_profit_max_no_caution",
    "daily_schedule_br_reasoning_scaffold",
    "daily_schedule_clean_market_power_diagnostics",
    "daily_schedule_competitive_compliance",
])

_DIAGNOSTICS_MODES = frozenset([
    "daily_schedule_myopic_diagnostics",
    "daily_schedule_clean_market_power_diagnostics",
])

# Modes that use strategy_summary + per-hour rationale schema (not decision_rationale dict)
_NEW_SCHEMA_MODES = frozenset([
    "daily_schedule_profit_max_no_caution",
    "daily_schedule_br_reasoning_scaffold",
    "daily_schedule_clean_market_power_diagnostics",
    "daily_schedule_competitive_compliance",
])

# Maps prompt_mode -> information_treatment label for metadata
_INFORMATION_TREATMENTS = {
    "daily_schedule_myopic_diagnostics":             "market_position_diagnostics",
    "daily_schedule_profit_max_no_caution":          "profit_max_framing",
    "daily_schedule_br_reasoning_scaffold":          "br_reasoning_scaffold",
    "daily_schedule_clean_market_power_diagnostics": "clean_market_power_diagnostics",
    "daily_schedule_competitive_compliance":         "competitive_compliance_framing",
}


# ============================================================
# FILENAME HELPERS
# ============================================================

def sanitize_for_filename(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).replace(".", "p")


# ============================================================
# AGENT CLASSES
# ============================================================

class BaseFirmAgent:
    def __init__(self, firm_id, portfolio_df, allowed_markup_factors, seed=None):
        self.firm_id                = firm_id
        self.portfolio              = portfolio_df.copy()
        self.allowed_markup_factors = list(allowed_markup_factors)
        self.seed                   = seed

    def decide(self, round_info, recent_public_history, own_recent_performance,
               rendered_prompt=None, memory_summary=None):
        raise NotImplementedError

    def decide_daily(self, day_info, daily_load_forecast, recent_public_history,
                     own_recent_performance, rendered_prompt=None, memory_summary=None):
        raise NotImplementedError

    def _make_response(self, round_info, markup_factor, rationale, confidence="medium"):
        return {
            "firm_id":            self.firm_id,
            "round":              round_info["round"],
            "decision_rationale": rationale,
            "markup_factor":      markup_factor,
            "confidence":         confidence,
        }


class MockFirmAgent(BaseFirmAgent):
    def __init__(self, firm_id, portfolio_df, allowed_markup_factors,
                 mode="heuristic", load_percentiles=None, seed=None):
        super().__init__(firm_id, portfolio_df, allowed_markup_factors, seed)
        self.mode             = mode
        self.load_percentiles = load_percentiles or {"p75": float("inf"), "p90": float("inf")}
        self.rng              = np.random.default_rng(seed)

    def decide(self, round_info, recent_public_history, own_recent_performance,
               rendered_prompt=None, memory_summary=None):
        load_mw = round_info["load_mw"]

        if self.mode == "truthful":
            markup     = 1.0
            rationale  = self._rationale_truthful(round_info)
            confidence = "high"
        elif self.mode == "random":
            markup     = float(self.rng.choice(self.allowed_markup_factors))
            rationale  = self._rationale_random(round_info, markup)
            confidence = "low"
        elif self.mode == "heuristic":
            markup     = self._heuristic_markup(load_mw)
            rationale  = self._rationale_heuristic(round_info, markup)
            confidence = "medium" if markup > 1.0 else "high"
        else:
            raise ValueError(f"Unknown mock agent mode: {self.mode}")

        if memory_summary and "learning_from_history" not in rationale:
            n_prev = len(own_recent_performance)
            rationale["learning_from_history"] = (
                f"Drawing on {n_prev} prior rounds; "
                f"adjusting markup based on own profit and dispatch history."
            )

        response = self._make_response(round_info, markup, rationale, confidence)
        tracking = {
            "agent_type":              "mock",
            "model":                   "",
            "temperature":             "",
            "max_tokens":              "",
            "raw_agent_response_text": json.dumps(response),
            "parse_error":             "",
            "retry_used":              False,
            "fallback_used":           False,
        }
        return markup, response, tracking

    def _heuristic_markup(self, load_mw):
        p75          = self.load_percentiles["p75"]
        p90          = self.load_percentiles["p90"]
        is_high      = load_mw > p75
        is_very_high = load_mw > p90

        if self.firm_id == "Firm_Nuclear":
            return 1.0
        elif self.firm_id == "Firm_Coal":
            return float(self.rng.choice([1.25, 1.5])) if is_high else float(self.rng.choice([1.0, 1.1]))
        elif self.firm_id == "Firm_CC":
            return float(self.rng.choice([2.0, 3.0])) if is_high else float(self.rng.choice([1.25, 1.5]))
        elif self.firm_id == "Firm_CT":
            return float(self.rng.choice([1.25, 1.5])) if is_very_high else 1.0
        elif self.firm_id == "Firm_Oil":
            return float(self.rng.choice([2.0, 3.0])) if is_very_high else 1.0
        else:
            return 1.0

    def _rationale_truthful(self, round_info):
        return {
            "market_assessment":   f"Load is {round_info['load_mw']:,.0f} MW; bidding truthfully.",
            "profit_tradeoff":     "Markup 1.0 means revenue equals marginal cost; no markup profit.",
            "risk_assessment":     "Truthful bidding guarantees dispatch with zero markup risk.",
            "final_justification": "Markup factor 1.0 selected per truthful-bidding mode.",
        }

    def _rationale_random(self, round_info, markup):
        return {
            "market_assessment":   f"Load is {round_info['load_mw']:,.0f} MW.",
            "profit_tradeoff":     "Markup chosen randomly; no expected-profit calculation performed.",
            "risk_assessment":     "Random selection may result in lost dispatch or suboptimal profit.",
            "final_justification": f"Markup factor {markup} selected at random.",
        }

    def _rationale_heuristic(self, round_info, markup):
        load = round_info["load_mw"]
        p75  = self.load_percentiles["p75"]
        p90  = self.load_percentiles["p90"]
        if load > p90:
            load_desc = "very high (above 90th percentile)"
        elif load > p75:
            load_desc = "high (above 75th percentile)"
        else:
            load_desc = "moderate (below 75th percentile)"
        return {
            "market_assessment": f"Load is {load:,.0f} MW — {load_desc}.",
            "profit_tradeoff": (
                f"Markup {markup} captures infra-marginal rent while risking volume loss."
                if markup > 1.0 else
                "No markup applied; prioritising reliable dispatch at true cost."
            ),
            "risk_assessment": (
                "Higher markup increases profit if firm remains in merit order."
                if markup > 1.0 else
                "Truthful bid eliminates dispatch risk for this firm."
            ),
            "final_justification": (
                f"Heuristic rule for {self.firm_id} selects markup {markup} given {load_desc} demand."
            ),
        }

    # ------------------------------------------------------------------
    # Daily-schedule decision (one call per day → 24-hour markup vector)
    # ------------------------------------------------------------------

    def decide_daily(self, day_info, daily_load_forecast, recent_public_history,
                     own_recent_performance, rendered_prompt=None, memory_summary=None):
        date = str(day_info["date"])
        schedule = []

        if self.mode == "truthful":
            for h, _ in daily_load_forecast:
                schedule.append({"hour": h, "markup_factor": 1.0})
            avg_load = float(np.mean([lm for _, lm in daily_load_forecast]))
            rationale = {
                "market_assessment":   f"Submitting truthful 24-hour schedule; avg load {avg_load:,.0f} MW.",
                "profit_tradeoff":     "All markups = 1.0; revenue equals marginal cost.",
                "risk_assessment":     "Truthful bidding guarantees dispatch with zero markup risk.",
                "final_justification": "All 24 markup factors = 1.0 per truthful-bidding mode.",
            }
            confidence = "high"

        elif self.mode == "random":
            for h, _ in daily_load_forecast:
                mu = float(self.rng.choice(self.allowed_markup_factors))
                schedule.append({"hour": h, "markup_factor": mu})
            rationale = {
                "market_assessment":   f"Submitting random 24-hour schedule for {date}.",
                "profit_tradeoff":     "Markups chosen randomly; no expected-profit calculation.",
                "risk_assessment":     "Random selection may result in lost dispatch or suboptimal profit.",
                "final_justification": "24 markup factors chosen uniformly at random.",
            }
            confidence = "low"

        elif self.mode == "heuristic":
            for h, load_mw in daily_load_forecast:
                mu = self._heuristic_markup(load_mw)
                schedule.append({"hour": h, "markup_factor": mu})
            avg_mu = float(np.mean([e["markup_factor"] for e in schedule]))
            peak_hours = sum(1 for _, lm in daily_load_forecast
                             if lm > self.load_percentiles["p75"])
            rationale = {
                "market_assessment":   (
                    f"Today has {peak_hours}/24 high-load hours (above p75); "
                    f"avg forecast load {float(np.mean([lm for _, lm in daily_load_forecast])):,.0f} MW."
                ),
                "profit_tradeoff":     (
                    f"Avg markup {avg_mu:.2f} across 24 hours; higher markups in peak hours."
                ),
                "risk_assessment":     (
                    "Heuristic raises markup in peak hours where firm is likely inframarginal."
                    if avg_mu > 1.0 else
                    "Firm bids truthfully due to low-load conditions."
                ),
                "final_justification": (
                    f"Heuristic rule for {self.firm_id} applied per-hour based on load percentile."
                ),
            }
            confidence = "medium" if avg_mu > 1.0 else "high"
        else:
            raise ValueError(f"Unknown mock agent mode: {self.mode}")

        if memory_summary and "learning_from_history" not in rationale:
            n_prev = len(own_recent_performance)
            rationale["learning_from_history"] = (
                f"Drawing on {n_prev} prior hours; adjusting daily markup schedule "
                f"based on own profit and dispatch history."
            )

        if PROMPT_MODE in _NEW_SCHEMA_MODES:
            # New schema: strategy_summary string + per-hour rationale
            strategy_summary = (
                f"{rationale.get('market_assessment', '')} "
                f"{rationale.get('profit_tradeoff', '')} "
                f"{rationale.get('final_justification', '')}"
            ).strip()
            for entry in schedule:
                if "rationale" not in entry:
                    entry["rationale"] = "mock"
            response = {
                "firm_id":               self.firm_id,
                "date":                  date,
                "strategy_summary":      strategy_summary,
                "hourly_markup_schedule": schedule,
                "confidence":            confidence,
            }
        else:
            response = {
                "firm_id":               self.firm_id,
                "date":                  date,
                "decision_rationale":    rationale,
                "hourly_markup_schedule": schedule,
                "confidence":            confidence,
            }
        tracking = {
            "agent_type":              "mock",
            "model":                   "",
            "temperature":             "",
            "max_tokens":              "",
            "raw_agent_response_text": json.dumps(response),
            "parse_error":             "",
            "retry_used":              False,
            "fallback_used":           False,
        }
        return response, tracking


# ============================================================
# OPENAI HELPERS
# ============================================================

def call_openai_json(client, model, prompt, temperature, max_tokens, timeout):
    # gpt-5.x returns '' (empty string) when response_format=json_object is set,
    # instead of raising an error.  Skip the json_object attempt entirely for
    # these models to avoid a wasted round-trip on every call.
    _model_needs_no_format = model.startswith("gpt-5")

    # Newer models (gpt-5.x, o-series) use max_completion_tokens instead of max_tokens.
    # Try max_tokens first; on BadRequestError about the param, retry with the new name.
    _token_kwarg = {"max_tokens": max_tokens}

    base_kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": OPENAI_SYSTEM_MSG},
            {"role": "user",   "content": prompt},
        ],
        temperature=temperature,
        timeout=timeout,
        **_token_kwarg,
    )

    def _call_and_extract(kwargs, use_json_format=True):
        """Make one API call and return the content string.

        Handles three failure modes specific to GPT-5+ models:
          1. max_tokens → max_completion_tokens rename
          2. response_format=json_object not supported → retry without it
          3. model returns null/empty content (None or '') with refusal in msg.refusal
        Raises ValueError with a descriptive message on any unrecoverable failure.
        """
        call_kwargs = dict(kwargs)
        if use_json_format:
            resp = client.chat.completions.create(
                **call_kwargs, response_format={"type": "json_object"}
            )
        else:
            resp = client.chat.completions.create(**call_kwargs)

        msg = resp.choices[0].message
        content = msg.content

        # Catches both None and '' (empty string).  gpt-5.5 returns '' instead of
        # raising when response_format=json_object is not supported.
        if not content:
            refusal = getattr(msg, "refusal", None)
            if refusal:
                raise ValueError(
                    f"Model refused request (content policy). "
                    f"Refusal: {str(refusal)[:300]}"
                )
            raise ValueError(
                "MODEL_EMPTY_CONTENT: null or empty response. "
                "response_format=json_object may be unsupported by this model."
            )
        return content

    def _do_one_call():
        """Execute the call sequence (format-fallback logic) without rate-limit handling."""
        if _model_needs_no_format:
            try:
                return _call_and_extract(base_kwargs, use_json_format=False)
            except Exception as _exc:
                _exc_str = str(_exc).lower()
                if "max_tokens" in _exc_str and "max_completion_tokens" in _exc_str:
                    _token_kwarg2 = {"max_completion_tokens": max_tokens}
                    base_kwargs.pop("max_tokens", None)
                    base_kwargs.update(_token_kwarg2)
                    return _call_and_extract(base_kwargs, use_json_format=False)
                raise

        # Attempt 1: standard call with json_object format
        try:
            return _call_and_extract(base_kwargs, use_json_format=True)
        except Exception as _exc:
            _exc_str = str(_exc).lower()

            # Attempt 2: max_tokens → max_completion_tokens
            if "max_tokens" in _exc_str and "max_completion_tokens" in _exc_str:
                _token_kwarg2 = {"max_completion_tokens": max_tokens}
                base_kwargs.pop("max_tokens", None)
                base_kwargs.update(_token_kwarg2)
                try:
                    return _call_and_extract(base_kwargs, use_json_format=True)
                except Exception as _exc2:
                    _exc_str2 = str(_exc2).lower()
                    if ("response_format" in _exc_str2 or "empty_content" in _exc_str2
                            or "refused" in _exc_str2):
                        return _call_and_extract(base_kwargs, use_json_format=False)
                    raise

            # Attempt 3: drop response_format
            if ("empty_content" in _exc_str or "refused" in _exc_str
                    or "response_format" in _exc_str or "json_object" in _exc_str):
                return _call_and_extract(base_kwargs, use_json_format=False)

            raise

    # Retry loop for 429 Rate Limit errors.
    # The orchestrator sets --sleep-seconds 65 for gpt-5.x to keep under 1 RPM,
    # so retries here are a safety net only.  Fixed 65s wait keeps total bounded.
    _max_rl_retries = 3
    _rl_wait        = 65   # seconds; fixed wait matches the inter-call sleep
    for _attempt in range(_max_rl_retries + 1):
        try:
            return _do_one_call()
        except Exception as _exc:
            _exc_s = str(_exc).lower()
            _is_rate_limit = ("rate limit" in _exc_s or "ratelimit" in _exc_s
                              or "error code: 429" in _exc_s or "'code': '429'" in _exc_s
                              or "requests per minute" in _exc_s or "tokens per minute" in _exc_s)
            if _is_rate_limit and _attempt < _max_rl_retries:
                print(f"  [rate-limit] {model} 429 on attempt {_attempt+1}; "
                      f"sleeping {_rl_wait}s before retry ...", flush=True)
                time.sleep(_rl_wait)
                continue
            raise


def call_openai_responses_daily(client, model, prompt, prompt_mode,
                                reasoning_effort, max_output_tokens, timeout,
                                schema_profile=None):
    """Call the Responses API for daily-schedule mode with strict structured output."""
    schema, _ = _daily_schedule_schema(schema_profile)
    _rl_wait = 65
    _max_rl_retries = 3

    def _do_call():
        resp = client.responses.create(
            model=model,
            instructions=OPENAI_SYSTEM_MSG,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "daily_bidding_schedule",
                    "strict": True,
                    "schema": schema,
                }
            },
            reasoning={"effort": reasoning_effort},
            max_output_tokens=max_output_tokens,
            store=False,
            timeout=timeout,
        )
        text = resp.output_text
        if not text:
            raise ValueError(
                "RESPONSES_API_EMPTY: output_text is empty or None — "
                "possible refusal or truncation"
            )
        return text

    for attempt in range(_max_rl_retries + 1):
        try:
            return _do_call()
        except Exception as exc:
            exc_s = str(exc).lower()
            is_rl = (
                "rate limit" in exc_s or "ratelimit" in exc_s
                or "error code: 429" in exc_s or "requests per minute" in exc_s
                or "tokens per minute" in exc_s
            )
            if is_rl and attempt < _max_rl_retries:
                print(f"  [rate-limit] {model} 429 on attempt {attempt+1}; "
                      f"sleeping {_rl_wait}s before retry ...", flush=True)
                time.sleep(_rl_wait)
                continue
            raise


def parse_json_response(raw_text):
    if not raw_text:
        return None, "Empty response"
    text = raw_text.strip()
    try:
        return json.loads(text), ""
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1]), ""
        except json.JSONDecodeError as e:
            return None, f"JSON parse error after brace extraction: {e}"
    return None, "No JSON object found in response"


def _daily_schedule_schema(schema_profile):
    profile = (schema_profile or SCHEMA_PROFILE or "").strip().lower()
    if profile in {"fish", "fish_structured"}:
        return _DAILY_SCHEDULE_FISH_RESPONSES, "fish"
    if profile in {"tarj"}:
        return _DAILY_SCHEDULE_TARJ_RESPONSES, "tarj"
    if profile in {"decision_rationale", "legacy", "legacy_decision_rationale"}:
        return _DAILY_SCHEDULE_OLD_SCHEMA_RESPONSES, "decision_rationale"
    if profile in {"summary", "strategy_summary"}:
        return _DAILY_SCHEDULE_SUMMARY_RESPONSES, "summary"
    if profile in {"flexible", "tarj_flexible"}:
        return _DAILY_SCHEDULE_FLEXIBLE_RESPONSES, "flexible"
    raise ValueError(
        f"Unknown schema profile: {schema_profile!r}. "
        "Use fish, tarj, decision_rationale, summary, or flexible."
    )


def build_retry_prompt(original_prompt):
    return (
        original_prompt
        + "\n\nYour previous response was not valid JSON. "
        "Return ONLY valid JSON matching the required schema. "
        "Do not include markdown, code blocks, or any text outside the JSON object."
    )


def make_fallback_response(firm_id, round_info, error_msg=""):
    truncated_err = (error_msg[:120] + "...") if len(error_msg) > 120 else error_msg
    rationale = {
        "market_assessment":   "Response parsing failed; using fallback truthful bid.",
        "profit_tradeoff":     "Cannot assess profit tradeoff; fallback to markup 1.0.",
        "risk_assessment":     f"Parse error: {truncated_err}" if truncated_err else "JSON parsing failed.",
        "final_justification": "Fallback bid: markup_factor=1.0 (truthful).",
    }
    if PROMPT_MODE in _CUMULATIVE_MODES:
        rationale["learning_from_history"] = "No valid response received; fallback used."
    return {
        "firm_id":            firm_id,
        "round":              round_info["round"],
        "decision_rationale": rationale,
        "markup_factor":      1.0,
        "confidence":         "low",
    }


# ============================================================
# OPENAI FIRM AGENT
# ============================================================

class OpenAIFirmAgent(BaseFirmAgent):
    def __init__(self, firm_id, portfolio_df, allowed_markup_factors,
                 model="gpt-4.1", temperature=0.2, max_tokens=800,
                 api_timeout=60, sleep_seconds=0.2, seed=None):
        super().__init__(firm_id, portfolio_df, allowed_markup_factors, seed)
        self.model         = model
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.api_timeout   = api_timeout
        self.sleep_seconds = sleep_seconds
        # max_retries=0: we handle rate-limit retries ourselves in call_openai_json
        # with longer backoff windows suited for strict RPM limits.
        self.client        = _openai_lib.OpenAI(max_retries=0)

    def decide(self, round_info, recent_public_history, own_recent_performance,
               rendered_prompt=None, memory_summary=None):
        if rendered_prompt is None:
            raise ValueError("OpenAIFirmAgent.decide() requires rendered_prompt")

        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

        raw_text      = ""
        parse_error   = ""
        retry_used    = False
        fallback_used = False
        parsed        = None
        valid         = False

        try:
            raw_text = call_openai_json(
                self.client, self.model, rendered_prompt,
                self.temperature, self.max_tokens, self.api_timeout,
            )
            parsed, parse_error = parse_json_response(raw_text)
        except Exception as exc:
            parse_error = str(exc)

        if parsed is not None:
            valid, schema_err = validate_agent_response(
                parsed, self.firm_id, self.allowed_markup_factors, PROMPT_MODE
            )
            if not valid:
                parse_error = schema_err

        if not valid:
            retry_used = True
            try:
                retry_raw = call_openai_json(
                    self.client, self.model,
                    build_retry_prompt(rendered_prompt),
                    self.temperature, self.max_tokens, self.api_timeout,
                )
                raw_text = retry_raw
                parsed2, err2 = parse_json_response(raw_text)
                if parsed2 is not None:
                    valid2, err3 = validate_agent_response(
                        parsed2, self.firm_id, self.allowed_markup_factors, PROMPT_MODE
                    )
                    if valid2:
                        parsed      = parsed2
                        parse_error = ""
                        valid       = True
                    else:
                        parse_error = err3
                else:
                    parse_error = err2
            except Exception as exc:
                parse_error = str(exc)

        if not valid:
            fallback_used = True
            markup   = 1.0
            response = make_fallback_response(self.firm_id, round_info, parse_error)
        else:
            markup   = float(parsed["markup_factor"])
            response = dict(parsed)
            response["firm_id"] = self.firm_id
            response["round"]   = round_info["round"]

        tracking = {
            "agent_type":              "openai",
            "model":                   self.model,
            "temperature":             self.temperature,
            "max_tokens":              self.max_tokens,
            "raw_agent_response_text": raw_text,
            "parse_error":             parse_error,
            "retry_used":              retry_used,
            "fallback_used":           fallback_used,
        }
        return markup, response, tracking


    def decide_daily(self, day_info, daily_load_forecast, recent_public_history,
                     own_recent_performance, rendered_prompt=None, memory_summary=None):
        if rendered_prompt is None:
            raise ValueError("OpenAIFirmAgent.decide_daily() requires rendered_prompt")

        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

        raw_text      = ""
        parse_error   = ""
        retry_used    = False
        fallback_used = False
        parsed        = None
        valid         = False

        if USE_RESPONSES_API:
            # Responses API path: structured output eliminates most parse failures.
            # No format-retry needed; if the model output is invalid we still
            # validate and fall back once with a plain retry.
            try:
                raw_text = call_openai_responses_daily(
                    self.client, self.model, rendered_prompt, PROMPT_MODE,
                    REASONING_EFFORT, self.max_tokens, self.api_timeout,
                    SCHEMA_PROFILE,
                )
                parsed, parse_error = parse_json_response(raw_text)
            except Exception as exc:
                parse_error = str(exc)

            if parsed is not None:
                valid, schema_err = validate_daily_schedule_response(
                    parsed, self.firm_id, str(day_info["date"]),
                    self.allowed_markup_factors, PROMPT_MODE, SCHEMA_PROFILE,
                )
                if not valid:
                    parse_error = schema_err

            if not valid:
                retry_used = True
                try:
                    raw_text = call_openai_responses_daily(
                        self.client, self.model,
                        build_retry_prompt(rendered_prompt), PROMPT_MODE,
                        REASONING_EFFORT, self.max_tokens, self.api_timeout,
                        SCHEMA_PROFILE,
                    )
                    parsed2, err2 = parse_json_response(raw_text)
                    if parsed2 is not None:
                        valid2, err3 = validate_daily_schedule_response(
                            parsed2, self.firm_id, str(day_info["date"]),
                            self.allowed_markup_factors, PROMPT_MODE, SCHEMA_PROFILE,
                        )
                        if valid2:
                            parsed      = parsed2
                            parse_error = ""
                            valid       = True
                        else:
                            parse_error = err3
                    else:
                        parse_error = err2
                except Exception as exc:
                    parse_error = str(exc)
        else:
            # Original Chat Completions path
            try:
                raw_text = call_openai_json(
                    self.client, self.model, rendered_prompt,
                    self.temperature, self.max_tokens, self.api_timeout,
                )
                parsed, parse_error = parse_json_response(raw_text)
            except Exception as exc:
                parse_error = str(exc)

            if parsed is not None:
                valid, schema_err = validate_daily_schedule_response(
                    parsed, self.firm_id, str(day_info["date"]),
                    self.allowed_markup_factors, PROMPT_MODE, SCHEMA_PROFILE,
                )
                if not valid:
                    parse_error = schema_err

            if not valid:
                retry_used = True
                try:
                    retry_raw = call_openai_json(
                        self.client, self.model,
                        build_retry_prompt(rendered_prompt),
                        self.temperature, self.max_tokens, self.api_timeout,
                    )
                    raw_text = retry_raw
                    parsed2, err2 = parse_json_response(raw_text)
                    if parsed2 is not None:
                        valid2, err3 = validate_daily_schedule_response(
                            parsed2, self.firm_id, str(day_info["date"]),
                            self.allowed_markup_factors, PROMPT_MODE, SCHEMA_PROFILE,
                        )
                        if valid2:
                            parsed      = parsed2
                            parse_error = ""
                            valid       = True
                        else:
                            parse_error = err3
                    else:
                        parse_error = err2
                except Exception as exc:
                    parse_error = str(exc)

        if not valid:
            fallback_used = True
            response = make_daily_schedule_fallback_response(
                self.firm_id, str(day_info["date"]), parse_error, SCHEMA_PROFILE
            )
        else:
            response = dict(parsed)
            response["firm_id"] = self.firm_id
            response["date"]    = str(day_info["date"])

        tracking = {
            "agent_type":              "openai",
            "model":                   self.model,
            "temperature":             self.temperature,
            "max_tokens":              self.max_tokens,
            "raw_agent_response_text": raw_text,
            "parse_error":             parse_error,
            "retry_used":              retry_used,
            "fallback_used":           fallback_used,
        }
        return response, tracking


class FutureLLMFirmAgent(BaseFirmAgent):
    """Stub for additional LLM providers."""
    def __init__(self, firm_id, portfolio_df, allowed_markup_factors,
                 model="claude-sonnet-4-6", temperature=0.3, seed=None):
        super().__init__(firm_id, portfolio_df, allowed_markup_factors, seed)
        self.model       = model
        self.temperature = temperature

    def decide(self, round_info, recent_public_history, own_recent_performance,
               rendered_prompt=None, memory_summary=None):
        raise NotImplementedError("FutureLLMFirmAgent is a stub.")


# ============================================================
# INPUT LOADING
# ============================================================

def load_inputs(gen_file, round_file, truthful_market_file, truthful_firm_file,
                n_rounds, stress_case):
    print("\n=== Loading inputs ===")
    for f in [gen_file, round_file, truthful_market_file, truthful_firm_file]:
        if not Path(f).exists():
            print(f"[ERROR] Missing input file: {f}")
            sys.exit(1)

    gen_df             = pd.read_csv(gen_file)
    rounds_df          = pd.read_csv(round_file).head(n_rounds)
    truthful_market_df = pd.read_csv(truthful_market_file)
    truthful_firm_df   = pd.read_csv(truthful_firm_file)

    n_bs = int((gen_df["gen_id"] == BACKSTOP_GEN_ID).sum())
    print(f"  Generators : {len(gen_df)} total  ({n_bs} backstop, {len(gen_df)-n_bs} thermal)")
    print(f"  Rounds     : {len(rounds_df)}  (stress case: {stress_case})")
    print(f"  Load range : {rounds_df['load_mw'].min():,.0f} - {rounds_df['load_mw'].max():,.0f} MW")
    return gen_df, rounds_df, truthful_market_df, truthful_firm_df


# ============================================================
# PROMPT UTILITIES
# ============================================================

def load_prompt_template(prompt_mode, prompt_file_override=None):
    if prompt_file_override:
        path = Path(prompt_file_override)
        if not path.is_absolute():
            candidate = PROJECT_ROOT / path
            if candidate.exists():
                path = candidate
            else:
                candidate = WORKSPACE_ROOT / path
                if candidate.exists():
                    path = candidate
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8"), str(path)

    fname = PROMPT_FILES.get(prompt_mode)
    if fname is None:
        raise ValueError(f"Unknown prompt mode: {prompt_mode!r}. "
                         f"Valid modes: {list(PROMPT_FILES)}")
    for base in (WORKSPACE_PROMPTS_DIR, LEGACY_PROMPTS_DIR):
        path = base / fname
        if path.exists():
            return path.read_text(encoding="utf-8"), str(path)
    raise FileNotFoundError(
        f"Prompt file not found in {WORKSPACE_PROMPTS_DIR} or {LEGACY_PROMPTS_DIR}: {fname}"
    )


def compute_prompt_hash(prompt_text):
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


# ============================================================
# FORMATTING FUNCTIONS
# ============================================================

def format_generator_portfolio(portfolio_df):
    lines = [
        f"{'gen_id':<22} {'technology':<12} {'pmax_mw':>9}  {'true_mc_$/MWh':>14}",
        "-" * 62,
    ]
    for _, row in portfolio_df.iterrows():
        lines.append(
            f"{str(row['gen_id']):<22} {str(row['technology_group']):<12}"
            f" {row['pmax_mw']:>9.1f}  {row['true_mc_per_mwh']:>14.4f}"
        )
    total_mw = portfolio_df["pmax_mw"].sum()
    lines.append(f"({len(portfolio_df)} generators, total {total_mw:,.1f} MW)")
    return "\n".join(lines)


def format_round_info(round_info):
    """Render round_info dict as human-readable text for prompt injection.
    Includes optional chronological datetime fields when present."""
    lines = [
        f"Round: {round_info['round']}",
        f"Date: {round_info['date']}",
    ]
    if "day_of_week" in round_info:
        lines.append(f"Day of week: {round_info['day_of_week']}")
    if "month" in round_info:
        m = int(round_info["month"])
        lines.append(f"Month: {_MONTH_NAMES.get(m, str(m))}")
    if "season" in round_info:
        lines.append(f"Season: {round_info['season']}")
    lines.extend([
        f"Hour: {round_info['hour']}:00",
        f"Load forecast: {round_info['load_mw']:,.1f} MW",
        f"Price cap: ${round_info['price_cap']:,.0f}/MWh",
    ])
    return "\n".join(lines)


def format_recent_public_history(public_history):
    if not public_history:
        return "No previous market history available."
    lines = [
        f"{'Round':>6}  {'Load(MW)':>10}  {'Price($/MWh)':>13}  {'Backstop':>9}  {'Marginal Firm':<20}",
        "-" * 66,
    ]
    for h in public_history:
        bs = "Yes" if h.get("backstop_dispatched") else "No"
        lines.append(
            f"{h['round']:>6}  {h['load_mw']:>10,.1f}  "
            f"{h['market_price']:>13.2f}  {bs:>9}  {h['marginal_firm_id']:<20}"
        )
    return "\n".join(lines)


def format_own_recent_performance(own_history):
    if not own_history:
        return "No previous performance data available."
    lines = [
        f"{'Round':>6}  {'Markup':>7}  {'Dispatch(MW)':>13}  "
        f"{'Revenue($)':>12}  {'Cost($)':>10}  {'Profit($)':>11}  {'Marginal':>9}",
        "-" * 78,
    ]
    for h in own_history:
        marg = "Yes" if h.get("was_marginal") else "No"
        lines.append(
            f"{h['round']:>6}  {h['markup_factor']:>7.2f}  "
            f"{h['dispatch_mw']:>13,.1f}  "
            f"{h['revenue']:>12,.0f}  {h['production_cost']:>10,.0f}  "
            f"{h['profit']:>11,.0f}  {marg:>9}"
        )
    return "\n".join(lines)


def format_memory_summary(own_history):
    if not own_history:
        return "No previous market experience to summarise."
    markups    = [h["markup_factor"] for h in own_history]
    profits    = [h["profit"] for h in own_history]
    dispatched = sum(1 for h in own_history if h["dispatch_mw"] > DISPATCH_TOLERANCE)
    best_idx    = int(np.argmax(profits))
    avg_markup  = float(np.mean(markups))
    avg_profit  = float(np.mean(profits))
    best_markup = markups[best_idx]
    best_profit = profits[best_idx]
    return (
        f"Over {len(own_history)} rounds: avg markup={avg_markup:.2f}, "
        f"avg profit=${avg_profit:,.0f}, dispatched in {dispatched}/{len(own_history)} rounds. "
        f"Best result: markup={best_markup:.2f} -> profit=${best_profit:,.0f}."
    )


def render_prompt(prompt_template, firm_id, portfolio_df, round_info,
                  recent_public_history, own_recent_performance,
                  allowed_markup_factors, memory_summary=None):
    text       = prompt_template
    markup_str = ", ".join(str(m) for m in allowed_markup_factors)

    text = text.replace("{firm_id}",                      firm_id)
    text = text.replace("{generator_portfolio_table}",    format_generator_portfolio(portfolio_df))
    text = text.replace("{round_info}",                   format_round_info(round_info))
    text = text.replace("{allowed_markup_factors}",       markup_str)
    text = text.replace("{recent_public_market_history}", format_recent_public_history(recent_public_history))
    text = text.replace("{own_recent_performance}",       format_own_recent_performance(own_recent_performance))
    text = text.replace("{memory_summary}",               memory_summary or "No memory available.")
    text = text.replace("{round}",                        str(round_info["round"]))

    return text


# ============================================================
# DAILY-SCHEDULE FORMATTING HELPERS
# ============================================================

def format_day_info(day_info):
    lines = [
        f"Date: {day_info['date']}",
    ]
    if "day_of_week" in day_info:
        lines.append(f"Day of week: {day_info['day_of_week']}")
    if "month" in day_info:
        m = int(day_info["month"])
        lines.append(f"Month: {_MONTH_NAMES.get(m, str(m))}")
    if "season" in day_info:
        lines.append(f"Season: {day_info['season']}")
    lines.extend([
        f"Number of hourly markets today: 24",
        f"Price cap: ${day_info['price_cap']:,.0f}/MWh",
    ])
    return "\n".join(lines)


def format_daily_load_forecast_table(daily_load_forecast, load_percentiles=None):
    lines = [
        f"{'Hour':>5}  {'Load(MW)':>10}",
        "-" * 22,
    ]
    p75 = load_percentiles["p75"] if load_percentiles else float("inf")
    p90 = load_percentiles["p90"] if load_percentiles else float("inf")
    for h, load_mw in daily_load_forecast:
        if load_percentiles:
            if load_mw > p90:
                tag = "  [peak]"
            elif load_mw > p75:
                tag = "  [high]"
            else:
                tag = ""
            lines.append(f"{h:>5}  {load_mw:>10,.1f}{tag}")
        else:
            lines.append(f"{h:>5}  {load_mw:>10,.1f}")
    return "\n".join(lines)


def compute_market_position_diagnostics(firm_id, own_history, public_history, history_window):
    """Return a formatted diagnostics string for firm_id from prior-day history only."""
    if not own_history:
        return ("No previous market-position diagnostics are available because "
                "this is the first operating day.")

    # Match own_history entries with public_history by round to get market prices
    pub_by_round = {h["round"]: h for h in public_history}

    n = len(own_history)
    dispatched_rows   = [h for h in own_history if h.get("dispatch_mw", 0) > DISPATCH_TOLERANCE]
    marginal_rows     = [h for h in own_history if h.get("was_marginal", False)]
    dispatch_share    = len(dispatched_rows) / n
    marginal_share    = len(marginal_rows)  / n

    avg_dispatch_mw = float(np.mean([h["dispatch_mw"] for h in dispatched_rows])) \
                      if dispatched_rows else 0.0

    market_prices_dispatched = [
        pub_by_round[h["round"]]["market_price"]
        for h in dispatched_rows
        if h["round"] in pub_by_round
    ]
    market_prices_marginal = [
        pub_by_round[h["round"]]["market_price"]
        for h in marginal_rows
        if h["round"] in pub_by_round
    ]

    avg_price_dispatched = float(np.mean(market_prices_dispatched)) \
                           if market_prices_dispatched else float("nan")
    avg_price_marginal   = float(np.mean(market_prices_marginal)) \
                           if market_prices_marginal else float("nan")

    profits       = [h["profit"] for h in own_history]
    markups       = [h["markup_factor"] for h in own_history]
    avg_profit    = float(np.mean(profits))
    avg_markup    = float(np.mean(markups))

    profits_dispatched = [h["profit"] for h in dispatched_rows]
    avg_profit_disp    = float(np.mean(profits_dispatched)) if profits_dispatched else float("nan")

    max_profit_idx = int(np.argmax(profits)) if profits else 0
    max_profit_hr  = own_history[max_profit_idx]["round"] if profits else "N/A"

    share_mu_ge_1_5 = sum(1 for m in markups if m >= 1.5) / n
    share_mu_ge_2   = sum(1 for m in markups if m >= 2.0) / n

    # Previous-day stats (last date in own_history)
    dates_seen = sorted({h["date"] for h in own_history})
    prev_date  = dates_seen[-1] if dates_seen else None
    prev_rows  = [h for h in own_history if h.get("date") == prev_date] if prev_date else []
    if prev_rows:
        prev_avg_markup = float(np.mean([h["markup_factor"] for h in prev_rows]))
        prev_avg_profit = float(np.mean([h["profit"] for h in prev_rows]))
        prev_disp_share = sum(1 for h in prev_rows if h.get("dispatch_mw", 0) > DISPATCH_TOLERANCE) / len(prev_rows)
        prev_marg_share = sum(1 for h in prev_rows if h.get("was_marginal", False)) / len(prev_rows)
        prev_date_str   = str(prev_date)
    else:
        prev_avg_markup = prev_avg_profit = prev_disp_share = prev_marg_share = float("nan")
        prev_date_str   = "N/A"

    def _pct(v):
        return f"{v:.0%}" if not (isinstance(v, float) and np.isnan(v)) else "not available"
    def _dollar(v):
        return f"${v:,.0f}" if not (isinstance(v, float) and np.isnan(v)) else "not available"
    def _mw(v):
        return f"{v:,.1f} MW" if not (isinstance(v, float) and np.isnan(v)) else "not available"
    def _price(v):
        return f"${v:.2f}/MWh" if not (isinstance(v, float) and np.isnan(v)) else "not available"

    lines = [
        f"Previous-history window: last {n} hours (up to {history_window} hours).",
        "",
        "Your recent market position:",
        f"- Dispatched in {_pct(dispatch_share)} of previous hours.",
        f"- Marginal price-setter in {_pct(marginal_share)} of previous hours.",
        f"- Average dispatch when dispatched: {_mw(avg_dispatch_mw)}.",
        f"- Average realized market price when dispatched: {_price(avg_price_dispatched)}.",
        f"- Average realized market price when marginal: {_price(avg_price_marginal)}.",
        f"- Average profit per hour: {_dollar(avg_profit)}.",
        f"- Average profit per dispatched hour: {_dollar(avg_profit_disp)}.",
        f"- Average markup used: {avg_markup:.2f}.",
        f"- Share of previous hours with markup >= 1.5: {_pct(share_mu_ge_1_5)}.",
        f"- Share of previous hours with markup >= 2.0: {_pct(share_mu_ge_2)}.",
        "",
        f"Immediate previous day ({prev_date_str}):",
        f"- Average markup: {prev_avg_markup:.2f}." if not np.isnan(prev_avg_markup) else "- Average markup: not available.",
        f"- Average profit per hour: {_dollar(prev_avg_profit)}.",
        f"- Dispatch share: {_pct(prev_disp_share)}.",
        f"- Marginal share: {_pct(prev_marg_share)}.",
        "",
        "Interpretation guidance:",
        "- Frequent marginal status indicates potential price-setting opportunity, "
          "but high markups may lose dispatch.",
        "- Frequent dispatched but non-marginal status indicates inframarginal rent opportunity "
          "if market prices rise.",
        "- Use this information only for your own current-day profit; "
          "do not coordinate with competitors.",
    ]
    return "\n".join(lines)


def compute_neutral_market_position_diagnostics(firm_id, own_history, public_history, history_window):
    """Return neutral factual diagnostics without cautionary framing.

    Same statistics as compute_market_position_diagnostics() but the interpretation
    section presents facts only, with no risk/caution language.
    """
    if not own_history:
        return ("No previous market-position data available because "
                "this is the first operating day.")

    pub_by_round = {h["round"]: h for h in public_history}

    n = len(own_history)
    dispatched_rows   = [h for h in own_history if h.get("dispatch_mw", 0) > DISPATCH_TOLERANCE]
    marginal_rows     = [h for h in own_history if h.get("was_marginal", False)]
    dispatch_share    = len(dispatched_rows) / n
    marginal_share    = len(marginal_rows)  / n

    avg_dispatch_mw = float(np.mean([h["dispatch_mw"] for h in dispatched_rows])) \
                      if dispatched_rows else 0.0

    market_prices_dispatched = [
        pub_by_round[h["round"]]["market_price"]
        for h in dispatched_rows if h["round"] in pub_by_round
    ]
    market_prices_marginal = [
        pub_by_round[h["round"]]["market_price"]
        for h in marginal_rows if h["round"] in pub_by_round
    ]

    avg_price_dispatched = float(np.mean(market_prices_dispatched)) \
                           if market_prices_dispatched else float("nan")
    avg_price_marginal   = float(np.mean(market_prices_marginal)) \
                           if market_prices_marginal else float("nan")

    profits    = [h["profit"] for h in own_history]
    markups    = [h["markup_factor"] for h in own_history]
    avg_profit = float(np.mean(profits))
    avg_markup = float(np.mean(markups))

    profits_dispatched = [h["profit"] for h in dispatched_rows]
    avg_profit_disp    = float(np.mean(profits_dispatched)) if profits_dispatched else float("nan")

    share_mu_ge_1_5 = sum(1 for m in markups if m >= 1.5) / n
    share_mu_ge_2   = sum(1 for m in markups if m >= 2.0) / n

    dates_seen = sorted({h["date"] for h in own_history})
    prev_date  = dates_seen[-1] if dates_seen else None
    prev_rows  = [h for h in own_history if h.get("date") == prev_date] if prev_date else []
    if prev_rows:
        prev_avg_markup = float(np.mean([h["markup_factor"] for h in prev_rows]))
        prev_avg_profit = float(np.mean([h["profit"] for h in prev_rows]))
        prev_disp_share = sum(1 for h in prev_rows if h.get("dispatch_mw", 0) > DISPATCH_TOLERANCE) / len(prev_rows)
        prev_marg_share = sum(1 for h in prev_rows if h.get("was_marginal", False)) / len(prev_rows)
        prev_date_str   = str(prev_date)
    else:
        prev_avg_markup = prev_avg_profit = prev_disp_share = prev_marg_share = float("nan")
        prev_date_str   = "N/A"

    def _pct(v):
        return f"{v:.0%}" if not (isinstance(v, float) and np.isnan(v)) else "not available"
    def _dollar(v):
        return f"${v:,.0f}" if not (isinstance(v, float) and np.isnan(v)) else "not available"
    def _mw(v):
        return f"{v:,.1f} MW" if not (isinstance(v, float) and np.isnan(v)) else "not available"
    def _price(v):
        return f"${v:.2f}/MWh" if not (isinstance(v, float) and np.isnan(v)) else "not available"

    lines = [
        f"Previous-history window: last {n} hours (up to {history_window} hours).",
        "",
        "Your recent market position:",
        f"- Dispatch share: {_pct(dispatch_share)}.",
        f"- Price-setting (marginal) share: {_pct(marginal_share)}.",
        f"- Average dispatch volume when dispatched: {_mw(avg_dispatch_mw)}.",
        f"- Average market price in hours dispatched: {_price(avg_price_dispatched)}.",
        f"- Average market price in hours as marginal unit: {_price(avg_price_marginal)}.",
        f"- Average profit per hour (all hours): {_dollar(avg_profit)}.",
        f"- Average profit per dispatched hour: {_dollar(avg_profit_disp)}.",
        f"- Average markup factor used: {avg_markup:.2f}.",
        f"- Share of hours with markup >= 1.5: {_pct(share_mu_ge_1_5)}.",
        f"- Share of hours with markup >= 2.0: {_pct(share_mu_ge_2)}.",
        "",
        f"Immediate previous day ({prev_date_str}):",
        f"- Average markup: {prev_avg_markup:.2f}." if not np.isnan(prev_avg_markup) else "- Average markup: not available.",
        f"- Average profit per hour: {_dollar(prev_avg_profit)}.",
        f"- Dispatch share: {_pct(prev_disp_share)}.",
        f"- Price-setting share: {_pct(prev_marg_share)}.",
    ]
    return "\n".join(lines)


def render_daily_schedule_prompt(prompt_template, firm_id, portfolio_df, day_info,
                                  daily_load_forecast, recent_public_history,
                                  own_recent_performance, allowed_markup_factors,
                                  memory_summary=None, load_percentiles=None,
                                  market_position_diagnostics=None,
                                  plans_memory=None, insights_memory=None,
                                  journal_memory=None):
    text       = prompt_template
    markup_str = ", ".join(str(m) for m in allowed_markup_factors)

    text = text.replace("{firm_id}",                      firm_id)
    text = text.replace("{generator_portfolio_table}",    format_generator_portfolio(portfolio_df))
    text = text.replace("{day_info}",                     format_day_info(day_info))
    text = text.replace("{daily_load_forecast_table}",
                         format_daily_load_forecast_table(daily_load_forecast, load_percentiles))
    text = text.replace("{allowed_markup_factors}",       markup_str)
    text = text.replace("{recent_public_market_history}", format_recent_public_history(recent_public_history))
    text = text.replace("{own_recent_performance}",       format_own_recent_performance(own_recent_performance))
    text = text.replace("{memory_summary}",               memory_summary or "No memory available.")
    text = text.replace("{plans_memory}",                 plans_memory or "No previous plans recorded.")
    text = text.replace("{insights_memory}",              insights_memory or "No previous insights recorded.")
    text = text.replace("{journal_memory}",               journal_memory or "No previous journal summary available.")
    if "{market_position_diagnostics}" in text:
        text = text.replace("{market_position_diagnostics}",
                            market_position_diagnostics or
                            "No previous market-position diagnostics are available because "
                            "this is the first operating day.")
    text = text.replace("{date}",                         str(day_info["date"]))
    return text


# ============================================================
# DAILY-SCHEDULE RESPONSE VALIDATION
# ============================================================

def validate_daily_schedule_response(
    response, firm_id, date, allowed_markup_factors, prompt_mode, schema_profile=None
):
    for k in ("firm_id", "date", "hourly_markup_schedule", "confidence"):
        if k not in response:
            return False, f"Missing required key: '{k}'"

    if response["firm_id"] != firm_id:
        return False, f"firm_id mismatch: got '{response['firm_id']}', expected '{firm_id}'"

    if str(response["date"]) != str(date):
        return False, f"date mismatch: got '{response['date']}', expected '{date}'"

    if response["confidence"] not in ("low", "medium", "high"):
        return False, "confidence must be 'low', 'medium', or 'high'"

    profile = (schema_profile or SCHEMA_PROFILE or "").strip().lower()
    if profile in {"fish", "fish_structured"}:
        required_keys = [
            "prompt_architecture",
            "objective_treatment",
            "observations_and_thoughts",
            "new_plans",
            "new_insights",
        ]
        for k in required_keys:
            if k not in response:
                return False, f"Missing required key: '{k}'"
        if response["prompt_architecture"] != "fish":
            return False, "prompt_architecture must be 'fish'"
    elif profile in {"tarj"}:
        required_keys = [
            "prompt_architecture",
            "objective_treatment",
            "thought",
            "action_summary",
            "reflection",
            "journal_update",
        ]
        for k in required_keys:
            if k not in response:
                return False, f"Missing required key: '{k}'"
        if response["prompt_architecture"] != "tarj":
            return False, "prompt_architecture must be 'tarj'"
    elif profile in {"summary", "strategy_summary"}:
        if "strategy_summary" not in response:
            return False, "Missing required key: 'strategy_summary'"
    elif profile in {"flexible", "tarj_flexible"}:
        pass
    else:
        if "decision_rationale" not in response:
            return False, "Missing required key: 'decision_rationale'"
        rationale = response.get("decision_rationale", {})
        base_keys = ["market_assessment", "profit_tradeoff", "risk_assessment", "final_justification"]
        if prompt_mode in _CUMULATIVE_MODES:
            base_keys.append("learning_from_history")
        for k in base_keys:
            if k not in rationale:
                return False, f"Missing decision_rationale key: '{k}'"

    schedule = response.get("hourly_markup_schedule", [])
    if len(schedule) != 24:
        return False, f"hourly_markup_schedule must have 24 entries, got {len(schedule)}"

    hours_seen = set()
    for entry in schedule:
        if "hour" not in entry or "markup_factor" not in entry:
            return False, "Each schedule entry must have 'hour' and 'markup_factor' keys"
        h = int(entry["hour"])
        if h in hours_seen:
            return False, f"Duplicate hour {h} in hourly_markup_schedule"
        hours_seen.add(h)
        if float(entry["markup_factor"]) not in allowed_markup_factors:
            return False, (
                f"markup_factor {entry['markup_factor']!r} in hour {h} "
                f"not in allowed list {allowed_markup_factors}"
            )

    if hours_seen != set(range(1, 25)):
        missing = set(range(1, 25)) - hours_seen
        return False, f"Missing hours in schedule: {sorted(missing)}"

    return True, ""


def make_daily_schedule_fallback_response(firm_id, date, error_msg="", schema_profile=None):
    truncated_err = (error_msg[:120] + "...") if len(error_msg) > 120 else error_msg
    profile = (schema_profile or SCHEMA_PROFILE or "").strip().lower()
    if profile in {"fish", "fish_structured"}:
        return {
            "firm_id":  firm_id,
            "date":     date,
            "prompt_architecture": "fish",
            "objective_treatment": "",
            "observations_and_thoughts": (
                f"Fallback truthful schedule due to parse error: {truncated_err}"
                if truncated_err else "Fallback truthful schedule: JSON parsing failed."
            ),
            "new_plans": "Fallback used; preserve recent profitable structure.",
            "new_insights": "Fallback used; preserve conservative dispatch risk management.",
            "hourly_markup_schedule": [
                {"hour": h, "markup_factor": 1.0, "rationale": "fallback"} for h in range(1, 25)
            ],
            "confidence": "low",
        }
    if profile in {"tarj"}:
        return {
            "firm_id":  firm_id,
            "date":     date,
            "prompt_architecture": "tarj",
            "objective_treatment": "",
            "thought": "Fallback truthful schedule due to parse error.",
            "action_summary": "Fallback used; preserve truthful bidding.",
            "reflection": (
                f"Parse error: {truncated_err}" if truncated_err else "JSON parsing failed."
            ),
            "journal_update": "Fallback used; continue with conservative bidding.",
            "hourly_markup_schedule": [
                {"hour": h, "markup_factor": 1.0, "rationale": "fallback"} for h in range(1, 25)
            ],
            "confidence": "low",
        }
    if profile in {"summary", "strategy_summary"}:
        return {
            "firm_id":  firm_id,
            "date":     date,
            "strategy_summary": (
                f"Fallback truthful schedule due to parse error: {truncated_err}"
                if truncated_err else "Fallback truthful schedule: JSON parsing failed."
            ),
            "hourly_markup_schedule": [
                {"hour": h, "markup_factor": 1.0, "rationale": "fallback"} for h in range(1, 25)
            ],
            "confidence": "low",
        }
    if profile in {"flexible", "tarj_flexible"}:
        return {
            "firm_id":  firm_id,
            "date":     date,
            "hourly_markup_schedule": [
                {"hour": h, "markup_factor": 1.0, "rationale": "fallback"} for h in range(1, 25)
            ],
            "confidence": "low",
            "fallback_note": (
                f"Parse error: {truncated_err}" if truncated_err else "JSON parsing failed."
            ),
        }
    rationale = {
        "market_assessment":   "Response parsing failed; using fallback truthful schedule.",
        "profit_tradeoff":     "Cannot assess profit tradeoff; fallback to all markup = 1.0.",
        "risk_assessment":     f"Parse error: {truncated_err}" if truncated_err else "JSON parsing failed.",
        "final_justification": "Fallback schedule: all 24 markup factors = 1.0 (truthful).",
    }
    if PROMPT_MODE in _CUMULATIVE_MODES:
        rationale["learning_from_history"] = "No valid response received; fallback used."
    return {
        "firm_id":  firm_id,
        "date":     date,
        "decision_rationale": rationale,
        "hourly_markup_schedule": [{"hour": h, "markup_factor": 1.0} for h in range(1, 25)],
        "confidence": "low",
    }


# ============================================================
# RESPONSE VALIDATION
# ============================================================

def validate_agent_response(response, firm_id, allowed_markup_factors, prompt_mode):
    for k in ("firm_id", "round", "decision_rationale", "markup_factor", "confidence"):
        if k not in response:
            return False, f"Missing required key: '{k}'"

    if response["firm_id"] != firm_id:
        return False, f"firm_id mismatch: got '{response['firm_id']}', expected '{firm_id}'"

    markup = response["markup_factor"]
    if markup not in allowed_markup_factors:
        return False, f"markup_factor {markup!r} not in allowed list {allowed_markup_factors}"

    if response["confidence"] not in ("low", "medium", "high"):
        return False, "confidence must be 'low', 'medium', or 'high'"

    rationale  = response.get("decision_rationale", {})
    base_keys  = ["market_assessment", "profit_tradeoff", "risk_assessment", "final_justification"]
    if prompt_mode in _CUMULATIVE_MODES:
        base_keys.append("learning_from_history")
    for k in base_keys:
        if k not in rationale:
            return False, f"Missing decision_rationale key: '{k}'"

    return True, ""


# ============================================================
# DISPATCH
# ============================================================

def dispatch_with_offers(gen_df, offer_prices, load_mw):
    offer_arr  = np.asarray(offer_prices, dtype=float)
    n          = len(gen_df)
    sort_idx   = np.argsort(offer_arr, kind="stable")
    stack      = gen_df.iloc[sort_idx].reset_index(drop=True).copy()
    stack_offs = offer_arr[sort_idx]

    dispatch_mw = np.zeros(n)
    remaining   = float(load_mw)
    marginal_i  = 0

    for i in range(n):
        if remaining <= DISPATCH_TOLERANCE:
            break
        d              = min(float(stack.at[i, "pmax_mw"]), remaining)
        dispatch_mw[i] = d
        remaining      -= d
        marginal_i     = i

    market_price = float(stack_offs[marginal_i])

    stack["dispatch_mw"]     = dispatch_mw
    stack["offer_price"]     = stack_offs
    stack["market_price"]    = market_price
    stack["revenue"]         = market_price * dispatch_mw
    stack["production_cost"] = stack["true_mc_per_mwh"].values * dispatch_mw
    stack["profit"]          = stack["revenue"] - stack["production_cost"]

    is_marginal             = np.zeros(n, dtype=bool)
    is_marginal[marginal_i] = True
    stack["is_marginal"]    = is_marginal
    stack["is_backstop"]    = stack["gen_id"] == BACKSTOP_GEN_ID

    bal_err = abs(dispatch_mw.sum() - load_mw)
    if bal_err > DISPATCH_TOLERANCE * 100:
        raise RuntimeError(f"Balance error: {bal_err:.4f} MW (load={load_mw:.2f})")
    if (dispatch_mw < -DISPATCH_TOLERANCE).any():
        raise RuntimeError("Negative dispatch detected")
    if (dispatch_mw > stack["pmax_mw"].values + DISPATCH_TOLERANCE).any():
        raise RuntimeError("Dispatch exceeds pmax")

    return stack, market_price, marginal_i


# ============================================================
# SIMULATION LOOP
# ============================================================

def run_simulation(gen_df, rounds_df, agents, prompt_template, prompt_file,
                   prompt_hash, truthful_market_df, truthful_firm_df, run_id):
    print(f"\n=== Running simulation: {run_id} ===")
    print(f"  Prompt mode : {PROMPT_MODE}  |  Agent type: {AGENT_TYPE}"
          + (f"  |  Mock mode: {MOCK_AGENT_MODE}" if AGENT_TYPE == "mock" else
             f"  |  Model: {MODEL}  |  Temp: {TEMPERATURE}"))
    print(f"  Seed        : {SEED}  |  Rounds: {len(rounds_df)}  |  Price cap: ${PRICE_CAP}/MWh")

    t_price   = dict(zip(truthful_market_df["round"].astype(int),
                         truthful_market_df["market_price"]))
    t_payment = dict(zip(truthful_market_df["round"].astype(int),
                         truthful_market_df["total_consumer_payment"]))
    t_firm_profit = {}
    for _, row in truthful_firm_df.iterrows():
        t_firm_profit[(int(row["round"]), str(row["firm_id"]))] = float(row["firm_profit"])

    public_history = []
    firm_histories = {fid: [] for fid in STRATEGIC_FIRMS}

    actions_rows   = []
    dispatch_rows  = []
    market_rows    = []
    firm_sum_rows  = []
    reasoning_rows = []

    true_mc_arr = gen_df["true_mc_per_mwh"].values.copy()
    firm_id_arr = gen_df["firm_id"].values

    n_retries   = 0
    n_fallbacks = 0
    n_total_rds = len(rounds_df)
    print_every = max(1, min(100, n_total_rds // 10))

    for r_num, (_, r_row) in enumerate(rounds_df.iterrows()):
        rnd     = int(r_row["round"])
        date    = str(r_row["date"])
        hour    = int(r_row["hour"])
        load_mw = float(r_row["load_mw"])

        # Build round_info; include optional chronological datetime fields when present
        round_info = {
            "round"    : rnd,
            "date"     : date,
            "hour"     : hour,
            "load_mw"  : load_mw,
            "price_cap": PRICE_CAP,
        }
        extra_dt = {col: r_row[col] for col in _DATETIME_COLS if col in r_row.index}
        round_info.update(extra_dt)

        t_mkt_price = t_price.get(rnd, np.nan)
        t_cons_pay  = t_payment.get(rnd, np.nan)

        markups = {}

        for firm_id in STRATEGIC_FIRMS:
            agent      = agents[firm_id]
            recent_pub = public_history[-HISTORY_WINDOW:]
            recent_own = firm_histories[firm_id][-HISTORY_WINDOW:]

            mem_summary = None
            if PROMPT_MODE in _CUMULATIVE_MODES:
                mem_summary = format_memory_summary(firm_histories[firm_id])

            rendered = render_prompt(
                prompt_template, firm_id, agent.portfolio, round_info,
                recent_pub, recent_own, ALLOWED_MARKUP_FACTORS, mem_summary,
            )

            markup, response, tracking = agent.decide(
                round_info, recent_pub, recent_own,
                rendered_prompt=rendered,
                memory_summary=mem_summary,
            )
            markups[firm_id] = markup

            if tracking["retry_used"]:
                n_retries += 1
            if tracking["fallback_used"]:
                n_fallbacks += 1

            is_valid, err_msg = validate_agent_response(
                response, firm_id, ALLOWED_MARKUP_FACTORS, PROMPT_MODE
            )

            rationale_json = json.dumps(response.get("decision_rationale", {}))
            actions_rows.append({
                "run_id":                  run_id,
                "prompt_mode":             PROMPT_MODE,
                "prompt_id":               TREATMENT_NAME,
                "schema_profile":          SCHEMA_PROFILE,
                "mock_agent_mode":         MOCK_AGENT_MODE if AGENT_TYPE == "mock" else "",
                "agent_type":              tracking["agent_type"],
                "model":                   tracking["model"],
                "temperature":             tracking["temperature"],
                "max_tokens":              tracking["max_tokens"],
                "seed":                    SEED,
                "prompt_file":             prompt_file,
                "prompt_sha256":           prompt_hash,
                "round":                   rnd,
                "date":                    date,
                "hour":                    hour,
                **extra_dt,
                "firm_id":                 firm_id,
                "markup_factor":           markup,
                "confidence":              response.get("confidence", ""),
                "decision_rationale_json": rationale_json,
                "rendered_prompt":         rendered,
                "raw_agent_response_json": json.dumps(response),
                "raw_agent_response_text": tracking["raw_agent_response_text"],
                "parse_error":             tracking["parse_error"],
                "retry_used":              tracking["retry_used"],
                "fallback_used":           tracking["fallback_used"],
                "valid_response":          is_valid,
            })
            reasoning_rows.append({
                "run_id":                  run_id,
                "round":                   rnd,
                "firm_id":                 firm_id,
                "prompt_mode":             PROMPT_MODE,
                "prompt_id":               TREATMENT_NAME,
                "schema_profile":          SCHEMA_PROFILE,
                "decision_rationale_json": rationale_json,
                "markup_factor":           markup,
                "confidence":              response.get("confidence", ""),
            })

            if AGENT_TYPE == "openai":
                flags  = ""
                if tracking["retry_used"]:
                    flags += " [RETRY]"
                if tracking["fallback_used"]:
                    flags += " [FALLBACK]"
                ok_str = "ok" if is_valid else "INVALID"
                print(f"    R{rnd:>2} {firm_id:<15} markup={markup:<5.2f}  {ok_str}{flags}")

        # Build offer prices
        offer_arr = true_mc_arr.copy()
        for firm_id, markup in markups.items():
            mask            = firm_id_arr == firm_id
            offer_arr[mask] = np.minimum(markup * true_mc_arr[mask], PRICE_CAP)
        bs_mask = gen_df["gen_id"].values == BACKSTOP_GEN_ID
        offer_arr[bs_mask] = PRICE_CAP

        stack, market_price, marginal_i = dispatch_with_offers(gen_df, offer_arr, load_mw)

        marginal_row = stack.iloc[marginal_i]
        bs_dispatch  = float(stack.loc[stack["gen_id"] == BACKSTOP_GEN_ID, "dispatch_mw"].sum())

        markup_map = {**markups, BACKSTOP_FIRM_ID: 1.0}
        stack["markup_factor"] = [markup_map.get(f, np.nan) for f in stack["firm_id"]]

        for _, row in stack.iterrows():
            dispatch_rows.append({
                "run_id":           run_id,
                "round":            rnd,
                "date":             date,
                "hour":             hour,
                **extra_dt,
                "gen_id":           row["gen_id"],
                "firm_id":          row["firm_id"],
                "technology_group": row["technology_group"],
                "pmax_mw":          row["pmax_mw"],
                "true_mc_per_mwh":  row["true_mc_per_mwh"],
                "offer_price":      row["offer_price"],
                "markup_factor":    row["markup_factor"],
                "dispatch_mw":      row["dispatch_mw"],
                "market_price":     row["market_price"],
                "revenue":          row["revenue"],
                "production_cost":  row["production_cost"],
                "profit":           row["profit"],
                "is_marginal":      row["is_marginal"],
                "is_backstop":      row["is_backstop"],
            })

        market_rows.append({
            "run_id":                          run_id,
            "round":                           rnd,
            "date":                            date,
            "hour":                            hour,
            **extra_dt,
            "load_mw":                         load_mw,
            "market_price":                    market_price,
            "truthful_market_price":           t_mkt_price,
            "price_increase":                  market_price - t_mkt_price,
            "total_generation_mw":             float(stack["dispatch_mw"].sum()),
            "total_consumer_payment":          market_price * load_mw,
            "truthful_total_consumer_payment": t_cons_pay,
            "consumer_payment_increase":       market_price * load_mw - t_cons_pay,
            "total_production_cost":           float(stack["production_cost"].sum()),
            "total_generator_profit":          float(stack["profit"].sum()),
            "marginal_gen_id":                 str(marginal_row["gen_id"]),
            "marginal_firm_id":                str(marginal_row["firm_id"]),
            "marginal_technology_group":       str(marginal_row["technology_group"]),
            "backstop_dispatch_mw":            bs_dispatch,
            "backstop_dispatched":             bs_dispatch > DISPATCH_TOLERANCE,
        })

        for firm_id in STRATEGIC_FIRMS + [BACKSTOP_FIRM_ID]:
            f_mask = stack["firm_id"] == firm_id
            f_disp = float(stack.loc[f_mask, "dispatch_mw"].sum())
            f_rev  = float(stack.loc[f_mask, "revenue"].sum())
            f_cost = float(stack.loc[f_mask, "production_cost"].sum())
            f_prof = float(stack.loc[f_mask, "profit"].sum())
            f_marg = bool(stack.loc[f_mask, "is_marginal"].any())
            f_mu   = markup_map.get(firm_id, np.nan)
            t_fp   = t_firm_profit.get((rnd, firm_id), np.nan)

            firm_sum_rows.append({
                "run_id":               run_id,
                "round":                rnd,
                "date":                 date,
                "hour":                 hour,
                **extra_dt,
                "firm_id":              firm_id,
                "markup_factor":        f_mu,
                "firm_dispatch_mw":     f_disp,
                "firm_revenue":         f_rev,
                "firm_production_cost": f_cost,
                "firm_profit":          f_prof,
                "truthful_firm_profit": t_fp,
                "firm_profit_gain":     f_prof - t_fp if pd.notna(t_fp) else np.nan,
                "was_marginal":         f_marg,
            })

        public_history.append({
            "round":               rnd,
            "date":                date,
            "hour":                hour,
            "load_mw":             load_mw,
            "market_price":        market_price,
            "backstop_dispatched": bs_dispatch > DISPATCH_TOLERANCE,
            "marginal_firm_id":    str(marginal_row["firm_id"]),
        })
        for firm_id in STRATEGIC_FIRMS:
            f_mask = stack["firm_id"] == firm_id
            firm_histories[firm_id].append({
                "round":           rnd,
                "markup_factor":   markups[firm_id],
                "dispatch_mw":     float(stack.loc[f_mask, "dispatch_mw"].sum()),
                "revenue":         float(stack.loc[f_mask, "revenue"].sum()),
                "production_cost": float(stack.loc[f_mask, "production_cost"].sum()),
                "profit":          float(stack.loc[f_mask, "profit"].sum()),
                "was_marginal":    bool(stack.loc[f_mask, "is_marginal"].any()),
            })

        if AGENT_TYPE == "mock":
            if (r_num + 1) % print_every == 0 or r_num + 1 == n_total_rds:
                print(f"  Round {rnd:>4} ({r_num+1:>4}/{n_total_rds}) | "
                      f"price=${market_price:>7.2f}/MWh | "
                      f"marginal={marginal_row['firm_id']}")
        else:
            print(f"  --- Round {rnd} cleared: price=${market_price:.2f}/MWh | "
                  f"marginal={marginal_row['firm_id']}")

    sim_stats = {"n_retries": n_retries, "n_fallbacks": n_fallbacks}

    return {
        "actions":   pd.DataFrame(actions_rows),
        "dispatch":  pd.DataFrame(dispatch_rows),
        "market":    pd.DataFrame(market_rows),
        "firm_sum":  pd.DataFrame(firm_sum_rows),
        "reasoning": pd.DataFrame(reasoning_rows),
        "sim_stats": sim_stats,
    }


# ============================================================
# DAILY-SCHEDULE SIMULATION LOOP
# ============================================================

def run_daily_schedule_simulation(gen_df, rounds_df, agents, prompt_template, prompt_file,
                                   prompt_hash, truthful_market_df, truthful_firm_df,
                                   run_id, load_percentiles):
    print(f"\n=== Running simulation (daily schedule): {run_id} ===")
    print(f"  Prompt mode : {PROMPT_MODE}  |  Agent type: {AGENT_TYPE}"
          + (f"  |  Mock mode: {MOCK_AGENT_MODE}" if AGENT_TYPE == "mock" else
             f"  |  Model: {MODEL}  |  Temp: {TEMPERATURE}"))
    print(f"  Seed        : {SEED}  |  Rounds: {len(rounds_df)}  |  Price cap: ${PRICE_CAP}/MWh")

    t_price   = dict(zip(truthful_market_df["round"].astype(int),
                         truthful_market_df["market_price"]))
    t_payment = dict(zip(truthful_market_df["round"].astype(int),
                         truthful_market_df["total_consumer_payment"]))
    t_firm_profit = {}
    for _, row in truthful_firm_df.iterrows():
        t_firm_profit[(int(row["round"]), str(row["firm_id"]))] = float(row["firm_profit"])

    dates = rounds_df["date"].unique()  # chronological order

    public_history  = []
    firm_histories  = {fid: [] for fid in STRATEGIC_FIRMS}
    firm_prompt_memory = {
        fid: {"plans": "", "insights": "", "journal": ""}
        for fid in STRATEGIC_FIRMS
    }

    actions_rows        = []
    dispatch_rows       = []
    market_rows         = []
    firm_sum_rows       = []
    reasoning_rows      = []
    daily_response_rows = []

    true_mc_arr = gen_df["true_mc_per_mwh"].values.copy()
    firm_id_arr = gen_df["firm_id"].values

    n_retries    = 0
    n_fallbacks  = 0
    n_total_days = len(dates)
    print_every  = max(1, min(20, n_total_days // 5))

    day_counter  = 0
    market_price = 0.0  # track last hour price for print

    for date in dates:
        day_rounds = rounds_df[rounds_df["date"] == date].sort_values("hour")

        if len(day_rounds) != 24:
            print(f"  [WARN] Date {date} has {len(day_rounds)} rounds (expected 24) — skipping")
            continue

        first_row = day_rounds.iloc[0]

        # Build day_info
        day_info = {"date": date, "price_cap": PRICE_CAP}
        for col in ["day_of_week", "month", "season"]:
            if col in first_row.index:
                day_info[col] = first_row[col]

        # 24-hour load forecast (hours 1-24, matching rounds_df['hour'])
        daily_load_forecast = [(int(r["hour"]), float(r["load_mw"]))
                               for _, r in day_rounds.iterrows()]

        # History limited to hours strictly before this date
        date_str = str(date)
        recent_pub_for_day = [h for h in public_history
                              if str(h.get("date", "")) < date_str][-HISTORY_WINDOW:]

        # daily metadata cache for hourly expansion: (date, firm_id) -> dict
        daily_meta = {}

        for firm_id in STRATEGIC_FIRMS:
            agent      = agents[firm_id]
            own_hist_full = [h for h in firm_histories[firm_id]
                             if str(h.get("date", "")) < date_str]
            recent_own = own_hist_full[-HISTORY_WINDOW:]

            mem_summary = None
            if PROMPT_MODE in _CUMULATIVE_MODES:
                mem_summary = format_memory_summary(own_hist_full)

            diag_text = None
            if PROMPT_MODE in _DIAGNOSTICS_MODES:
                if PROMPT_MODE == "daily_schedule_clean_market_power_diagnostics":
                    diag_text = compute_neutral_market_position_diagnostics(
                        firm_id, recent_own, recent_pub_for_day, HISTORY_WINDOW
                    )
                else:
                    diag_text = compute_market_position_diagnostics(
                        firm_id, recent_own, recent_pub_for_day, HISTORY_WINDOW
                    )

            plans_memory = None
            insights_memory = None
            journal_memory = None
            if SCHEMA_PROFILE in {"fish", "fish_structured"}:
                plans_memory = firm_prompt_memory[firm_id]["plans"]
                insights_memory = firm_prompt_memory[firm_id]["insights"]
            elif SCHEMA_PROFILE in {"tarj"}:
                journal_memory = firm_prompt_memory[firm_id]["journal"]

            rendered = render_daily_schedule_prompt(
                prompt_template, firm_id, agent.portfolio, day_info,
                daily_load_forecast, recent_pub_for_day, recent_own,
                ALLOWED_MARKUP_FACTORS, mem_summary, load_percentiles,
                market_position_diagnostics=diag_text,
                plans_memory=plans_memory, insights_memory=insights_memory,
                journal_memory=journal_memory,
            )

            response, tracking = agent.decide_daily(
                day_info, daily_load_forecast, recent_pub_for_day, recent_own,
                rendered_prompt=rendered, memory_summary=mem_summary,
            )

            is_valid, err_msg = validate_daily_schedule_response(
                response, firm_id, date_str, ALLOWED_MARKUP_FACTORS, PROMPT_MODE, SCHEMA_PROFILE
            )

            if tracking["retry_used"]:   n_retries   += 1
            if tracking["fallback_used"]: n_fallbacks += 1

            schedule     = response.get("hourly_markup_schedule", [])
            sched_by_hr  = {int(e["hour"]): float(e["markup_factor"]) for e in schedule}
            markups_day  = list(sched_by_hr.values())
            n_mu         = len(markups_day)

            daily_response_rows.append({
                "run_id":                   run_id,
                "date":                     date_str,
                "firm_id":                  firm_id,
                "decision_frequency":       "daily_schedule",
                "prompt_mode":              PROMPT_MODE,
                "prompt_id":                TREATMENT_NAME,
                "schema_profile":           SCHEMA_PROFILE,
                "agent_type":               tracking["agent_type"],
                "model":                    tracking["model"],
                "temperature":              tracking["temperature"],
                "raw_agent_response_text":  tracking["raw_agent_response_text"],
                "parsed_response_json":     json.dumps(response),
                "valid_response":           is_valid,
                "retry_used":               tracking["retry_used"],
                "fallback_used":            tracking["fallback_used"],
                "average_markup_for_day":   round(float(np.mean(markups_day)) if markups_day else 1.0, 4),
                "min_markup_for_day":       float(np.min(markups_day)) if markups_day else 1.0,
                "max_markup_for_day":       float(np.max(markups_day)) if markups_day else 1.0,
                "share_alpha_1":    sum(1 for m in markups_day if m == 1.0)  / max(n_mu, 1),
                "share_alpha_1_25": sum(1 for m in markups_day if m == 1.25) / max(n_mu, 1),
                "share_alpha_1_5":  sum(1 for m in markups_day if m == 1.5)  / max(n_mu, 1),
                "share_alpha_ge_2": sum(1 for m in markups_day if m >= 2.0)  / max(n_mu, 1),
                "share_alpha_10":   sum(1 for m in markups_day if m == 10.0) / max(n_mu, 1),
                "daily_call_id":    f"{date_str}_{firm_id}",
                "rendered_prompt":  rendered,
            })

            if SCHEMA_PROFILE in {"summary", "strategy_summary"}:
                rationale_json = json.dumps({"strategy_summary": response.get("strategy_summary", "")})
            elif SCHEMA_PROFILE in {"fish", "fish_structured", "tarj", "tarj_flexible"}:
                rationale_json = json.dumps(response)
            else:
                rationale_json = json.dumps(response.get("decision_rationale", {}))
            reasoning_rows.append({
                "run_id":                  run_id,
                "date":                    date_str,
                "firm_id":                 firm_id,
                "decision_frequency":      "daily_schedule",
                "prompt_mode":             PROMPT_MODE,
                "prompt_id":               TREATMENT_NAME,
                "schema_profile":          SCHEMA_PROFILE,
                "decision_rationale_json": rationale_json,
                "average_markup_for_day":  round(float(np.mean(markups_day)) if markups_day else 1.0, 4),
                "confidence":              response.get("confidence", ""),
            })

            daily_meta[(date_str, firm_id)] = {
                "sched_by_hr":    sched_by_hr,
                "valid_response": is_valid,
                "retry_used":     tracking["retry_used"],
                "fallback_used":  tracking["fallback_used"],
                "confidence":     response.get("confidence", ""),
                "rationale_json": rationale_json,
                "call_id":        f"{date_str}_{firm_id}",
            }

            if SCHEMA_PROFILE in {"fish", "fish_structured"}:
                firm_prompt_memory[firm_id]["plans"] = str(response.get("new_plans", "")).strip()
                firm_prompt_memory[firm_id]["insights"] = str(response.get("new_insights", "")).strip()
            elif SCHEMA_PROFILE in {"tarj"}:
                firm_prompt_memory[firm_id]["journal"] = str(response.get("journal_update", "")).strip()

        # Clear 24 hourly markets in chronological order
        for _, r_row in day_rounds.iterrows():
            rnd     = int(r_row["round"])
            hour    = int(r_row["hour"])
            load_mw = float(r_row["load_mw"])

            extra_dt = {col: r_row[col] for col in _DATETIME_COLS if col in r_row.index}

            markups = {}
            for firm_id in STRATEGIC_FIRMS:
                meta_f = daily_meta[(date_str, firm_id)]
                markups[firm_id] = meta_f["sched_by_hr"].get(hour, 1.0)

            t_mkt_price = t_price.get(rnd, np.nan)
            t_cons_pay  = t_payment.get(rnd, np.nan)

            offer_arr = true_mc_arr.copy()
            for firm_id, markup in markups.items():
                mask            = firm_id_arr == firm_id
                offer_arr[mask] = np.minimum(markup * true_mc_arr[mask], PRICE_CAP)
            bs_mask = gen_df["gen_id"].values == BACKSTOP_GEN_ID
            offer_arr[bs_mask] = PRICE_CAP

            stack, market_price, marginal_i = dispatch_with_offers(gen_df, offer_arr, load_mw)
            marginal_row = stack.iloc[marginal_i]
            bs_dispatch  = float(stack.loc[stack["gen_id"] == BACKSTOP_GEN_ID, "dispatch_mw"].sum())

            markup_map = {**markups, BACKSTOP_FIRM_ID: 1.0}
            stack["markup_factor"] = [markup_map.get(f, np.nan) for f in stack["firm_id"]]

            for firm_id in STRATEGIC_FIRMS:
                meta_f = daily_meta[(date_str, firm_id)]
                actions_rows.append({
                    "run_id":                  run_id,
                    "date":                    date_str,
                    "hour":                    hour,
                    "round":                   rnd,
                    **extra_dt,
                    "firm_id":                 firm_id,
                    "markup_factor":           markups[firm_id],
                    "decision_frequency":      "daily_schedule",
                    "daily_call_id":           meta_f["call_id"],
                    "valid_response":          meta_f["valid_response"],
                    "retry_used":              meta_f["retry_used"],
                    "fallback_used":           meta_f["fallback_used"],
                    "confidence":              meta_f["confidence"],
                    "decision_rationale_json": meta_f["rationale_json"],
                    "prompt_mode":             PROMPT_MODE,
                    "prompt_id":               TREATMENT_NAME,
                    "schema_profile":          SCHEMA_PROFILE,
                    "agent_type":              AGENT_TYPE,
                    "seed":                    SEED,
                    "prompt_file":             prompt_file,
                    "prompt_sha256":           prompt_hash,
                })

            for _, row in stack.iterrows():
                dispatch_rows.append({
                    "run_id":           run_id,
                    "round":            rnd,
                    "date":             date_str,
                    "hour":             hour,
                    **extra_dt,
                    "gen_id":           row["gen_id"],
                    "firm_id":          row["firm_id"],
                    "technology_group": row["technology_group"],
                    "pmax_mw":          row["pmax_mw"],
                    "true_mc_per_mwh":  row["true_mc_per_mwh"],
                    "offer_price":      row["offer_price"],
                    "markup_factor":    row["markup_factor"],
                    "dispatch_mw":      row["dispatch_mw"],
                    "market_price":     row["market_price"],
                    "revenue":          row["revenue"],
                    "production_cost":  row["production_cost"],
                    "profit":           row["profit"],
                    "is_marginal":      row["is_marginal"],
                    "is_backstop":      row["is_backstop"],
                })

            market_rows.append({
                "run_id":                          run_id,
                "round":                           rnd,
                "date":                            date_str,
                "hour":                            hour,
                **extra_dt,
                "load_mw":                         load_mw,
                "market_price":                    market_price,
                "truthful_market_price":           t_mkt_price,
                "price_increase":                  market_price - t_mkt_price,
                "total_generation_mw":             float(stack["dispatch_mw"].sum()),
                "total_consumer_payment":          market_price * load_mw,
                "truthful_total_consumer_payment": t_cons_pay,
                "consumer_payment_increase":       market_price * load_mw - t_cons_pay,
                "total_production_cost":           float(stack["production_cost"].sum()),
                "total_generator_profit":          float(stack["profit"].sum()),
                "marginal_gen_id":                 str(marginal_row["gen_id"]),
                "marginal_firm_id":                str(marginal_row["firm_id"]),
                "marginal_technology_group":       str(marginal_row["technology_group"]),
                "backstop_dispatch_mw":            bs_dispatch,
                "backstop_dispatched":             bs_dispatch > DISPATCH_TOLERANCE,
            })

            for firm_id in STRATEGIC_FIRMS + [BACKSTOP_FIRM_ID]:
                f_mask = stack["firm_id"] == firm_id
                f_disp = float(stack.loc[f_mask, "dispatch_mw"].sum())
                f_rev  = float(stack.loc[f_mask, "revenue"].sum())
                f_cost = float(stack.loc[f_mask, "production_cost"].sum())
                f_prof = float(stack.loc[f_mask, "profit"].sum())
                f_marg = bool(stack.loc[f_mask, "is_marginal"].any())
                f_mu   = markup_map.get(firm_id, np.nan)
                t_fp   = t_firm_profit.get((rnd, firm_id), np.nan)

                firm_sum_rows.append({
                    "run_id":               run_id,
                    "round":                rnd,
                    "date":                 date_str,
                    "hour":                 hour,
                    **extra_dt,
                    "firm_id":              firm_id,
                    "markup_factor":        f_mu,
                    "firm_dispatch_mw":     f_disp,
                    "firm_revenue":         f_rev,
                    "firm_production_cost": f_cost,
                    "firm_profit":          f_prof,
                    "truthful_firm_profit": t_fp,
                    "firm_profit_gain":     f_prof - t_fp if pd.notna(t_fp) else np.nan,
                    "was_marginal":         f_marg,
                })

            public_history.append({
                "round":               rnd,
                "date":                date_str,
                "hour":                hour,
                "load_mw":             load_mw,
                "market_price":        market_price,
                "backstop_dispatched": bs_dispatch > DISPATCH_TOLERANCE,
                "marginal_firm_id":    str(marginal_row["firm_id"]),
            })
            for firm_id in STRATEGIC_FIRMS:
                f_mask = stack["firm_id"] == firm_id
                firm_histories[firm_id].append({
                    "round":           rnd,
                    "date":            date_str,
                    "hour":            hour,
                    "markup_factor":   markups[firm_id],
                    "dispatch_mw":     float(stack.loc[f_mask, "dispatch_mw"].sum()),
                    "revenue":         float(stack.loc[f_mask, "revenue"].sum()),
                    "production_cost": float(stack.loc[f_mask, "production_cost"].sum()),
                    "profit":          float(stack.loc[f_mask, "profit"].sum()),
                    "was_marginal":    bool(stack.loc[f_mask, "is_marginal"].any()),
                })

        day_counter += 1
        if day_counter % print_every == 0 or day_counter == n_total_days:
            print(f"  Day {day_counter:>3}/{n_total_days} ({date_str}) | "
                  f"last-hr price=${market_price:>7.2f}/MWh | "
                  f"marginal={marginal_row['firm_id']}")

    sim_stats = {"n_retries": n_retries, "n_fallbacks": n_fallbacks}

    return {
        "actions":          pd.DataFrame(actions_rows),
        "dispatch":         pd.DataFrame(dispatch_rows),
        "market":           pd.DataFrame(market_rows),
        "firm_sum":         pd.DataFrame(firm_sum_rows),
        "reasoning":        pd.DataFrame(reasoning_rows),
        "daily_responses":  pd.DataFrame(daily_response_rows),
        "sim_stats":        sim_stats,
    }


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(run_id, results, run_metadata, output_dir, validation_passed=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Saving outputs to {output_dir} ===")

    file_map = {
        f"llm_firm_actions_{run_id}.csv":    results["actions"],
        f"llm_dispatch_{run_id}.csv":        results["dispatch"],
        f"llm_market_summary_{run_id}.csv":  results["market"],
        f"llm_firm_summary_{run_id}.csv":    results["firm_sum"],
        f"llm_reasoning_log_{run_id}.csv":   results["reasoning"],
    }
    for fname, df in file_map.items():
        df.to_csv(output_dir / fname, index=False)
        print(f"  {fname}  ({len(df):,} rows)")

    if "daily_responses" in results and not results["daily_responses"].empty:
        dr_fname = f"llm_daily_agent_responses_{run_id}.csv"
        results["daily_responses"].to_csv(output_dir / dr_fname, index=False)
        print(f"  {dr_fname}  ({len(results['daily_responses']):,} rows)")

    if validation_passed is not None:
        run_metadata = dict(run_metadata)
        run_metadata["validation_passed"] = validation_passed

    meta_path = output_dir / f"llm_run_metadata_{run_id}.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(run_metadata, fh, indent=2)
    print(f"  llm_run_metadata_{run_id}.json")


# ============================================================
# PLOTS
# ============================================================

def make_plots(run_id, results, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Creating plots in {plots_dir} ===")

    market_df  = results["market"]
    firm_df    = results["firm_sum"]
    actions_df = results["actions"]
    rounds     = market_df["round"].values

    # 1. Market price vs truthful ED
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rounds, market_df["market_price"],
            lw=1.5, color="steelblue", label="Agent price")
    ax.plot(rounds, market_df["truthful_market_price"],
            lw=1.2, color="gray", ls="--", label="Truthful ED price")
    ax.fill_between(rounds,
                    market_df["truthful_market_price"],
                    market_df["market_price"],
                    alpha=0.15, color="steelblue", label="Price increase")
    ax.set_xlabel("Round")
    ax.set_ylabel("Market Clearing Price ($/MWh)")
    ax.set_title(f"Market Price vs Truthful ED — {run_id}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, plots_dir, f"price_vs_truthful_{run_id}.png")

    # 2. Markup by firm over rounds
    fig, ax = plt.subplots(figsize=(12, 4))
    for firm_id in STRATEGIC_FIRMS:
        fd = actions_df[actions_df["firm_id"] == firm_id].sort_values("round")
        ax.plot(fd["round"], fd["markup_factor"],
                label=firm_id.replace("Firm_", ""),
                color=FIRM_COLORS.get(firm_id, "#888888"),
                linewidth=1.5, marker="o", markersize=3)
    ax.axhline(1.0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("Round")
    ax.set_ylabel("Markup Factor (alpha)")
    ax.set_title(f"Markup Factors by Firm — {run_id}")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, plots_dir, f"markup_by_firm_{run_id}.png")

    # 3. Cumulative profit gain vs truthful ED
    fig, ax = plt.subplots(figsize=(12, 5))
    for firm_id in STRATEGIC_FIRMS:
        fd       = firm_df[firm_df["firm_id"] == firm_id].sort_values("round")
        cum_gain = fd["firm_profit_gain"].fillna(0).cumsum()
        ax.plot(fd["round"], cum_gain / 1e3,
                label=firm_id.replace("Firm_", ""),
                color=FIRM_COLORS.get(firm_id, "#888888"),
                linewidth=1.5)
    ax.axhline(0, color="black", ls=":", lw=0.8)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Profit Gain vs Truthful ED ($1,000s)")
    ax.set_title(f"Cumulative Profit Gain vs Truthful ED — {run_id}")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, plots_dir, f"firm_profit_gain_{run_id}.png")

    # 4. Consumer payment increase
    pay_inc_k = market_df["consumer_payment_increase"] / 1e3
    fig, ax   = plt.subplots(figsize=(12, 4))
    ax.bar(rounds, pay_inc_k, color="tomato", alpha=0.7, label="Per-round increase ($1000s)")
    ax.plot(rounds, pay_inc_k.cumsum(), color="darkred", lw=1.5, label="Cumulative ($1000s)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Consumer Payment Increase vs Truthful ($1,000s)")
    ax.set_title(f"Consumer Payment Increase vs Truthful ED — {run_id}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, plots_dir, f"consumer_payment_increase_{run_id}.png")

    # 5. Marginal firm frequency
    marg_counts = market_df["marginal_firm_id"].value_counts().sort_index()
    colors      = [FIRM_COLORS.get(f, "#cccccc") for f in marg_counts.index]
    fig, ax     = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(marg_counts)), marg_counts.values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(marg_counts)))
    ax.set_xticklabels([f.replace("Firm_", "") for f in marg_counts.index])
    ax.set_ylabel("Rounds as Marginal (count)")
    ax.set_title(f"Marginal Firm Frequency — {run_id}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, plots_dir, f"marginal_firm_frequency_{run_id}.png")


def _save_plot(fig, plots_dir, fname):
    path = plots_dir / fname
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {fname}")


# ============================================================
# VALIDATION
# ============================================================

def validate_results(results, gen_df, truthful_market_df, truthful_firm_df):
    print("\n=== Validation checks ===")
    dispatch_df = results["dispatch"]
    market_df   = results["market"]
    actions_df  = results["actions"]
    passed      = True

    # 1. Balance
    gen_by_round = dispatch_df.groupby("round")["dispatch_mw"].sum().reset_index()
    merged       = gen_by_round.merge(market_df[["round", "load_mw"]], on="round")
    max_bal_err  = (merged["dispatch_mw"] - merged["load_mw"]).abs().max()
    ok1 = max_bal_err < DISPATCH_TOLERANCE * 100
    print(f"  [{'OK' if ok1 else 'FAIL'}] Balance:    max |generation - load| = {max_bal_err:.2e} MW")
    passed = passed and ok1

    # 2. Capacity
    cap_violation = (dispatch_df["dispatch_mw"] - dispatch_df["pmax_mw"]).clip(lower=0).max()
    ok2 = cap_violation < DISPATCH_TOLERANCE
    print(f"  [{'OK' if ok2 else 'FAIL'}] Capacity:   max overrun = {cap_violation:.2e} MW")
    passed = passed and ok2

    # 3. Price cap
    max_offer = dispatch_df["offer_price"].max()
    ok3 = max_offer <= PRICE_CAP + 1e-6
    print(f"  [{'OK' if ok3 else 'FAIL'}] Price cap:  max offer = ${max_offer:.4f}/MWh (cap=${PRICE_CAP})")
    passed = passed and ok3

    # 4. Backstop at price cap
    bs_offers = dispatch_df.loc[dispatch_df["gen_id"] == BACKSTOP_GEN_ID, "offer_price"]
    ok4 = (bs_offers - PRICE_CAP).abs().max() < 1e-6 if len(bs_offers) > 0 else True
    print(f"  [{'OK' if ok4 else 'FAIL'}] Backstop:   offer prices = "
          f"{bs_offers.unique().tolist()} (expected [{PRICE_CAP}])")
    passed = passed and ok4

    # 5. Backstop not strategic
    ok5 = BACKSTOP_FIRM_ID not in actions_df["firm_id"].unique()
    print(f"  [{'OK' if ok5 else 'FAIL'}] Non-strategic backstop: "
          f"Firm_Backstop in actions = {not ok5}")
    passed = passed and ok5

    # 6. Valid markups
    bad_markup = actions_df[~actions_df["markup_factor"].isin(ALLOWED_MARKUP_FACTORS)]
    ok6 = len(bad_markup) == 0
    print(f"  [{'OK' if ok6 else 'FAIL'}] Markup validity: {len(bad_markup)} invalid markup row(s)")
    passed = passed and ok6

    # 7. Response validity
    n_invalid = int((~actions_df["valid_response"]).sum())
    ok7 = n_invalid == 0
    print(f"  [{'OK' if ok7 else 'FAIL'}] Response validity: {n_invalid} invalid response(s) "
          f"(out of {len(actions_df)})")
    passed = passed and ok7

    # 8. LLM stats
    n_retries   = int(actions_df["retry_used"].sum())   if "retry_used"   in actions_df.columns else 0
    n_fallbacks = int(actions_df["fallback_used"].sum()) if "fallback_used" in actions_df.columns else 0
    n_total     = len(actions_df)
    valid_share = 1.0 - n_invalid / n_total if n_total > 0 else 1.0
    print(f"  [INFO] LLM response stats: "
          f"valid={n_total - n_invalid}/{n_total} ({valid_share:.1%}), "
          f"retries={n_retries}, fallbacks={n_fallbacks}")

    # 9. Truthful-mode consistency check (applies to both hourly and daily_schedule)
    if MOCK_AGENT_MODE == "truthful" and AGENT_TYPE == "mock":
        t_prices   = dict(zip(truthful_market_df["round"].astype(int),
                              truthful_market_df["market_price"]))
        t_payments = dict(zip(truthful_market_df["round"].astype(int),
                              truthful_market_df["total_consumer_payment"]))
        price_diffs = market_df.apply(
            lambda r: abs(r["market_price"] - t_prices.get(int(r["round"]), r["market_price"])),
            axis=1,
        )
        payment_diffs = market_df.apply(
            lambda r: abs(r["total_consumer_payment"]
                          - t_payments.get(int(r["round"]), r["total_consumer_payment"])),
            axis=1,
        )
        firm_profit_diffs = results["firm_sum"].apply(
            lambda r: abs(r["firm_profit_gain"]) if pd.notna(r["firm_profit_gain"]) else 0.0,
            axis=1,
        )
        print(f"  [INFO] Truthful check:")
        print(f"    max_abs_price_diff            = ${price_diffs.max():.6f}/MWh")
        print(f"    max_abs_consumer_payment_diff = ${payment_diffs.max():.4f}")
        print(f"    max_abs_firm_profit_diff      = ${firm_profit_diffs.max():.6f}")

    print(f"\n  Result: {'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}")
    return passed


# ============================================================
# MAIN
# ============================================================

def main():
    global PROMPT_MODE, PROMPT_FILE_OVERRIDE, PROMPT_ID, SCHEMA_PROFILE, TREATMENT_NAME
    global MOCK_AGENT_MODE, AGENT_TYPE, MODEL, TEMPERATURE
    global MAX_TOKENS, API_TIMEOUT, SLEEP_SECONDS
    global SEED, HISTORY_WINDOW, N_ROUNDS, STRESS_CASE, DECISION_FREQUENCY
    global ALLOWED_MARKUP_FACTORS
    global USE_RESPONSES_API, REASONING_EFFORT

    parser = argparse.ArgumentParser(
        description="LLM-agent bidding simulation for the thermal-only 5-firm electricity market.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--case-dir",
        default="single_node_test_case",
        help="Case directory (relative to project root or absolute). "
             "Default: single_node_test_case",
    )
    parser.add_argument(
        "--agent-type", dest="agent_type",
        choices=["mock", "openai"], default=AGENT_TYPE,
    )
    parser.add_argument(
        "--mock-agent-mode", dest="mock_agent_mode",
        choices=["truthful", "random", "heuristic"], default=MOCK_AGENT_MODE,
    )
    parser.add_argument(
        "--prompt-file", dest="prompt_file", default=None,
        help="Path to a prompt template file. If set, this overrides --prompt-mode."
    )
    parser.add_argument(
        "--prompt-id", "--treatment-name", dest="prompt_id", default=None,
        help="Run label used in output filenames and metadata. Defaults to the prompt filename stem."
    )
    parser.add_argument(
        "--run-tag", dest="run_tag", default=None,
        help="Optional experiment tag prefixed to run_id and output filenames."
    )
    parser.add_argument(
        "--prompt-mode", dest="prompt_mode",
        choices=["myopic_profit", "cumulative_profit",
                 "hourly_myopic_profit", "hourly_cumulative_profit",
                 "daily_schedule_myopic_profit", "daily_schedule_cumulative_profit",
                 "daily_schedule_myopic_diagnostics",
                 "daily_schedule_profit_max_no_caution",
                 "daily_schedule_br_reasoning_scaffold",
                 "daily_schedule_clean_market_power_diagnostics",
                 "daily_schedule_competitive_compliance"],
        default=None,
    )
    parser.add_argument(
        "--schema-profile", dest="schema_profile",
        choices=["fish", "tarj", "decision_rationale", "summary", "flexible"],
        default=None,
        help="Validation/schema profile for daily-schedule outputs."
    )
    parser.add_argument(
        "--n-days", dest="n_days", type=int, default=None,
        help="Number of operating days to simulate (sets --n-rounds = n_days * 24). "
             "If both --n-days and --n-rounds are given, --n-rounds takes precedence.",
    )
    parser.add_argument(
        "--decision-frequency", dest="decision_frequency",
        choices=["hourly", "daily_schedule"], default=None,
        help="hourly: one call per firm-hour. daily_schedule: one call per firm-day. "
             "Default: None (backward-compat → writes to llm_bidding/ root; "
             "set explicitly to write to subfolders).",
    )
    parser.add_argument("--model",       default=MODEL)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max-tokens",  dest="max_tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--api-timeout", dest="api_timeout", type=int, default=API_TIMEOUT)
    parser.add_argument("--sleep-seconds", dest="sleep_seconds", type=float, default=SLEEP_SECONDS)
    parser.add_argument("--use-responses-api", dest="use_responses_api",
                        action="store_true", default=False,
                        help="Use the Responses API instead of Chat Completions "
                             "(daily-schedule mode only; intended for gpt-5.5 retest)")
    parser.add_argument("--reasoning-effort", dest="reasoning_effort",
                        default="medium", choices=["low", "medium", "high"],
                        help="Reasoning effort for Responses API (default: medium)")
    parser.add_argument("--save-llm-raw-text", dest="save_llm_raw_text",
                        action="store_true", default=True)
    parser.add_argument("--seed",           type=int, default=SEED)
    parser.add_argument("--stress-case",    dest="stress_case", default=STRESS_CASE)
    parser.add_argument("--history-window", dest="history_window", type=int, default=None,
                        help="History window size. Default: 24 for summer hourly case, 5 otherwise.")
    parser.add_argument("--n-rounds",       dest="n_rounds", type=int, default=N_ROUNDS)
    parser.add_argument(
        "--allowed-markup-grid",
        dest="allowed_markup_grid",
        default=None,
        help="Comma-separated markup multiplier grid, e.g. "
             "'1.0,1.1,1.25,1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.5,10.0'. "
             "Overrides the default 12-point grid. A _gridN suffix is appended to run_id.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir_override",
        default=None,
        help="Override output directory (default: workspace runs directory). "
             "All CSVs, JSONs, and plots are written here.",
    )

    args = parser.parse_args()

    # --- Parse --allowed-markup-grid ----------------------------------------
    if args.allowed_markup_grid is not None:
        try:
            _parsed = sorted(float(x.strip()) for x in args.allowed_markup_grid.split(","))
        except ValueError as exc:
            parser.error(f"--allowed-markup-grid parse error: {exc}")
        if len(_parsed) < 2:
            parser.error("--allowed-markup-grid must contain at least 2 values")
        if any(v < 1.0 for v in _parsed):
            parser.error("--allowed-markup-grid: all values must be >= 1.0 (markup, not price)")
        ALLOWED_MARKUP_FACTORS = _parsed

    AGENT_TYPE        = args.agent_type
    MOCK_AGENT_MODE   = args.mock_agent_mode
    PROMPT_FILE_OVERRIDE = args.prompt_file
    PROMPT_ID         = args.prompt_id
    RUN_TAG           = args.run_tag
    if args.prompt_file is not None:
        if PROMPT_ID is None:
            PROMPT_ID = Path(args.prompt_file).stem
    PROMPT_MODE       = args.prompt_mode or PROMPT_ID or PROMPT_MODE
    TREATMENT_NAME    = PROMPT_ID or PROMPT_MODE
    SCHEMA_PROFILE    = args.schema_profile or SCHEMA_PROFILE
    MODEL             = args.model
    TEMPERATURE       = args.temperature
    MAX_TOKENS        = args.max_tokens
    API_TIMEOUT       = args.api_timeout
    SLEEP_SECONDS     = args.sleep_seconds
    SEED              = args.seed
    STRESS_CASE       = args.stress_case
    USE_RESPONSES_API = args.use_responses_api
    REASONING_EFFORT  = args.reasoning_effort
    # --n-days sets N_ROUNDS = n_days * 24; explicit --n-rounds overrides
    if args.n_rounds != N_ROUNDS:  # user passed --n-rounds explicitly
        N_ROUNDS = args.n_rounds
    elif args.n_days is not None:
        N_ROUNDS = args.n_days * 24
    else:
        N_ROUNDS = args.n_rounds

    # Resolve case directory and derived paths
    case_dir = Path(args.case_dir)
    if not case_dir.is_absolute():
        case_dir = PROJECT_ROOT / case_dir

    # Decision frequency: infer from prompt_mode if not explicitly given
    if args.decision_frequency is not None:
        DECISION_FREQUENCY = args.decision_frequency
    elif args.prompt_file is not None or PROMPT_MODE in _DAILY_SCHEDULE_MODES:
        DECISION_FREQUENCY = "daily_schedule"
    else:
        DECISION_FREQUENCY = "hourly"

    # Default history window based on case and decision frequency
    if args.history_window is not None:
        HISTORY_WINDOW = args.history_window
    elif case_dir.name == "single_node_summer_hourly_case":
        HISTORY_WINDOW = 72 if DECISION_FREQUENCY == "daily_schedule" else 24
    else:
        HISTORY_WINDOW = 5

    case_tag = _CASE_TAG_MAP.get(case_dir.name, case_dir.name)

    gen_file             = case_dir / "generators_thermal_5firm.csv"
    round_file           = case_dir / f"market_rounds_{case_tag}_{STRESS_CASE}.csv"
    truthful_market_file = case_dir / "truthful_ed" / f"truthful_market_summary_{case_tag}_{STRESS_CASE}.csv"
    truthful_firm_file   = case_dir / "truthful_ed" / f"truthful_firm_summary_{case_tag}_{STRESS_CASE}.csv"

    # Build run ID; prefix with case context when not the default test case
    is_default_case = (case_dir.name == "single_node_test_case")
    case_prefix     = "" if is_default_case else f"{case_tag}_{STRESS_CASE}_"
    tag_prefix      = "" if not RUN_TAG else f"{sanitize_for_filename(RUN_TAG)}_"

    if AGENT_TYPE == "mock":
        run_label = sanitize_for_filename(TREATMENT_NAME or PROMPT_MODE)
        run_id = f"{tag_prefix}{case_prefix}{run_label}_{MOCK_AGENT_MODE}_seed{SEED}"
    else:
        temp_str = sanitize_for_filename(TEMPERATURE)
        run_label = sanitize_for_filename(TREATMENT_NAME or PROMPT_MODE)
        run_id   = f"{tag_prefix}{case_prefix}{run_label}_openai_{MODEL}_temp{temp_str}_seed{SEED}"

    if DECISION_FREQUENCY == "daily_schedule" and "daily_schedule" not in run_id:
        run_id = f"{tag_prefix}{case_prefix}daily_schedule_{run_id[len(tag_prefix) + len(case_prefix):]}"

    # Append grid tag when the markup grid differs from the 12-point default
    if sorted(ALLOWED_MARKUP_FACTORS) != sorted(_DEFAULT_MARKUP_GRID):
        run_id = f"{run_id}_grid{len(ALLOWED_MARKUP_FACTORS)}"

    # Route outputs to the dedicated workspace by default
    output_dir = RUNS_ROOT / run_id
    plots_dir = output_dir / "plots"

    # Apply --output-dir override AFTER the default output_dir is computed
    if args.output_dir_override is not None:
        output_dir = Path(args.output_dir_override)
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / "plots"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if AGENT_TYPE == "openai":
        if not OPENAI_AVAILABLE:
            print("[ERROR] openai package not installed. Run:")
            print("    pip install openai")
            sys.exit(1)
        _key_new = os.environ.get("OPENAI_API_KEY_NEW", "")
        _key_old = os.environ.get("OPENAI_API_KEY", "")
        if _key_new:
            api_key = _key_new
            _key_src = "OPENAI_API_KEY_NEW"
        elif _key_old:
            api_key = _key_old
            _key_src = "OPENAI_API_KEY"
        else:
            print("[ERROR] Neither OPENAI_API_KEY_NEW nor OPENAI_API_KEY is set.")
            sys.exit(1)
        print(f"  [auth] API key loaded from {_key_src}")
        if _key_src == "OPENAI_API_KEY" and _key_new == "":
            print("  [WARN] OPENAI_API_KEY_NEW is not set — using personal OPENAI_API_KEY. "
                  "Set OPENAI_API_KEY_NEW to use project credits instead.")
        # Ensure OpenAI SDK picks up the resolved key regardless of which env var was set
        os.environ["OPENAI_API_KEY"] = api_key

    print("=" * 64)
    print(f"LLM Bidding Simulation — {run_id}")
    print(f"  Case dir   : {case_dir}")
    print(f"  Case tag   : {case_tag}")
    print(f"  Stress     : {STRESS_CASE}")
    print(f"  Output     : {output_dir}")
    print(f"  Markup grid: {ALLOWED_MARKUP_FACTORS}")
    print("=" * 64)

    gen_df, rounds_df, truthful_market_df, truthful_firm_df = load_inputs(
        gen_file, round_file, truthful_market_file, truthful_firm_file,
        N_ROUNDS, STRESS_CASE,
    )

    loads = rounds_df["load_mw"]
    load_percentiles = {
        "p75": float(loads.quantile(0.75)),
        "p90": float(loads.quantile(0.90)),
    }
    print(f"\n  Load percentiles: p75={load_percentiles['p75']:,.1f} MW, "
          f"p90={load_percentiles['p90']:,.1f} MW")

    prompt_template, prompt_file = load_prompt_template(PROMPT_MODE, PROMPT_FILE_OVERRIDE)
    prompt_hash = compute_prompt_hash(prompt_template)
    print(f"\n  Prompt file   : {prompt_file}")
    print(f"  Prompt label  : {TREATMENT_NAME}")
    print(f"  Schema profile: {SCHEMA_PROFILE}")
    print(f"  Prompt SHA256 : {prompt_hash[:24]}...")

    agents = {}
    for i, firm_id in enumerate(STRATEGIC_FIRMS):
        portfolio = gen_df[
            (gen_df["firm_id"] == firm_id) &
            (gen_df["gen_id"]  != BACKSTOP_GEN_ID)
        ][["gen_id", "technology_group", "pmax_mw", "true_mc_per_mwh"]].copy()

        if AGENT_TYPE == "mock":
            agents[firm_id] = MockFirmAgent(
                firm_id=firm_id,
                portfolio_df=portfolio,
                allowed_markup_factors=ALLOWED_MARKUP_FACTORS,
                mode=MOCK_AGENT_MODE,
                load_percentiles=load_percentiles,
                seed=SEED * 100 + i,
            )
        else:
            agents[firm_id] = OpenAIFirmAgent(
                firm_id=firm_id,
                portfolio_df=portfolio,
                allowed_markup_factors=ALLOWED_MARKUP_FACTORS,
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                api_timeout=API_TIMEOUT,
                sleep_seconds=SLEEP_SECONDS,
                seed=SEED * 100 + i,
            )

    print(f"\n  Agent type : {AGENT_TYPE}"
          + (f"  ({MOCK_AGENT_MODE})" if AGENT_TYPE == "mock" else f"  ({MODEL})"))
    print(f"  Agents     : {list(agents.keys())}")
    if AGENT_TYPE == "openai":
        n_calls = len(STRATEGIC_FIRMS) * N_ROUNDS
        print(f"  OpenAI calls planned : {n_calls}  "
              f"({len(STRATEGIC_FIRMS)} firms x {N_ROUNDS} rounds)")
        print(f"  Est. min API cost    : ~${n_calls * 0.0003:.2f}  (rough API estimate)")

    if DECISION_FREQUENCY == "daily_schedule":
        results = run_daily_schedule_simulation(
            gen_df, rounds_df, agents,
            prompt_template, prompt_file, prompt_hash,
            truthful_market_df, truthful_firm_df,
            run_id, load_percentiles,
        )
    else:
        results = run_simulation(
            gen_df, rounds_df, agents,
            prompt_template, prompt_file, prompt_hash,
            truthful_market_df, truthful_firm_df,
            run_id,
        )
    sim_stats = results.pop("sim_stats")

    def _rel(p):
        try:
            return Path(p).relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(p)

    market_type = (
        "single_node_hourly_energy_market"
        if case_dir.name == "single_node_summer_hourly_case"
        else "single_node_thermal_market"
    )

    run_metadata = {
        "run_id":                  run_id,
        "timestamp":               timestamp,
        "case_dir":                str(case_dir),
        "case_tag":                case_tag,
        "stress_case":             STRESS_CASE,
        "market_type":             market_type,
        "decision_frequency":      DECISION_FREQUENCY,
        "prompt_mode":             PROMPT_MODE,
        "prompt_id":               TREATMENT_NAME,
        "run_tag":                 RUN_TAG,
        "prompt_file_name":        Path(prompt_file).name,
        "prompt_file_override":     PROMPT_FILE_OVERRIDE,
        "schema_profile":          SCHEMA_PROFILE,
        "mock_agent_mode":         MOCK_AGENT_MODE if AGENT_TYPE == "mock" else None,
        "agent_type":              AGENT_TYPE,
        "model":                   MODEL if AGENT_TYPE == "openai" else None,
        "temperature":             TEMPERATURE if AGENT_TYPE == "openai" else None,
        "max_tokens":              MAX_TOKENS if AGENT_TYPE == "openai" else None,
        "api_timeout":             API_TIMEOUT if AGENT_TYPE == "openai" else None,
        "openai_sdk_used":         AGENT_TYPE == "openai",
        "seed":                    SEED,
        "history_window":          HISTORY_WINDOW,
        "price_cap":               PRICE_CAP,
        "allowed_markup_factors":  ALLOWED_MARKUP_FACTORS,
        "prompt_file":             prompt_file,
        "prompt_sha256":           prompt_hash,
        "information_treatment":   TREATMENT_NAME or _INFORMATION_TREATMENTS.get(PROMPT_MODE, "standard"),
        "input_files": {
            "generators":      _rel(gen_file),
            "market_rounds":   _rel(round_file),
            "truthful_market": _rel(truthful_market_file),
            "truthful_firm":   _rel(truthful_firm_file),
        },
        "number_of_rounds": len(rounds_df),
        "strategic_firms":  STRATEGIC_FIRMS,
        "backstop_id":      BACKSTOP_GEN_ID,
        "note": (
            "Real OpenAI GPT agent run." if AGENT_TYPE == "openai" else
            "Mock-agent simulation — no real LLM API calls were made."
        ),
    }

    make_plots(run_id, results, plots_dir)
    all_ok = validate_results(results, gen_df, truthful_market_df, truthful_firm_df)
    save_outputs(run_id, results, run_metadata, output_dir, validation_passed=all_ok)

    market_df  = results["market"]
    firm_df    = results["firm_sum"]
    actions_df = results["actions"]

    n_total     = len(actions_df)
    n_invalid   = int((~actions_df["valid_response"]).sum())
    n_retries   = int(actions_df["retry_used"].sum())
    n_fallbacks = int(actions_df["fallback_used"].sum())
    valid_share = 1.0 - n_invalid / n_total if n_total > 0 else 1.0

    print("\n" + "=" * 64)
    print(f"SUMMARY — {run_id}")
    print("=" * 64)
    print(f"\n  Run ID        : {run_id}")
    print(f"  Agent type    : {AGENT_TYPE}"
          + (f"  ({MOCK_AGENT_MODE})" if AGENT_TYPE == "mock" else f"  ({MODEL}, temp={TEMPERATURE})"))
    print(f"  Prompt file   : {prompt_file}  (sha256: {prompt_hash[:16]}...)")
    print(f"  Prompt mode   : {PROMPT_MODE}")
    print(f"  Decision freq : {DECISION_FREQUENCY}")
    print(f"  Rounds        : {len(market_df)}")
    if DECISION_FREQUENCY == "daily_schedule" and "daily_responses" in results:
        dr_df   = results["daily_responses"]
        n_days  = dr_df["date"].nunique()   if "date"    in dr_df.columns else 0
        n_calls = len(dr_df)
        print(f"  Daily calls   : {n_calls}  ({n_days} days x {len(STRATEGIC_FIRMS)} firms)")

    if AGENT_TYPE == "openai":
        print(f"\n  OpenAI calls  : {n_total}")
        print(f"  Valid resp.   : {n_total - n_invalid}/{n_total}  ({valid_share:.1%})")
        print(f"  Retries       : {n_retries}")
        print(f"  Fallbacks     : {n_fallbacks}")

    print(f"\n  Avg price increase            : ${market_df['price_increase'].mean():.2f}/MWh")
    print(f"  Total consumer payment incr.  : ${market_df['consumer_payment_increase'].sum():,.0f}")

    print("\n  Total firm profit gain vs truthful ED:")
    for firm_id in STRATEGIC_FIRMS:
        fd   = firm_df[firm_df["firm_id"] == firm_id]
        gain = fd["firm_profit_gain"].sum()
        print(f"    {firm_id:<17}  ${gain:>12,.0f}")

    print("\n  Marginal firm frequency:")
    marg_counts = market_df["marginal_firm_id"].value_counts()
    for fid, cnt in marg_counts.items():
        pct = 100 * cnt / len(market_df)
        print(f"    {fid:<17}  {cnt:>3} / {len(market_df)} rounds  ({pct:.0f}%)")

    print(f"\n  Validation checks : {'PASSED' if all_ok else 'FAILED'}")
    print(f"  Output folder     : {output_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
