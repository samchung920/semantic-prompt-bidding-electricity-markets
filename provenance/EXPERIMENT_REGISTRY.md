# Paper experiment registry

This repository contains only the six prompt treatments reported in the paper.
All runs use GPT-4.1, temperature 0.2, the 12-point markup grid, and seeds 1-5.

| Code | Treatment ID | Label | Run tag | Seeds |
| --- | --- | --- | --- | --- |
| T1 | `tarj_compliance` | Compliance | `revised_v2` | 1-5 |
| T2 | `tarj_balanced_profit_compliance` | Balanced | `sbert_direction_v1` | 1-5 |
| T3 | `tarj_guided_profit` | Guided Profit | `sbert_direction_v2` | 1-5 |
| T4 | `tarj_default` | Baseline Profit | `revised_v2` | 1-5 |
| T5 | `tarj_profit_max` | Profit-Max | `revised_v2` | 1-5 |
| T6 | `tarj_aggressive_profit_max` | High Profit-Max | `sbert_direction_v1` | 1-5 |

The 30 source run directories are in `results/raw-runs/`. The cached analysis
tables used for the final figures are in `results/analysis/six-treatment/`.

The four post-paper exploratory prompt treatments are deliberately excluded.
