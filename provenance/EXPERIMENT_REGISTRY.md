# Paper experiment registry

This repository contains only the six prompt treatments reported in the paper.
All runs use GPT-4.1, temperature 0.2, the 12-point markup grid, and seeds 1-5.

| Code | Treatment ID | Archived prompt | Run tag | Seeds |
| --- | --- | --- | --- | --- |
| T1 | `tarj_compliance` | `T1_compliance.txt` | `revised_v2` | 1-5 |
| T2 | `tarj_balanced_profit_compliance` | `T2_balanced.txt` | `sbert_direction_v1` | 1-5 |
| T3 | `tarj_guided_profit` | `T3_guided_profit.txt` | `sbert_direction_v2` | 1-5 |
| T4 | `tarj_default` | `T4_baseline_profit.txt` | `revised_v2` | 1-5 |
| T5 | `tarj_profit_max` | `T5_profit_max.txt` | `revised_v2` | 1-5 |
| T6 | `tarj_aggressive_profit_max` | `T6_high_profit_max.txt` | `sbert_direction_v1` | 1-5 |

The 30 source run directories are in `results/raw-runs/`. The cached analysis
tables used for the final figures are in `results/analysis/six-treatment/`.

The four post-paper exploratory prompt treatments are deliberately excluded.
