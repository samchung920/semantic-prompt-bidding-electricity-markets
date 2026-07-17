# V3 Results Interpretation

This v3 package uses cached six-treatment outputs only. No simulations were rerun.

## Main descriptive results

- Prompt orientation is strongly associated with average price impact across treatments: Spearman rho = 0.943, Pearson r = 0.926.
- Prompt orientation is also strongly associated with average markup: Spearman rho = 0.943, Pearson r = 0.900.
- Prompt orientation tracks mean TARJ reasoning orientation closely: Spearman rho = 0.943, Pearson r = 0.983.
- At the firm-day level, TARJ reasoning orientation is positively associated with average markup: Spearman rho = 0.446, Pearson r = 0.364.

## Blocked comparisons

- With day and seed fixed effects, the prompt-orientation coefficient for daily average price increase is 181.733 with robust SE 8.311 (N = 210).
- With firm, day, and seed fixed effects, the prompt-orientation coefficient for daily firm average markup is 4.654 with robust SE 0.162 (N = 1050).

## Reasoning alignment by treatment

| code   | label           |   prompt_profit_minus_compliance |   mean_reasoning_profit_orientation |   rank_difference |
|:-------|:----------------|---------------------------------:|------------------------------------:|------------------:|
| T1     | Compliance      |                           -0.134 |                              -0.011 |             0.000 |
| T2     | Balanced        |                            0.009 |                               0.028 |             0.000 |
| T3     | Guided Profit   |                            0.060 |                               0.057 |             0.000 |
| T4     | Baseline Profit |                            0.125 |                               0.066 |             1.000 |
| T5     | Profit-Max      |                            0.132 |                               0.065 |            -1.000 |
| T6     | High Profit-Max |                            0.147 |                               0.067 |             0.000 |

## Interpretation

The final six-treatment results support the paper's directional story: more profit-oriented objective blocks are associated with more profit-oriented generated TARJ rationales and more aggressive bidding outcomes. The relationship is not perfectly monotonic at every adjacent treatment, especially around Guided Profit and Baseline Profit, but the broader ranking is strong. The blocked regressions preserve the same positive direction after common day and seed conditions are absorbed, which is useful evidence that the result is not just driven by a particular week/day pattern.

The reasoning-language results should remain framed as semantic-behavioral alignment rather than causation. TARJ rationales are generated after the prompt intervention and alongside the bid, so they are informative artifacts of agent posture, not independent causal mechanisms.
