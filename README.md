# Semantic Analysis of Prompt-Directed Bidding in Electricity Markets

Reproducibility materials for the accompanying paper on how prompt objectives
shape LLM-agent bidding in a simulated electricity market.

**Paper:** [PDF](paper/semantic_analysis_of_prompt-directed_bidding_in_electricity_markets.pdf)

## Paper snapshot

This is a frozen, paper-specific repository. It includes **only T1-T6**, the
six treatments reported in the paper, with five GPT-4.1 runs per treatment
(seeds 1-5; 30 runs total). Four later exploratory treatments are intentionally
excluded.

| Code | Treatment |
| --- | --- |
| T1 | Compliance |
| T2 | Balanced |
| T3 | Guided Profit |
| T4 | Baseline Profit |
| T5 | Profit-Max |
| T6 | High Profit-Max |

The complete mapping from paper treatment to prompt, run tag, and seed set is
in [the experiment registry](provenance/EXPERIMENT_REGISTRY.md).

## Repository contents

- `paper/` - final manuscript PDF and editable LaTeX source.
- `prompts/` and `config/` - exact treatment prompts and paper-only registry.
- `src/` - archived simulation, semantic-analysis, and paper-figure code.
- `data/market/` - market inputs and truthful benchmark inputs used by the runs.
- `results/raw-runs/` - exact output directories for all 30 paper runs.
- `results/analysis/six-treatment/` - cached tables and intermediate analyses.
- `paper/source/figures/` - the two final paper figures.
- `results/paper-figures/` - supporting tables and reports used to prepare the paper.
- `provenance/` - source state and checksums for this immutable snapshot.

`provenance/MANIFEST.sha256` is a cryptographic inventory of the frozen
snapshot. Verify it after cloning with `shasum -a 256 -c
provenance/MANIFEST.sha256`.

## Reproduce the paper figures

The figures can be regenerated from the included cached six-treatment analysis
tables; no API calls are made.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash reproduction/reproduce_figures.sh
```

Generated outputs are written to `results/regenerated/`. The frozen
final-candidate outputs used for the paper remain in `results/paper-figures/`.

## Provenance and scope

The original NAPS workspace was untracked when this repository was assembled.
Consequently, this repository's initial Git commit plus
`provenance/MANIFEST.sha256` define the citable paper snapshot. See
[SOURCE_STATE.md](provenance/SOURCE_STATE.md) for details.

Raw model-response material is included solely as a reproducibility record;
please cite the paper and repository rather than treating it as an independent
benchmark dataset.

## License

The code is released under the [MIT License](LICENSE-CODE). The paper and
research outputs are not covered by that code license.
