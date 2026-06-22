# Experiment Results

## Current Dev Results

All scores below are exact span + label micro scores on `data/dev.jsonl`.

| Experiment | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---|
| XLM-R BIO raw | 0.4097 | 0.5539 | 0.4710 | Best clean score so far from downloaded outputs. |
| XLM-R BILOU raw | 0.3951 | 0.5503 | 0.4600 | Cleaner boundary scheme, but did not beat BIO. |
| XLM-R BILOU + sqrt-balanced loss | 0.2959 | 0.5681 | 0.3891 | Improved recall for rare classes, but too many false positives. |
| XLM-R BIO + heuristic post-processing | 0.5235 | 0.5544 | 0.5385 | Ablation only; not the main score because parameters are heuristic and dev-tuned. |

## Interpretation

BILOU is valid, but this run did not improve exact span F1. Weighted loss increased recall, especially for neutral and rare classes, but precision collapsed because the model over-predicted spans. The current main clean result should remain `XLM-R BIO raw` unless a later clean experiment beats it.

The next clean direction should avoid dev-tuned output heuristics and instead improve the training objective or model structure, for example:

- tune learning rate and epoch count for BIO raw,
- try `vinai/phobert-base-v2`,
- try focal loss instead of balanced cross entropy,
- add a CRF layer or span-classification formulation,
- add confidence-threshold decoding only if the threshold is selected on dev and reported transparently.
