# Experiment Results

## Current Dev Results

All scores below are exact span + label micro scores on `data/dev.jsonl`.

| Experiment | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---|
| XLM-R BIO raw | 0.4097 | 0.5539 | 0.4710 | Best clean score so far from downloaded outputs. |
| PhoBERT base v2 BIO word-aligned | 0.4215 | 0.5243 | 0.4673 | Very close to XLM-R BIO; higher precision, lower recall. |
| XLM-R BIO raw, exact-span checkpoint selection | 0.3950 | 0.5394 | 0.4560 | Did not beat the earlier BIO run. |
| XLM-R BILOU raw | 0.3951 | 0.5503 | 0.4600 | Cleaner boundary scheme, but did not beat BIO. |
| XLM-R BILOU + sqrt-balanced loss | 0.2959 | 0.5681 | 0.3891 | Improved recall for rare classes, but too many false positives. |
| mBERT cased BIO | 0.3538 | 0.4978 | 0.4136 | Useful multilingual baseline, but worse than XLM-R. |
| XLM-R large BIO | 0.3905 | 0.5441 | 0.4547 | Larger model did not improve exact span F1 in this run. |
| mDeBERTa-v3 BIO | 0.0000 | 0.0000 | 0.0000 | Failed run: model predicted no spans after multiple attempts, omit from main comparison. |
| XLM-R BIO + heuristic post-processing | 0.5235 | 0.5544 | 0.5385 | Ablation only; not the main score because parameters are heuristic and dev-tuned. |

## Interpretation

PhoBERT base v2 with word alignment is competitive with XLM-R BIO but does not beat it overall. It improves precision and performs well on aspects such as `DESIGN`, `BATTERY`, and `SER&ACC`, but recall is lower and `NEUTRAL` remains almost unsolved. BILOU is valid, but this run did not improve exact span F1. Weighted loss increased recall, especially for neutral and rare classes, but precision collapsed because the model over-predicted spans. mBERT and XLM-R large were also below the earlier XLM-R BIO result. mDeBERTa-v3 failed repeatedly: it produced no predicted spans and had zero non-`O` token F1 during evaluation, so it should be treated as an incompatible/failed run rather than a competitive model.

The current main clean result should remain `XLM-R BIO raw` unless a later clean experiment beats it.

The next clean direction should avoid dev-tuned output heuristics and instead improve the training objective or model structure, for example:

- tune learning rate and epoch count for BIO raw,
- tune `vinai/phobert-base-v2` with a smaller learning rate or more epochs because it is close to XLM-R,
- try focal loss instead of balanced cross entropy,
- add a CRF layer or span-classification formulation,
- add confidence-threshold decoding only if the threshold is selected on dev and reported transparently.
