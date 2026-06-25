# Experiment Results

## Current Dev Results

All scores below are exact span + label micro scores on `data/dev.jsonl`.

| Experiment | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---|
| PhoBERT + BiLSTM-CRF | 0.5943 | 0.5966 | 0.5954 | Best clean score so far; CRF substantially improves boundary/label consistency. |
| BiLSTM-CRF | 0.5642 | 0.5374 | 0.5505 | Strong non-transformer CRF baseline; beats all token-classification transformer runs. |
| XLM-R BIO raw | 0.4097 | 0.5539 | 0.4710 | Best plain transformer token-classification run. |
| PhoBERT base v2 BIO word-aligned | 0.4215 | 0.5243 | 0.4673 | Very close to XLM-R BIO; higher precision, lower recall. |
| XLM-R BIO raw, exact-span checkpoint selection | 0.3950 | 0.5394 | 0.4560 | Did not beat the earlier BIO run. |
| XLM-R BILOU raw | 0.3951 | 0.5503 | 0.4600 | Cleaner boundary scheme, but did not beat BIO. |
| XLM-R BILOU + sqrt-balanced loss | 0.2959 | 0.5681 | 0.3891 | Improved recall for rare classes, but too many false positives. |
| mBERT cased BIO | 0.3538 | 0.4978 | 0.4136 | Useful multilingual baseline, but worse than XLM-R. |
| XLM-R large BIO | 0.3905 | 0.5441 | 0.4547 | Larger model did not improve exact span F1 in this run. |
| mDeBERTa-v3 BIO | 0.0000 | 0.0000 | 0.0000 | Failed run: model predicted no spans after multiple attempts, omit from main comparison. |
| XLM-R BIO + heuristic post-processing | 0.5235 | 0.5544 | 0.5385 | Ablation only; not the main score because parameters are heuristic and dev-tuned. |

## Interpretation

The CRF direction is clearly the strongest so far. Plain BiLSTM-CRF reaches `0.5505` F1, and PhoBERT + BiLSTM-CRF reaches `0.5954` F1 without heuristic post-processing. CRF decoding reduces false positives compared with plain token classification and significantly improves `NEGATIVE` and `NEUTRAL` polarities. PhoBERT base v2 with word alignment is competitive with XLM-R BIO but does not beat it overall as a plain token classifier. BILOU is valid, but this run did not improve exact span F1. Weighted loss increased recall, especially for neutral and rare classes, but precision collapsed because the model over-predicted spans. mBERT and XLM-R large were also below the earlier XLM-R BIO result. mDeBERTa-v3 failed repeatedly: it produced no predicted spans and had zero non-`O` token F1 during evaluation, so it should be treated as an incompatible/failed run rather than a competitive model.

The current main clean result should be `PhoBERT + BiLSTM-CRF`.

The next clean direction should avoid dev-tuned output heuristics and instead improve the training objective or model structure, for example:

- continue tuning `PhoBERT + BiLSTM-CRF` because it is now close to the target `0.6+` range,
- try 20-30 epochs if dev F1 is still rising,
- try `hidden-size 768`, `dropout 0.4`, or `head-learning-rate 5e-4`,
- run 2-3 seeds for the final CRF configuration,
- keep heuristic post-processing only as an ablation, not as the main score.
