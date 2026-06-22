# Next Clean Experiments

The current best clean dev result is `XLM-R BIO raw` with exact span + label F1 around `0.4710`.

## Why Change The Training Metric

Earlier runs selected the best checkpoint using token-level F1. That is not the project metric. The training script now reports and selects by `exact_span_f1`, which is closer to the final ABSA evaluation.

## Model Sweep Order

Run these in `notebooks/model_sweep_colab_drive.ipynb`, one section at a time:

1. `FacebookAI/xlm-roberta-base`
   - Same model family as the current best result.
   - New setting: exact-span checkpoint selection, warmup, 6 epochs.

2. `microsoft/mdeberta-v3-base`
   - Multilingual DeBERTa V3 model.
   - Worth trying because mDeBERTa reports strong multilingual NLU performance and includes Vietnamese.

3. `google-bert/bert-base-multilingual-cased`
   - Older but simple multilingual baseline.
   - Useful to show XLM-R is not only compared against a weak keyword baseline.

4. `FacebookAI/xlm-roberta-large`
   - Larger capacity model.
   - Run only if GPU memory allows it.

## Preprocessing Position

Do not normalize, remove accents, word-segment, or rewrite raw text unless an offset mapping back to the original text is implemented. This dataset is span-based, so destructive preprocessing can invalidate `start` and `end` labels.

Clean preprocessing that is currently allowed:

- validate and clamp invalid offsets,
- keep original Unicode text,
- tokenizer alignment through fast tokenizer offset mappings,
- report noisy-text issues in error analysis.

PhoBERT is not the first clean target because its model card recommends word-segmented input. Word segmentation changes character offsets, so it needs a separate alignment layer before being used fairly for exact span prediction.
