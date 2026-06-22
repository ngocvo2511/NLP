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

5. `vinai/phobert-base-v2`
   - Vietnamese-specific model.
   - Use `--tokenizer-alignment word` because the standard PhoBERT tokenizer is slow and does not provide fast offset mappings.
   - This keeps original character offsets, but it is not yet a full VnCoreNLP word-segmented PhoBERT setup.

## Preprocessing Position

Do not normalize, remove accents, word-segment, or rewrite raw text unless an offset mapping back to the original text is implemented. This dataset is span-based, so destructive preprocessing can invalidate `start` and `end` labels.

Clean preprocessing that is currently allowed:

- validate and clamp invalid offsets,
- keep original Unicode text,
- tokenizer alignment through fast tokenizer offset mappings,
- report noisy-text issues in error analysis.

PhoBERT's model card recommends word-segmented input and notes that standard Transformers includes a slow tokenizer. The current PhoBERT trial uses regex word alignment to preserve original offsets. A stricter future version can add VnCoreNLP word segmentation plus an explicit mapping back to original character offsets.
