# Vietnamese ABSA Project

Pipeline for aspect-based sentiment analysis on the JSONL dataset in `data/`.

Each row has:

```json
{"text": "...", "labels": [[start, end, "ASPECT#POLARITY"]]}
```

The project treats this as span-level ABSA: extract opinion spans and classify each span with `ASPECT#POLARITY`.

## Recommended Workflow

1. Validate and inspect the dataset locally.
2. Run a simple keyword baseline to verify the evaluator.
3. Fine-tune a transformer model on Colab/Kaggle.
4. Export predictions and run exact span-level evaluation locally.
5. Analyze errors by aspect and polarity for the report.

## Local Setup

Use Python 3.10+.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `py` is not available on this Windows machine, the bundled Codex runtime also works for local checks:

```powershell
& 'C:\Users\ACER\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m src.absa.validate_data --data-dir data
```

The core validation/evaluation scripts only need the Python standard library. Transformer training needs the optional ML packages in `requirements.txt`.

## Commands

Validate offsets and label distribution:

```powershell
py -3.14 -m src.absa.validate_data --data-dir data
```

Run the keyword baseline:

```powershell
py -3.14 -m src.absa.keyword_baseline --data-dir data --split dev --output outputs/keyword_dev_predictions.jsonl
```

Evaluate predictions:

```powershell
py -3.14 -m src.absa.evaluate --gold data/dev.jsonl --pred outputs/keyword_dev_predictions.jsonl
```

Optional post-processing ablation:

```powershell
py -3.14 -m src.absa.postprocess_predictions --input outputs/xlmr_dev_predictions.jsonl --output outputs/xlmr_dev_predictions_pp.jsonl --min-span-chars 6 --merge-gap 30
py -3.14 -m src.absa.evaluate --gold data/dev.jsonl --pred outputs/xlmr_dev_predictions_pp.jsonl --json-output outputs/xlmr_dev_metrics_pp.json
```

This heuristic post-processing should not be treated as the main score. Use it only as an ablation/error-analysis result and disclose the rule and dev-tuned parameters.

Fine-tune a transformer, preferably on Colab/Kaggle GPU:

For Colab, prefer `notebooks/train_colab_drive.ipynb`. It saves checkpoints, predictions, and metrics to Google Drive under `MyDrive/NLP_outputs`, so outputs survive runtime disconnects.

For model optimization, use `notebooks/model_sweep_colab_drive.ipynb`. It tries XLM-R base, mDeBERTa-v3, multilingual BERT, and XLM-R large while saving all artifacts to Google Drive.

```bash
python -m src.absa.train_transformer \
  --model-name FacebookAI/xlm-roberta-base \
  --data-dir data \
  --output-dir outputs/xlmr-absa \
  --tag-scheme bio \
  --class-weight none \
  --epochs 5 \
  --batch-size 8
```

Clean follow-up experiments:

```bash
# Better boundary modeling than BIO.
python -m src.absa.train_transformer \
  --model-name FacebookAI/xlm-roberta-base \
  --data-dir data \
  --output-dir outputs/xlmr-bilou \
  --tag-scheme bilou \
  --class-weight none \
  --epochs 5 \
  --batch-size 8

# Handle label imbalance using train-set-derived weights.
python -m src.absa.train_transformer \
  --model-name FacebookAI/xlm-roberta-base \
  --data-dir data \
  --output-dir outputs/xlmr-bio-weighted \
  --tag-scheme bio \
  --class-weight sqrt-balanced \
  --epochs 5 \
  --batch-size 8

# Combine both clean improvements.
python -m src.absa.train_transformer \
  --model-name FacebookAI/xlm-roberta-base \
  --data-dir data \
  --output-dir outputs/xlmr-bilou-weighted \
  --tag-scheme bilou \
  --class-weight sqrt-balanced \
  --epochs 5 \
  --batch-size 8
```

For Vietnamese-only experiments, also try:

```bash
python -m src.absa.train_transformer \
  --model-name vinai/phobert-base-v2 \
  --data-dir data \
  --output-dir outputs/phobert-base-v2-bio-wordalign \
  --tokenizer-alignment word \
  --tag-scheme bio \
  --epochs 6 \
  --batch-size 8
```

PhoBERT's standard tokenizer is slow and does not provide fast offset mappings in mainline Transformers, so this project uses word-level alignment to preserve original character offsets for span evaluation.

CRF experiments:

```bash
# Token embedding + character CNN + BiLSTM + CRF.
python -m src.absa.train_bilstm_crf \
  --data-dir data \
  --output-dir outputs/bilstm-crf \
  --batch-tokens 5000 \
  --epochs 30 \
  --word-dim 300 \
  --hidden-size 512 \
  --dropout 0.5

# PhoBERT contextual embeddings + character CNN + BiLSTM + CRF.
python -m src.absa.train_transformer_bilstm_crf \
  --model-name vinai/phobert-base-v2 \
  --data-dir data \
  --output-dir outputs/phobert-bilstm-crf \
  --epochs 12 \
  --batch-size 8 \
  --hidden-size 512
```

## Report Structure

- Dataset description: split sizes, aspect distribution, sentiment distribution.
- Task formulation: span extraction plus aspect-polarity classification.
- Preprocessing: offset validation, BIO conversion, tokenizer alignment.
- Optional post-processing ablation: merge nearby same-label fragments and filter very short spans, reported separately from the main score.
- Baseline: keyword baseline or CRF/BiLSTM if added later.
- Main model: XLM-R or PhoBERT token classification.
- Evaluation: exact span + label micro precision, recall, F1.
- Error analysis: boundary errors, `GENERAL` confusion, neutral class, rare aspects such as `STORAGE`.
