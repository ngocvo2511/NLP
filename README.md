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

Fine-tune a transformer, preferably on Colab/Kaggle GPU:

```bash
python -m src.absa.train_transformer \
  --model-name FacebookAI/xlm-roberta-base \
  --data-dir data \
  --output-dir outputs/xlmr-absa \
  --epochs 5 \
  --batch-size 8
```

For Vietnamese-only experiments, also try:

```bash
python -m src.absa.train_transformer --model-name vinai/phobert-base-v2 --data-dir data --output-dir outputs/phobert-absa
```

## Report Structure

- Dataset description: split sizes, aspect distribution, sentiment distribution.
- Task formulation: span extraction plus aspect-polarity classification.
- Preprocessing: offset validation, BIO conversion, tokenizer alignment.
- Baseline: keyword baseline or CRF/BiLSTM if added later.
- Main model: XLM-R or PhoBERT token classification.
- Evaluation: exact span + label micro precision, recall, F1.
- Error analysis: boundary errors, `GENERAL` confusion, neutral class, rare aspects such as `STORAGE`.
