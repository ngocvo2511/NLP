# ABSA Implementation Plan

## Goal

Build a Vietnamese aspect-based sentiment analysis system that predicts opinion spans and their `ASPECT#POLARITY` labels.

## Dataset Observations

- Format: JSONL with `text` and span labels `[start, end, "ASPECT#POLARITY"]`.
- Train/dev/test rows: 7,785 / 1,112 / 2,225.
- Average labels per review: about 3.2.
- Main aspects: `GENERAL`, `PERFORMANCE`, `BATTERY`, `FEATURES`, `CAMERA`, `SER&ACC`, `DESIGN`, `PRICE`, `SCREEN`, `STORAGE`.
- Sentiment distribution is imbalanced: positive is dominant, neutral is rare.
- Some offsets exceed text length slightly, so preprocessing clamps them to valid character boundaries.

## Execution Order

1. Data validation
   - Run `src.absa.validate_data`.
   - Record aspect and polarity distributions for the report.

2. Preprocessing
   - Convert character spans to BIO token tags.
   - Keep exact character spans for final evaluation.

3. Baseline
   - Run `src.absa.keyword_baseline`.
   - Use it mainly to validate the evaluator and establish a weak lower bound.

4. Main model
   - Fine-tune `FacebookAI/xlm-roberta-base` first.
   - Try `vinai/phobert-base-v2` as a Vietnamese-specific comparison.
   - Try `BILOU` tags as a clean boundary-modeling improvement over `BIO`.
   - Try train-set-derived class weights to reduce polarity/aspect imbalance.
   - Train on Colab/Kaggle GPU, then download outputs locally.

5. Evaluation
   - Convert model BIO predictions back to spans.
   - Evaluate exact span + label micro precision, recall, and F1.
   - Report scores by aspect and polarity.
   - Keep heuristic post-processing, if used, as a separate ablation result rather than the main score.

6. Error analysis
   - Boundary errors: correct label but span too short/long.
   - Aspect confusion: especially `GENERAL` versus specific aspects.
   - Rare classes: `STORAGE`, `NEUTRAL`.
   - Sentiment ambiguity in informal reviews.

7. Final deliverables
   - Source code and reproducible commands.
   - Model metrics and prediction files.
   - Report with dataset statistics, methods, results, and error analysis.
   - Optional Hugging Face Spaces demo for inference.
