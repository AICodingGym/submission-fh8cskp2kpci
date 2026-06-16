# Learning Agency Lab - Automated Essay Scoring 2.0

Local workspace for the AI Coding Gym MLE challenge.

## Environment

Use the existing `foundationpose` environment. It already provides
`torch==2.0.1+cu118` and has been extended with the NLP dependencies for this
competition.

```bash
conda activate foundationpose
python -m ipykernel install --user --name foundationpose --display-name "Python (foundationpose)"
```

Install or refresh the non-torch dependencies:

```bash
conda activate foundationpose
pip install -r requirements-cu118.txt
```

Validate CUDA:

```bash
python scripts/check_env.py
```

## Baseline

Train a TF-IDF baseline with QWK threshold optimization:

```bash
python -m aes2.train_tfidf
```

This writes:

- `outputs/models/tfidf_ridge.joblib`
- `outputs/submissions/tfidf_ridge_submission.csv`

Submit with:

```bash
aicodinggym mle submit learning-agency-lab-automated-essay-scoring-2 \
  -F outputs/submissions/tfidf_ridge_submission.csv \
  -m "tfidf ridge baseline"
```

## Transformer Direction

The 4070 Ti / CUDA 11.8 path should start with `microsoft/deberta-v3-base`,
`max_length=512`, fp16, batch size 4-8, and gradient accumulation 2-4.
Use stratified folds and optimize thresholds for QWK.
