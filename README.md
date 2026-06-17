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

If direct access to `huggingface.co` fails, download the model through a
reachable mirror first:

```bash
mkdir -p models/hf
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  microsoft/deberta-v3-base \
  --local-dir models/hf/microsoft_deberta-v3-base \
  --local-dir-use-symlinks False
```

Then point training at the local directory and force offline loading:

```bash
python -m aes2.train_transformer \
  --model-name models/hf/microsoft_deberta-v3-base \
  --local-files-only \
  --fold 0 \
  --batch-size 4 \
  --grad-accum 4 \
  --epochs 4 \
  --fp16
```

Quick single-fold GPU run:

```bash
python -m aes2.train_transformer \
  --model-name microsoft/deberta-v3-base \
  --fold 0 \
  --batch-size 4 \
  --grad-accum 4 \
  --epochs 4 \
  --fp16
```

Full 5-fold ensemble:

```bash
python -m aes2.train_transformer \
  --model-name microsoft/deberta-v3-base \
  --fold -1 \
  --batch-size 4 \
  --grad-accum 4 \
  --epochs 4 \
  --fp16
```

Outputs are written to `outputs/models/` and `outputs/submissions/`.

## H100 Background Training

For an H100 server with `models/hf/microsoft_deberta-v3-large` already
downloaded locally, start the large 768-token 5-fold run in the background:

DeBERTa v3 uses SentencePiece. If the environment does not already have it,
install `sentencepiece` before launching.

```bash
chmod +x scripts/run_h100_deberta_large_len768_5fold.sh
chmod +x scripts/launch_h100_deberta_large_len768_5fold.sh

./scripts/launch_h100_deberta_large_len768_5fold.sh
```

The launcher writes a PID file and log under `logs/`. Follow progress with:

```bash
tail -f logs/h100_deberta_large_len768_5fold_*.log
```

The foreground command is available as:

```bash
./scripts/run_h100_deberta_large_len768_5fold.sh
```
