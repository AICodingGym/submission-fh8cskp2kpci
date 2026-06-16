# AES2 Journal

## Workspace Decisions

- Environment manager: conda
- Environment name: foundationpose
- CUDA/PyTorch: CUDA 11.8, torch 2.0.1+cu118
- Tabular dataframe: pandas
- Deep-learning framework: pytorch
- Evaluation metric: quadratic weighted kappa
- Primary local GPU target: RTX 4070 Ti

## Experiments

| ID | Status | Description | CV QWK | Submission | Notes |
| --- | --- | --- | --- | --- | --- |
| 001 | planned | TF-IDF word/char + Ridge + threshold optimization | | | First stable baseline |
