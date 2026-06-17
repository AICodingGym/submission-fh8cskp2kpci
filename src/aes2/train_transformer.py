from __future__ import annotations

import argparse
import inspect
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from aes2.data import load_sample_submission, load_test, load_train
from aes2.metrics import (
    SCORE_MAX,
    SCORE_MIN,
    apply_thresholds,
    optimize_thresholds,
    quadratic_weighted_kappa,
)
from aes2.paths import MODEL_DIR, SUBMISSION_DIR, ensure_output_dirs


@dataclass
class RunResult:
    fold: int
    valid_qwk: float
    thresholds: list[float]
    output_dir: str


class EssayDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        tokenizer: AutoTokenizer,
        max_length: int,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(values[idx], dtype=torch.long)
            for key, values in self.encodings.items()
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a transformer regression model for AES2."
    )
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold to train. Use -1 to train all folds and average test predictions.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-fast-tokenizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the fast tokenizer. Keep false for DeBERTa v3 on offline clusters.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--dev-limit",
        type=int,
        default=0,
        help="Limit rows for smoke tests. Keep 0 for real training.",
    )
    return parser.parse_args()


def safe_name(model_name: str) -> str:
    name = model_name.strip().replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def compute_rounded_qwk(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    predictions, labels = eval_pred
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).astype(int)
    rounded = np.rint(predictions).clip(SCORE_MIN, SCORE_MAX).astype(int)
    return {"qwk": quadratic_weighted_kappa(labels, rounded)}


def make_training_args(
    args: argparse.Namespace,
    output_dir: Path,
    train_size: int,
) -> TrainingArguments:
    steps_per_epoch = max(1, train_size // max(1, args.batch_size * args.grad_accum))
    eval_steps = max(50, steps_per_epoch // 2)
    kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": eval_steps,
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "qwk",
        "greater_is_better": True,
        "fp16": args.fp16,
        "logging_steps": 25,
        "report_to": [],
        "dataloader_num_workers": 2,
        "seed": args.seed,
    }
    training_args_params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_name = (
        "eval_strategy"
        if "eval_strategy" in training_args_params
        else "evaluation_strategy"
    )
    kwargs[strategy_name] = "steps"
    return TrainingArguments(**kwargs)


def train_one_fold(
    args: argparse.Namespace,
    fold: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> tuple[RunResult, np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=args.use_fast_tokenizer,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        problem_type="regression",
        local_files_only=args.local_files_only,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    texts = train["full_text"].astype(str).tolist()
    labels = train["score"].astype(float).to_numpy()
    test_texts = test["full_text"].astype(str).tolist()

    train_dataset = EssayDataset(
        [texts[i] for i in train_idx],
        tokenizer=tokenizer,
        max_length=args.max_length,
        labels=labels[train_idx],
    )
    valid_dataset = EssayDataset(
        [texts[i] for i in valid_idx],
        tokenizer=tokenizer,
        max_length=args.max_length,
        labels=labels[valid_idx],
    )
    test_dataset = EssayDataset(
        test_texts,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    run_slug = args.run_name or safe_name(args.model_name)
    output_dir = MODEL_DIR / f"transformer_{run_slug}_fold{fold}"
    training_args = make_training_args(args, output_dir, len(train_dataset))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_rounded_qwk,
    )
    trainer.train()

    valid_raw = trainer.predict(valid_dataset).predictions.reshape(-1)
    test_raw = trainer.predict(test_dataset).predictions.reshape(-1)
    thresholds = optimize_thresholds(labels[valid_idx].astype(int), valid_raw)
    valid_pred = apply_thresholds(valid_raw, thresholds)
    valid_qwk = quadratic_weighted_kappa(labels[valid_idx].astype(int), valid_pred)

    trainer.save_model(output_dir / "best")
    tokenizer.save_pretrained(output_dir / "best")
    np.save(output_dir / "thresholds.npy", thresholds)
    result = RunResult(
        fold=fold,
        valid_qwk=float(valid_qwk),
        thresholds=[float(x) for x in thresholds],
        output_dir=str(output_dir),
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)

    return result, valid_raw, test_raw


def selected_folds(args: argparse.Namespace) -> list[int]:
    if args.fold == -1:
        return list(range(args.folds))
    if args.fold < 0 or args.fold >= args.folds:
        raise ValueError(f"--fold must be -1 or in [0, {args.folds - 1}]")
    return [args.fold]


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    set_seed(args.seed)

    train = load_train()
    test = load_test()
    sample_submission = load_sample_submission()
    if args.dev_limit:
        train = train.head(args.dev_limit).copy()
        test = test.head(min(args.dev_limit, len(test))).copy()
        sample_submission = sample_submission.head(len(test)).copy()

    labels = train["score"].astype(int).to_numpy()
    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    splits = list(splitter.split(train["full_text"], labels))
    fold_ids = selected_folds(args)

    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"training_folds={fold_ids}")
    print(f"model_name={args.model_name}")

    oof_raw = np.full(len(train), np.nan, dtype=float)
    test_raw = np.zeros(len(test), dtype=float)
    results = []

    for fold in fold_ids:
        train_idx, valid_idx = splits[fold]
        result, valid_raw, fold_test_raw = train_one_fold(
            args=args,
            fold=fold,
            train=train,
            test=test,
            train_idx=train_idx,
            valid_idx=valid_idx,
        )
        oof_raw[valid_idx] = valid_raw
        test_raw += fold_test_raw / len(fold_ids)
        results.append(result)
        print(
            f"fold={fold} qwk={result.valid_qwk:.5f} "
            f"thresholds={result.thresholds}"
        )

    trained_mask = ~np.isnan(oof_raw)
    if trained_mask.all():
        thresholds = optimize_thresholds(labels, oof_raw)
        cv_qwk = quadratic_weighted_kappa(labels, apply_thresholds(oof_raw, thresholds))
        threshold_source = "oof_all_folds"
    else:
        first = results[0]
        thresholds = np.asarray(first.thresholds, dtype=float)
        cv_qwk = first.valid_qwk
        threshold_source = f"fold{first.fold}_valid"

    predictions = apply_thresholds(test_raw, thresholds)
    run_slug = args.run_name or safe_name(args.model_name)
    fold_tag = "all" if args.fold == -1 else f"fold{args.fold}"
    submission_path = (
        SUBMISSION_DIR / f"transformer_{run_slug}_{fold_tag}_submission.csv"
    )
    submission = pd.DataFrame(
        {
            "essay_id": sample_submission["essay_id"],
            "score": predictions,
        }
    )
    submission.to_csv(submission_path, index=False)

    summary = {
        "args": vars(args),
        "results": [asdict(result) for result in results],
        "qwk": float(cv_qwk),
        "thresholds": [float(x) for x in thresholds],
        "threshold_source": threshold_source,
        "submission": str(submission_path),
    }
    summary_path = MODEL_DIR / f"transformer_{run_slug}_{fold_tag}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"qwk={cv_qwk:.5f} threshold_source={threshold_source}")
    print(f"wrote {submission_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
