from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transformer training entrypoint.")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Transformer plan:")
    print(args)
    print(
        "Implement fine-tuning here after the TF-IDF baseline is verified. "
        "Use regression or ordinal classification, fp16, stratified folds, "
        "and QWK threshold optimization."
    )


if __name__ == "__main__":
    main()
