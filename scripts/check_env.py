import importlib

import torch


def main() -> None:
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
        print(f"cuda_capability={torch.cuda.get_device_capability(0)}")

    for package in [
        "pandas",
        "numpy",
        "scipy",
        "sklearn",
        "transformers",
        "datasets",
        "accelerate",
    ]:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "unknown")
        print(f"{package}={version}")


if __name__ == "__main__":
    main()
