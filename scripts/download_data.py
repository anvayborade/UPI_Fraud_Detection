from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DATASETS = {
    "paysim": {
        "slug": "ealaxi/paysim1",
        "destination": Path("data/raw/paysim"),
        "files": None,
    },
    "banksim": {
        "slug": "ealaxi/banksim1",
        "destination": Path("data/raw/banksim"),
        "files": None,
    },
    "ibm_aml": {
        "slug": "ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
        "destination": Path("data/raw/ibm_aml"),
        "files": [
            "HI-Small_Trans.csv",
            "HI-Small_Patterns.txt",
            "LI-Small_Trans.csv",
            "LI-Small_Patterns.txt",
        ],
    },
}


def _kaggle_executable() -> str:
    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError(
            "The Kaggle CLI is not installed or is not on PATH. Run 'pip install kaggle' "
            "and configure ~/.kaggle/kaggle.json first."
        )
    return executable


def _run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def download_dataset(name: str) -> None:
    specification = DATASETS[name]
    destination: Path = specification["destination"]
    destination.mkdir(parents=True, exist_ok=True)
    executable = _kaggle_executable()
    base = [
        executable,
        "datasets",
        "download",
        "-d",
        str(specification["slug"]),
        "-p",
        str(destination),
        "--unzip",
    ]
    files: list[str] | None = specification["files"]
    if files:
        for filename in files:
            _run(base + ["-f", filename])
    else:
        _run(base)
    print(f"Downloaded {name} to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PaySim, BankSim and IBM AML Small datasets")
    parser.add_argument("dataset", choices=["paysim", "banksim", "ibm_aml", "all"])
    args = parser.parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        download_dataset(name)


if __name__ == "__main__":
    main()
