from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.settings import get_settings


def describe_csv(path: Path, required: set[str]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = required.difference(header)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    print(f"OK  {path}  ({path.stat().st_size / (1024**2):.1f} MB)")


def find_banksim(path: Path) -> Path:
    required = {"step", "customer", "merchant", "amount", "fraud"}
    candidates = [path] if path.is_file() else list(path.rglob("*.csv"))
    for candidate in candidates:
        try:
            header = set(pd.read_csv(candidate, nrows=0).columns)
        except Exception:
            continue
        if required.issubset(header):
            return candidate
    raise FileNotFoundError(f"No valid BankSim transaction CSV found under {path}")


def main() -> None:
    settings = get_settings()
    describe_csv(
        settings.paysim_csv,
        {"step", "type", "amount", "nameOrig", "nameDest", "isFraud"},
    )
    banksim = find_banksim(settings.banksim_csv)
    describe_csv(banksim, {"step", "customer", "merchant", "amount", "fraud"})
    ibm_required = {
        "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
        "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering",
    }
    describe_csv(settings.ibm_hi_csv, ibm_required)
    describe_csv(settings.ibm_li_csv, ibm_required)

    for pattern in [
        settings.ibm_hi_csv.with_name("HI-Small_Patterns.txt"),
        settings.ibm_li_csv.with_name("LI-Small_Patterns.txt"),
    ]:
        if pattern.exists():
            print(f"OK  {pattern}  (retained for later typology evaluation)")
        else:
            print(f"WARN {pattern} is missing; model training can continue without it")
    print("All required transaction datasets are ready.")


if __name__ == "__main__":
    main()
