from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATES = {
    "fake_refund_qr": "The caller claimed to process a refund and asked me to scan a QR code urgently. I entered my UPI PIN and money was debited.",
    "collect_request_scam": "A person posing as customer care sent a collect request and said it was required to receive money.",
    "remote_access_scam": "The caller asked me to install a screen-sharing application and guided me through a UPI payment.",
    "account_takeover": "My UPI account was re-registered on another device after a SIM issue and an unknown transfer was made.",
    "mule_directed": "Money was sent to this beneficiary and immediately moved through several connected accounts.",
}

LEGITIMATE_DISPUTE = (
    "I raised a query about this payment because I did not immediately recognise the merchant, "
    "but it may be a genuine transaction."
)


def generate_complaints(
    transactions: pd.DataFrame,
    output_path: Path,
    seed: int = 42,
    max_rows: int = 500,
    fraud_report_rate: float = 0.45,
    legitimate_dispute_rate: float = 0.01,
) -> pd.DataFrame:
    """Generate delayed and incomplete reports.

    Only a portion of fraud is reported, some genuine transactions are disputed,
    and every complaint becomes available after the payment timestamp.
    """
    rng = np.random.default_rng(seed)
    frame = transactions.copy()
    fraud_mask = frame["scenario"].isin(TEMPLATES)
    selected_mask = (fraud_mask & (rng.random(len(frame)) < fraud_report_rate)) | (
        ~fraud_mask & (rng.random(len(frame)) < legitimate_dispute_rate)
    )
    selected = frame.loc[selected_mask].copy()
    if len(selected) > max_rows:
        selected = selected.sample(n=max_rows, random_state=seed)

    rows: list[dict] = []
    for idx, record in enumerate(selected.itertuples(index=False)):
        is_fraud_report = str(record.scenario) in TEMPLATES
        base_text = TEMPLATES.get(str(record.scenario), LEGITIMATE_DISPUTE)
        noise = rng.choice(
            ["Please investigate.", "This happened very quickly.", "I did not know the recipient.", "I noticed it later."],
        )
        transaction_time = pd.Timestamp(record.timestamp)
        delay_hours = int(rng.integers(1, 24 * 7 + 1))
        complaint_time = transaction_time + pd.Timedelta(hours=delay_hours)
        rows.append(
            {
                "complaint_id": f"CMP{idx:07d}",
                "transaction_id": record.transaction_id,
                "payee_id": record.payee_id,
                "transaction_timestamp": transaction_time,
                "complaint_timestamp": complaint_time,
                "reported_is_fraud": int(is_fraud_report),
                "text": f"{base_text} {noise}",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_parquet(output_path, index=False)
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic fraud complaint narratives")
    parser.add_argument("--transactions", type=Path, default=Path("data/processed/upi_transactions.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/complaints/raw_complaints.parquet"))
    parser.add_argument("--max-rows", type=int, default=500)
    args = parser.parse_args()
    if not args.transactions.exists():
        raise FileNotFoundError(
            f"{args.transactions} is missing. Run 'python -m src.data.prepare_multisource' first."
        )
    frame = pd.read_parquet(args.transactions)
    output = generate_complaints(frame, args.output, max_rows=args.max_rows)
    print(f"Generated {len(output):,} complaints at {args.output}")


if __name__ == "__main__":
    main()
