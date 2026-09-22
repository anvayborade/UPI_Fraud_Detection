from __future__ import annotations

from prometheus_client import Counter, Histogram

SCORED_TRANSACTIONS = Counter(
    "upi_scored_transactions_total",
    "Number of UPI transactions scored",
    ["action"],
)
SCORING_LATENCY = Histogram(
    "upi_scoring_latency_seconds",
    "End-to-end transaction scoring latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
)
COMPLAINTS_PROCESSED = Counter(
    "upi_complaints_processed_total",
    "Number of complaints converted into structured intelligence",
    ["scam_type"],
)
