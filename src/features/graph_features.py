from __future__ import annotations

from collections import defaultdict, deque
from datetime import timedelta

import numpy as np
import pandas as pd


def add_prior_graph_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create graph-flow features using only events before the current payment.

    The current transaction is scored first and inserted into the account history
    afterwards. This prevents future/test activity from entering pre-authorisation
    features.
    """
    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

    incoming_windows: dict[str, deque[tuple[pd.Timestamp, float]]] = defaultdict(deque)
    outgoing_windows: dict[str, deque[tuple[pd.Timestamp, float]]] = defaultdict(deque)
    incoming_sums: dict[str, float] = defaultdict(float)
    outgoing_sums: dict[str, float] = defaultdict(float)
    last_incoming_time: dict[str, pd.Timestamp] = {}
    holding_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=200))
    first_seen: dict[str, pd.Timestamp] = {}

    fan_in: list[int] = []
    fan_out: list[int] = []
    rapid_outflow: list[float] = []
    holding_minutes: list[float] = []
    account_age_days: list[float] = []

    one_hour = timedelta(hours=1)

    def purge(account: str, now: pd.Timestamp) -> None:
        while incoming_windows[account] and now - incoming_windows[account][0][0] > one_hour:
            _, old_amount = incoming_windows[account].popleft()
            incoming_sums[account] -= old_amount
        while outgoing_windows[account] and now - outgoing_windows[account][0][0] > one_hour:
            _, old_amount = outgoing_windows[account].popleft()
            outgoing_sums[account] -= old_amount

    for row in out.itertuples(index=False):
        now = pd.Timestamp(row.timestamp)
        payer = str(row.payer_id)
        payee = str(row.payee_id)
        amount = float(row.amount)

        purge(payee, now)

        # Capture the recipient's state before the current transaction arrives.
        current_fan_in = len(incoming_windows[payee])
        current_fan_out = len(outgoing_windows[payee])
        in_amount = max(incoming_sums[payee], 0.0)
        out_amount = max(outgoing_sums[payee], 0.0)
        ratio = out_amount / (in_amount + 1e-6)

        fan_in.append(current_fan_in)
        fan_out.append(current_fan_out)
        rapid_outflow.append(float(np.clip(ratio / 1.5, 0.0, 1.0)))
        holding_minutes.append(
            float(np.median(holding_history[payee])) if holding_history[payee] else 24.0 * 60.0
        )
        first = first_seen.get(payee, now)
        account_age_days.append(max(1.0, (now - first).total_seconds() / 86_400.0))

        # Update histories only after the current transaction has been scored.
        purge(payer, now)
        prior_incoming = last_incoming_time.get(payer)
        if prior_incoming is not None and now >= prior_incoming:
            holding_history[payer].append(
                max(0.01, (now - prior_incoming).total_seconds() / 60.0)
            )
        outgoing_windows[payer].append((now, amount))
        outgoing_sums[payer] += amount

        incoming_windows[payee].append((now, amount))
        incoming_sums[payee] += amount
        last_incoming_time[payee] = now

        first_seen.setdefault(payer, now)
        first_seen.setdefault(payee, now)

    out["fan_in_1h"] = fan_in
    out["fan_out_1h"] = fan_out
    out["rapid_outflow_ratio"] = rapid_outflow
    out["median_holding_minutes"] = np.clip(holding_minutes, 0.01, 30 * 24 * 60)
    out["recipient_account_age_days"] = account_age_days
    return out


def add_peer_normalised_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group = out.groupby("recipient_archetype", observed=True)
    for source, target in [("fan_in_1h", "peer_fan_in_z"), ("fan_out_1h", "peer_fan_out_z")]:
        med = group[source].transform("median")
        mad = group[source].transform(lambda s: np.median(np.abs(s - np.median(s))) + 1e-6)
        out[target] = ((out[source] - med) / (1.4826 * mad)).clip(-20, 20)
    out["flow_through_score"] = np.clip(
        0.55 * out.get("rapid_outflow_ratio", 0).astype(float)
        + 0.25 * np.exp(-out.get("median_holding_minutes", 60).astype(float) / 10)
        + 0.20 * (out.get("fan_out_1h", 0).astype(float) / (out.get("fan_out_1h", 0).astype(float) + 5)),
        0,
        1,
    )
    return out