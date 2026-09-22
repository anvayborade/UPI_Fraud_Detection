from __future__ import annotations

"""Stable recipient identities and chronological compromise transitions.

The previous generator assigned a scenario independently to every transaction.  That
allowed one payee to behave as a hospital, auto driver, mule and QR scammer at the
same time.  This module assigns one persistent recipient profile per payee.  Fraud
profiles may have a legitimate warm-up period followed by a compromise/activation
point, which is a realistic chronological state transition rather than random
row-level identity changes.
"""

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd


FRAUD_PROFILES = {
    "money_mule",
    "qr_scam_recipient",
    "collect_scam_recipient",
    "remote_scam_recipient",
    "compromised_merchant",
}

PROFILE_TO_ARCHETYPE = {
    "personal": "personal",
    "merchant": "merchant",
    "transport_provider": "transport_provider",
    "hospital": "merchant",
    "hotel": "merchant",
    "gig_worker": "gig_worker",
    "new_business": "new_business",
    "money_mule": "money_mule",
    "qr_scam_recipient": "personal",
    "collect_scam_recipient": "personal",
    "remote_scam_recipient": "personal",
    "compromised_merchant": "merchant",
}

PROFILE_TO_CATEGORY = {
    "personal": "PERSONAL",
    "merchant": "SERVICES",
    "transport_provider": "TRANSPORT",
    "hospital": "HEALTHCARE",
    "hotel": "HOTEL",
    "gig_worker": "SERVICES",
    "new_business": "SERVICES",
    "money_mule": "PERSONAL",
    "qr_scam_recipient": "PERSONAL",
    "collect_scam_recipient": "PERSONAL",
    "remote_scam_recipient": "PERSONAL",
    "compromised_merchant": "SERVICES",
}


@dataclass(frozen=True)
class RecipientProfile:
    payee_id: str
    profile: str
    archetype: str
    merchant_category: str
    compromise_fraction: float
    has_source_fraud: int


def _stable_unit_interval(value: str, seed: int, salt: str) -> float:
    payload = f"{seed}|{salt}|{value}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def _stable_choice(value: str, seed: int, salt: str, choices: list[str], probabilities: list[float]) -> str:
    u = _stable_unit_interval(value, seed, salt)
    cumulative = 0.0
    for choice, probability in zip(choices, probabilities, strict=True):
        cumulative += probability
        if u <= cumulative:
            return choice
    return choices[-1]


def _normalise_category(value: object) -> str:
    return str(value).strip().strip("'").strip('"').upper()


def _category_profile(category: str, source: str, payee_id: str, seed: int) -> str:
    category = _normalise_category(category)
    if any(token in category for token in ["TRANSPORT", "TAXI", "TRAVEL"]):
        return "transport_provider"
    if any(token in category for token in ["HEALTH", "HOSPITAL", "PHARMA"]):
        return "hospital"
    if any(token in category for token in ["HOTEL", "ACCOMMODATION"]):
        return "hotel"
    if any(token in category for token in ["SERVICE", "RESTAURANT", "GROCERY", "RETAIL"]):
        return "merchant"
    if source == "banksim":
        return _stable_choice(
            payee_id,
            seed,
            "banksim_legitimate_profile",
            ["merchant", "transport_provider", "hospital", "hotel", "gig_worker", "new_business"],
            [0.52, 0.14, 0.08, 0.08, 0.10, 0.08],
        )
    return _stable_choice(
        payee_id,
        seed,
        "paysim_legitimate_profile",
        ["personal", "merchant", "transport_provider", "gig_worker", "new_business"],
        [0.55, 0.20, 0.08, 0.08, 0.09],
    )


def build_recipient_profiles(
    base: pd.DataFrame,
    *,
    seed: int,
    synthetic_fraud_rate: float,
) -> pd.DataFrame:
    """Return one persistent profile for every recipient."""
    required = {"payee_id", "source_dataset", "merchant_category_source", "base_is_fraud"}
    missing = sorted(required.difference(base.columns))
    if missing:
        raise ValueError(f"Recipient profiling is missing columns: {missing}")

    summaries = (
        base.groupby("payee_id", observed=True, sort=False)
        .agg(
            source_dataset=("source_dataset", "first"),
            merchant_category_source=(
                "merchant_category_source",
                lambda values: values.astype(str).mode().iloc[0] if not values.empty else "UNKNOWN",
            ),
            has_source_fraud=("base_is_fraud", "max"),
            transaction_count=("transaction_id", "count"),
        )
        .reset_index()
    )

    # A fraud profile affects only transactions after its activation point.  The
    # multiplier gives roughly the requested row-level synthetic fraud rate after
    # accounting for legitimate warm-up transactions.
    fraud_profile_rate = min(0.30, max(0.04, synthetic_fraud_rate * 3.0))
    rows: list[dict[str, object]] = []
    for record in summaries.itertuples(index=False):
        payee_id = str(record.payee_id)
        source = str(record.source_dataset)
        category = _normalise_category(record.merchant_category_source)
        source_fraud = int(record.has_source_fraud)
        legitimate_profile = _category_profile(category, source, payee_id, seed)

        should_be_fraud_profile = source_fraud == 1 or (
            _stable_unit_interval(payee_id, seed, "fraud_profile") < fraud_profile_rate
        )
        if should_be_fraud_profile:
            if source == "banksim":
                profile = _stable_choice(
                    payee_id,
                    seed,
                    "fraud_profile_banksim",
                    ["money_mule", "qr_scam_recipient", "collect_scam_recipient", "remote_scam_recipient", "compromised_merchant"],
                    [0.32, 0.25, 0.16, 0.10, 0.17],
                )
            else:
                profile = _stable_choice(
                    payee_id,
                    seed,
                    "fraud_profile_paysim",
                    ["money_mule", "qr_scam_recipient", "collect_scam_recipient", "remote_scam_recipient", "compromised_merchant"],
                    [0.45, 0.16, 0.16, 0.08, 0.15],
                )
            # Every malicious recipient has a legitimate-looking warm-up period.
            compromise_fraction = 0.40 + 0.35 * _stable_unit_interval(
                payee_id, seed, "compromise_fraction"
            )
        else:
            profile = legitimate_profile
            compromise_fraction = 1.10

        rows.append(
            {
                "payee_id": payee_id,
                "recipient_profile": profile,
                "recipient_archetype": PROFILE_TO_ARCHETYPE[profile],
                "recipient_merchant_category": (
                    category
                    if category not in {"UNKNOWN", "NAN", "NONE", ""}
                    and profile in {"merchant", "compromised_merchant"}
                    else PROFILE_TO_CATEGORY[profile]
                ),
                "compromise_fraction": float(compromise_fraction),
                "recipient_has_source_fraud": source_fraud,
            }
        )
    return pd.DataFrame(rows)


def _legitimate_scenario(profile: str, key: str, seed: int) -> str:
    mapping: dict[str, tuple[list[str], list[float]]] = {
        "personal": (["familiar_legitimate", "legitimate_split_payment"], [0.92, 0.08]),
        "merchant": (["familiar_legitimate", "new_merchant_legitimate", "legitimate_split_payment"], [0.78, 0.15, 0.07]),
        "transport_provider": (["auto_long_distance_legitimate", "gig_worker_payment_legitimate", "familiar_legitimate"], [0.62, 0.23, 0.15]),
        "hospital": (["hospital_emergency_legitimate", "familiar_legitimate"], [0.78, 0.22]),
        "hotel": (["travel_hotel_legitimate", "familiar_legitimate"], [0.78, 0.22]),
        "gig_worker": (["gig_worker_payment_legitimate", "familiar_legitimate"], [0.78, 0.22]),
        "new_business": (["new_merchant_legitimate", "familiar_legitimate"], [0.76, 0.24]),
        # Fraud profiles behave normally before activation.
        "money_mule": (["familiar_legitimate", "legitimate_split_payment"], [0.90, 0.10]),
        "qr_scam_recipient": (["familiar_legitimate", "new_merchant_legitimate"], [0.86, 0.14]),
        "collect_scam_recipient": (["familiar_legitimate", "new_merchant_legitimate"], [0.88, 0.12]),
        "remote_scam_recipient": (["familiar_legitimate", "new_merchant_legitimate"], [0.88, 0.12]),
        "compromised_merchant": (["familiar_legitimate", "new_merchant_legitimate"], [0.82, 0.18]),
    }
    choices, probabilities = mapping[profile]
    return _stable_choice(key, seed, "legitimate_scenario", choices, probabilities)


def _fraud_scenario(profile: str, key: str, seed: int) -> str:
    if profile == "money_mule":
        return "mule_directed"
    if profile == "qr_scam_recipient":
        return "fake_refund_qr"
    if profile == "collect_scam_recipient":
        return "collect_request_scam"
    if profile == "remote_scam_recipient":
        return "remote_access_scam"
    if profile == "compromised_merchant":
        return _stable_choice(
            key,
            seed,
            "compromised_merchant_fraud",
            ["fake_refund_qr", "collect_request_scam", "mule_directed"],
            [0.48, 0.30, 0.22],
        )
    return "account_takeover"


def assign_recipient_aware_scenarios(
    base: pd.DataFrame,
    *,
    seed: int,
    synthetic_fraud_rate: float,
) -> pd.DataFrame:
    """Assign scenarios using stable payee profiles and chronological state."""
    out = base.copy().sort_values(["payee_id", "timestamp", "transaction_id"]).reset_index(drop=True)
    profiles = build_recipient_profiles(out, seed=seed, synthetic_fraud_rate=synthetic_fraud_rate)
    out = out.merge(profiles, on="payee_id", how="left", validate="many_to_one")

    out["payee_event_index"] = out.groupby("payee_id", observed=True).cumcount()
    out["payee_event_count"] = out.groupby("payee_id", observed=True)["transaction_id"].transform("count")
    denominator = (out["payee_event_count"] - 1).clip(lower=1)
    out["payee_history_fraction"] = out["payee_event_index"] / denominator
    out["recipient_compromised"] = (
        out["payee_history_fraction"] >= out["compromise_fraction"]
    ).astype(int)

    scenarios: list[str] = []
    origins: list[str] = []
    for record in out.itertuples(index=False):
        key = str(record.transaction_id)
        profile = str(record.recipient_profile)
        source_fraud = int(record.base_is_fraud) == 1
        compromised = int(record.recipient_compromised) == 1

        # Account takeover is a payer-side event and can therefore target an
        # otherwise legitimate recipient without changing that recipient's identity.
        payer_ato_prone = _stable_unit_interval(str(record.payer_id), seed, "payer_ato") < 0.045
        synthetic_payer_attack = (
            profile not in FRAUD_PROFILES
            and payer_ato_prone
            and _stable_unit_interval(key, seed, "synthetic_payer_attack") < synthetic_fraud_rate
        )

        if source_fraud:
            if profile in FRAUD_PROFILES and compromised:
                scenario = _fraud_scenario(profile, key, seed)
            else:
                scenario = "account_takeover"
            origin = "source_label"
        elif synthetic_payer_attack:
            scenario = _stable_choice(
                key,
                seed,
                "payer_attack_type",
                ["account_takeover", "remote_access_scam"],
                [0.72, 0.28],
            )
            origin = "synthetic_payer_attack"
        elif profile in FRAUD_PROFILES and compromised:
            post_activation_probability = min(0.88, 0.60 + synthetic_fraud_rate * 4.0)
            if _stable_unit_interval(key, seed, "post_activation_fraud") < post_activation_probability:
                scenario = _fraud_scenario(profile, key, seed)
                origin = "synthetic_recipient_activation"
            else:
                scenario = _legitimate_scenario(profile, key, seed)
                origin = "fraud_profile_benign_cover"
        else:
            scenario = _legitimate_scenario(profile, key, seed)
            origin = "stable_recipient_legitimate"

        scenarios.append(scenario)
        origins.append(origin)

    out["scenario"] = scenarios
    out["label_origin"] = origins
    return out.sort_values("timestamp").reset_index(drop=True)


def recipient_consistency_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise whether recipient identities remain internally consistent."""
    return (
        frame.groupby("payee_id", observed=True)
        .agg(
            recipient_profile=("recipient_profile", "nunique"),
            recipient_archetype=("recipient_archetype", "nunique"),
            merchant_category=("merchant_category", "nunique"),
            scenario_count=("scenario", "nunique"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
        )
        .reset_index()
    )
