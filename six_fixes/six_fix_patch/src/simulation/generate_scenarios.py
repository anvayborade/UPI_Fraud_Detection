from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    probability: float
    fraud: int
    legitimate_novelty: int


SCENARIOS = [
    ScenarioSpec("familiar_legitimate", 0.58, 0, 0),
    ScenarioSpec("auto_long_distance_legitimate", 0.065, 0, 1),
    ScenarioSpec("hospital_emergency_legitimate", 0.035, 0, 1),
    ScenarioSpec("travel_hotel_legitimate", 0.035, 0, 1),
    ScenarioSpec("gig_worker_payment_legitimate", 0.035, 0, 1),
    ScenarioSpec("new_merchant_legitimate", 0.035, 0, 1),
    ScenarioSpec("legitimate_split_payment", 0.025, 0, 1),
    ScenarioSpec("fake_refund_qr", 0.055, 1, 0),
    ScenarioSpec("collect_request_scam", 0.045, 1, 0),
    ScenarioSpec("account_takeover", 0.035, 1, 0),
    ScenarioSpec("mule_directed", 0.035, 1, 0),
    ScenarioSpec("remote_access_scam", 0.02, 1, 0),
]


def _clip01(values: np.ndarray | float) -> np.ndarray | float:
    return np.clip(values, 0.0, 1.0)


def _assign_archetypes(
    df: pd.DataFrame,
    scenario: np.ndarray,
    masks: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign overlapping recipient types rather than copying the fraud label."""
    archetype = np.full(len(df), "personal", dtype=object)

    banksim = df.get("source_dataset", pd.Series("unknown", index=df.index)).astype(str).eq("banksim").to_numpy()
    bank_merchant = banksim & (rng.random(len(df)) < 0.70)
    archetype[bank_merchant] = "merchant"

    archetype[np.isin(scenario, ["auto_long_distance_legitimate", "gig_worker_payment_legitimate"])] = "transport_provider"
    archetype[np.isin(scenario, ["hospital_emergency_legitimate", "travel_hotel_legitimate", "new_merchant_legitimate"])] = "merchant"

    # Fraud recipients are often, but not always, mules. Some look like personal
    # users or merchants, which prevents recipient type from becoming the answer.
    mule_probability = {
        "mule_directed": 0.76,
        "fake_refund_qr": 0.52,
        "collect_request_scam": 0.46,
        "remote_access_scam": 0.50,
        "account_takeover": 0.34,
    }
    for name, probability in mule_probability.items():
        mask = masks[name]
        selected = mask & (rng.random(len(df)) < probability)
        archetype[selected] = "money_mule"

    # Difficult legitimate hubs: a small number of genuine merchants have very
    # fast settlement behaviour and resemble a mule superficially.
    legitimate = df["is_fraud"].to_numpy() == 0
    difficult_hubs = legitimate & (rng.random(len(df)) < 0.025)
    archetype[difficult_hubs] = rng.choice(
        ["merchant", "transport_provider"], size=difficult_hubs.sum()
    )
    return archetype


def inject_upi_scenarios(base: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add noisy, overlapping UPI context and fraud labels.

    Labels still describe the intended research scenarios, but no generated input
    feature deterministically reproduces those labels. Graph-flow values are only
    placeholders here and are replaced later with prior-only historical features.
    """
    rng = np.random.default_rng(seed)
    df = base.copy().sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    names = [s.name for s in SCENARIOS]
    probs = np.asarray([s.probability for s in SCENARIOS], dtype=float)
    probs /= probs.sum()
    if "scenario" in df.columns and df["scenario"].notna().all():
        invalid = sorted(set(df["scenario"].astype(str)).difference(names))
        if invalid:
            raise ValueError(f"Unsupported preassigned scenarios: {invalid}")
        scenario = df["scenario"].astype(str).to_numpy()
    else:
        scenario = rng.choice(names, size=n, p=probs)

    spec_map = {s.name: s for s in SCENARIOS}
    masks = {name: scenario == name for name in names}
    df["scenario"] = scenario
    df["is_fraud"] = np.asarray([spec_map[x].fraud for x in scenario], dtype=int)
    df["is_legitimate_novelty"] = np.asarray(
        [spec_map[x].legitimate_novelty for x in scenario], dtype=int
    )

    payer_codes = pd.factorize(df["payer_id"])[0]
    payee_codes = pd.factorize(df["payee_id"])[0]
    df["device_id"] = [f"D{v:06d}" for v in payer_codes]
    df["vpa_id"] = [f"vpa{v:06d}@upi" for v in payee_codes]
    df["ip_asn"] = [f"AS{1000 + (v % 250)}" for v in payer_codes]
    df["qr_id"] = [f"QR{v:06d}" for v in payee_codes]

    # Coarse location and journey context with broad overlap.
    home_lat = 19.076 + ((payer_codes % 200) - 100) * 0.0009
    home_lon = 72.878 + (((payer_codes // 200) % 200) - 100) * 0.0009
    distance_km = np.abs(rng.normal(4.0, 8.0, n))
    travel_mask = np.isin(
        scenario,
        ["auto_long_distance_legitimate", "travel_hotel_legitimate", "hospital_emergency_legitimate"],
    )
    distance_km[travel_mask] = np.abs(rng.normal(38, 30, travel_mask.sum())).clip(3, 1800)
    suspicious_remote = np.isin(scenario, ["account_takeover", "remote_access_scam"])
    distance_km[suspicious_remote] = np.abs(
        rng.normal(70, 150, suspicious_remote.sum())
    ).clip(0, 1800)
    angle = rng.uniform(0, 2 * math.pi, n)
    lat = home_lat + (distance_km / 111.0) * np.cos(angle)
    lon = home_lon + (
        distance_km / (111.0 * np.cos(np.deg2rad(np.maximum(np.abs(home_lat), 1))))
    ) * np.sin(angle)
    df["latitude"] = lat
    df["longitude"] = lon
    df["home_latitude"] = home_lat
    df["home_longitude"] = home_lon
    df["distance_from_usual_km"] = distance_km

    df["payment_mode"] = rng.choice(["P2P", "P2M", "QR"], n, p=[0.46, 0.25, 0.29])
    qr_scenarios = np.isin(
        scenario,
        ["fake_refund_qr", "auto_long_distance_legitimate", "gig_worker_payment_legitimate"],
    )
    df.loc[qr_scenarios, "payment_mode"] = "QR"
    # Some genuine collect requests and some social scams use ordinary push/QR.
    df.loc[masks["collect_request_scam"] & (rng.random(n) < 0.78), "payment_mode"] = "COLLECT"
    genuine_collect = (df["is_fraud"].to_numpy() == 0) & (rng.random(n) < 0.015)
    df.loc[genuine_collect, "payment_mode"] = "COLLECT"
    df["collect_request"] = (df["payment_mode"] == "COLLECT").astype(int)

    if "recipient_merchant_category" in df.columns:
        df["merchant_category"] = df["recipient_merchant_category"].astype(str)
    else:
        df["merchant_category"] = rng.choice(
            ["PERSONAL", "GROCERY", "TRANSPORT", "HEALTHCARE", "HOTEL", "RESTAURANT", "SERVICES"],
            n,
        )

    amount = df["amount"].to_numpy(float)
    amount[masks["auto_long_distance_legitimate"]] = rng.uniform(350, 2200, masks["auto_long_distance_legitimate"].sum())
    amount[masks["hospital_emergency_legitimate"]] = np.exp(
        rng.normal(np.log(15_000), 1.0, masks["hospital_emergency_legitimate"].sum())
    ).clip(1200, 180_000)
    amount[masks["travel_hotel_legitimate"]] = rng.uniform(1400, 38_000, masks["travel_hotel_legitimate"].sum())
    amount[masks["legitimate_split_payment"]] = rng.uniform(1500, 24_000, masks["legitimate_split_payment"].sum())
    amount[masks["fake_refund_qr"]] = rng.uniform(1400, 58_000, masks["fake_refund_qr"].sum())
    amount[masks["collect_request_scam"]] = rng.uniform(800, 42_000, masks["collect_request_scam"].sum())
    amount[masks["account_takeover"]] = rng.uniform(3500, 125_000, masks["account_takeover"].sum())
    amount[masks["mule_directed"]] = rng.uniform(1800, 85_000, masks["mule_directed"].sum())
    amount[masks["remote_access_scam"]] = rng.uniform(2500, 95_000, masks["remote_access_scam"].sum())
    df["amount"] = np.round(amount, 2)

    # New payee and device signals remain probabilistic in both classes.
    df["is_new_payee"] = rng.binomial(1, 0.18, n)
    novelty_mask = df["is_legitimate_novelty"].to_numpy() == 1
    fraud_mask = df["is_fraud"].to_numpy() == 1
    df.loc[novelty_mask, "is_new_payee"] = rng.binomial(1, 0.82, novelty_mask.sum())
    df.loc[fraud_mask, "is_new_payee"] = rng.binomial(1, 0.67, fraud_mask.sum())

    df["device_age_days"] = rng.integers(3, 1500, n)
    df["sim_age_days"] = rng.integers(5, 2200, n)
    df["app_registration_age_days"] = rng.integers(2, 1000, n)
    df["device_known"] = rng.binomial(1, 0.94, n)
    ato_like = masks["account_takeover"] | masks["remote_access_scam"]
    df.loc[ato_like, "device_age_days"] = rng.integers(0, 45, ato_like.sum())
    df.loc[ato_like, "sim_age_days"] = rng.integers(0, 90, ato_like.sum())
    df.loc[ato_like, "app_registration_age_days"] = rng.integers(0, 40, ato_like.sum())
    df.loc[ato_like, "device_known"] = rng.binomial(1, 0.34, ato_like.sum())

    df["recent_pin_reset"] = rng.binomial(1, 0.025, n)
    df["app_reregistered"] = rng.binomial(1, 0.018, n)
    df.loc[ato_like, "recent_pin_reset"] = rng.binomial(1, 0.63, ato_like.sum())
    df.loc[ato_like, "app_reregistered"] = rng.binomial(1, 0.55, ato_like.sum())
    genuine_phone_change = (~fraud_mask) & (rng.random(n) < 0.018)
    df.loc[genuine_phone_change, "device_known"] = 0
    df.loc[genuine_phone_change, "recent_pin_reset"] = rng.binomial(1, 0.45, genuine_phone_change.sum())
    df.loc[genuine_phone_change, "app_reregistered"] = rng.binomial(1, 0.55, genuine_phone_change.sum())

    df["remote_access_indicator"] = rng.binomial(1, 0.006, n)
    df["overlay_indicator"] = rng.binomial(1, 0.01, n)
    ra = masks["remote_access_scam"]
    df.loc[ra, "remote_access_indicator"] = rng.binomial(1, 0.73, ra.sum())
    df.loc[ra, "overlay_indicator"] = rng.binomial(1, 0.48, ra.sum())
    other_social = masks["fake_refund_qr"] | masks["collect_request_scam"]
    df.loc[other_social, "remote_access_indicator"] = rng.binomial(1, 0.08, other_social.sum())

    df["qr_verified"] = rng.binomial(1, 0.72, n)
    legitimate_qr = np.isin(
        scenario,
        ["auto_long_distance_legitimate", "hospital_emergency_legitimate", "travel_hotel_legitimate", "new_merchant_legitimate"],
    )
    df.loc[legitimate_qr, "qr_verified"] = rng.binomial(1, 0.84, legitimate_qr.sum())
    df.loc[masks["fake_refund_qr"], "qr_verified"] = rng.binomial(1, 0.31, masks["fake_refund_qr"].sum())

    # Recipient identity is assigned once per payee by recipient_profiles.py.
    # Do not overwrite it from the row-level scenario.
    if "recipient_archetype" not in df.columns:
        df["recipient_archetype"] = "personal"
    if "recipient_profile" not in df.columns:
        df["recipient_profile"] = df["recipient_archetype"].astype(str)
    mule_mask = df["recipient_archetype"].eq("money_mule").to_numpy()

    # Account age is stable per recipient and increases chronologically.
    unique_payees = df[["payee_id", "recipient_profile"]].drop_duplicates("payee_id")
    base_age: dict[str, float] = {}
    for record in unique_payees.itertuples(index=False):
        profile = str(record.recipient_profile)
        if profile == "new_business":
            value = float(np.exp(rng.normal(np.log(45), 0.65)))
        elif profile == "money_mule":
            value = float(np.exp(rng.normal(np.log(100), 1.0)))
        else:
            value = float(np.exp(rng.normal(np.log(420), 0.95)))
        base_age[str(record.payee_id)] = float(np.clip(value, 1, 3500))
    first_seen = df.groupby("payee_id", observed=True)["timestamp"].transform("min")
    elapsed_days = (pd.to_datetime(df["timestamp"], utc=True) - pd.to_datetime(first_seen, utc=True)).dt.total_seconds() / 86_400
    df["recipient_account_age_days"] = (
        df["payee_id"].astype(str).map(base_age).astype(float) + elapsed_days.clip(lower=0)
    )

    # Direct complaint score is only weak random prior reputation. Actual delayed
    # complaint intelligence is created later from train-period reports.
    df["recipient_complaint_score"] = _clip01(rng.beta(0.8, 12.0, n))

    # Placeholders are overwritten by add_prior_graph_flow_features(). Keeping
    # them here preserves the dataframe contract for notebooks and tests.
    df["rapid_outflow_ratio"] = _clip01(rng.beta(1.5, 5.5, n))
    df["median_holding_minutes"] = rng.lognormal(np.log(180), 1.1, n).clip(0.1, 20_000)
    df["fan_in_1h"] = rng.poisson(1.5, n)
    df["fan_out_1h"] = rng.poisson(1.0, n)

    df["payer_payee_proximity_km"] = np.abs(rng.normal(0.8, 2.0, n)).clip(0, 100)
    df.loc[ato_like, "payer_payee_proximity_km"] = np.abs(
        rng.normal(12, 35, ato_like.sum())
    ).clip(0, 1500)
    df["location_continuity"] = _clip01(rng.beta(6.0, 2.0, n))
    df.loc[ato_like, "location_continuity"] = _clip01(rng.beta(2.2, 3.5, ato_like.sum()))
    df["journey_plausibility"] = _clip01(
        0.60 * df["location_continuity"]
        + 0.40 * np.exp(-df["payer_payee_proximity_km"] / 15)
    )
    df["impossible_travel"] = (
        (df["distance_from_usual_km"] > 900) & (df["location_continuity"] < 0.20)
    ).astype(int)
    profile_consistency = {
        "personal": 0.55,
        "merchant": 0.88,
        "transport_provider": 0.90,
        "hospital": 0.94,
        "hotel": 0.92,
        "gig_worker": 0.82,
        "new_business": 0.72,
        "money_mule": 0.32,
        "qr_scam_recipient": 0.48,
        "collect_scam_recipient": 0.45,
        "remote_scam_recipient": 0.42,
        # A compromised merchant remains merchant-like, creating a realistic
        # contradictory case for the fusion model.
        "compromised_merchant": 0.86,
    }
    baseline_consistency = df["recipient_profile"].astype(str).map(profile_consistency).fillna(0.55).to_numpy(float)
    df["merchant_category_consistency"] = _clip01(
        baseline_consistency + rng.normal(0, 0.08, n)
    )

    for label in [
        "is_account_takeover",
        "is_social_engineering",
        "is_qr_deception",
        "is_collect_scam",
        "is_mule_directed",
        "is_transaction_splitting",
        "is_rapid_pass_through",
        "is_remote_access_fraud",
    ]:
        df[label] = 0
    df.loc[masks["account_takeover"], "is_account_takeover"] = 1
    df.loc[np.isin(scenario, ["fake_refund_qr", "collect_request_scam", "remote_access_scam"]), "is_social_engineering"] = 1
    df.loc[masks["fake_refund_qr"], "is_qr_deception"] = 1
    df.loc[masks["collect_request_scam"], "is_collect_scam"] = 1
    df.loc[masks["mule_directed"], "is_mule_directed"] = 1
    splitting_mask = masks["mule_directed"] & (rng.random(n) < 0.45)
    df.loc[splitting_mask, "is_transaction_splitting"] = 1
    df.loc[masks["mule_directed"], "is_rapid_pass_through"] = rng.binomial(1, 0.72, masks["mule_directed"].sum())
    df.loc[masks["remote_access_scam"], "is_remote_access_fraud"] = 1

    # These columns are retained for API compatibility but are not used as input
    # features. They will be replaced by actual trained model outputs in train_all.
    df["recipient_mule_score_cached"] = 0.0
    df["sequence_account_takeover_score_cached"] = 0.0
    df["sequence_social_engineering_score_cached"] = 0.0

    df["merchant_id"] = np.where(
        df["recipient_archetype"].isin(["merchant", "transport_provider", "gig_worker", "new_business"]),
        "M" + df["payee_id"].astype(str),
        None,
    )
    df["common_owner_verified"] = (df["payment_mode"] == "SELF_TRANSFER").astype(int)
    df["play_integrity_ok"] = rng.binomial(1, 0.985, n)
    df.loc[ato_like, "play_integrity_ok"] = rng.binomial(1, 0.58, ato_like.sum())
    return df
