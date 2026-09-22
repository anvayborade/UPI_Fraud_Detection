from __future__ import annotations


def integrity_risk(
    play_integrity_ok: bool | None,
    device_known: bool,
    app_registration_age_days: int,
    overlay_indicator: bool | None,
    remote_access_indicator: bool | None,
) -> float:
    risk = 0.0
    if play_integrity_ok is False:
        risk += 0.35
    if not device_known:
        risk += 0.20
    if app_registration_age_days <= 2:
        risk += 0.15
    if overlay_indicator:
        risk += 0.15
    if remote_access_indicator:
        risk += 0.25
    return max(0.0, min(1.0, risk))
