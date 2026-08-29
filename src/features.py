"""Session-level feature engineering.

A single request carries almost no signal. Everything useful lives in the shape
of a session -- how its timing is distributed, whether it walked the funnel or
teleported into it -- and in how sessions relate to each other across a shared
fingerprint or ASN.

The label column is split out and returned separately so it cannot leak into
the feature matrix by accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .generate import BROWSE_PAGES, FUNNEL, FUNNEL_INDEX, TLS_STACKS

DATACENTER = "datacenter"


def _entropy(values: np.ndarray, bins: int = 8) -> float:
    """Shannon entropy of a binned distribution, in nats."""
    if values.size < 2:
        return 0.0
    counts, _ = np.histogram(values, bins=bins)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def _burstiness(iats: np.ndarray) -> float:
    """Goh-Barabasi burstiness: +1 bursty, 0 Poisson, -1 perfectly regular.

    A metronomic retry loop sits near -1; human think times sit above 0.
    """
    if iats.size < 2:
        return 0.0
    mu, sd = iats.mean(), iats.std()
    if mu + sd == 0:
        return 0.0
    return float((sd - mu) / (sd + mu))


def _session_features(g: pd.DataFrame) -> dict:
    ts = g["ts"].to_numpy()
    iats = np.diff(ts)
    paths = g["path"].tolist()
    n = len(g)

    # --- funnel traversal ---
    funnel_hits = [FUNNEL_INDEX[p] for p in paths if p in FUNNEL_INDEX]
    # A "skip" is any forward jump of more than one stage mid-session.
    skips = sum(1 for a, b in zip(funnel_hits, funnel_hits[1:]) if b - a > 1)
    # The bigger tell is where they *entered*. A visitor who materialises at
    # /cart never loaded the pages that issue a seat token; a client that
    # starts at / walked in the front door.
    entry_stage = funnel_hits[0] if funnel_hits else -1
    max_stage = max(funnel_hits) if funnel_hits else -1
    # How many requests before the first checkout touch. Bots go near-straight.
    checkout_positions = [i for i, p in enumerate(paths) if p.startswith("/checkout")]
    requests_before_checkout = checkout_positions[0] if checkout_positions else n

    # --- layer consistency ---
    # The TLS stack implies a browser family. If the UA claims something else,
    # one of the two layers is lying. Spoofed stacks report as consistent, so
    # this catches naive clients only -- by design.
    stack = g["tls_stack"].iloc[0]
    implied = TLS_STACKS.get(stack)
    claimed = g["ua_claimed_browser"].iloc[0]
    ua_tls_consistent = float(implied is not None and implied == claimed)

    onsale = float(g["onsale_ts"].iloc[0])

    return {
        "session_id": g["session_id"].iloc[0],
        # volume / pacing
        "n_requests": n,
        "duration_s": float(ts[-1] - ts[0]),
        "requests_per_min": n / max((ts[-1] - ts[0]) / 60.0, 1e-3),
        # inter-arrival shape -- the core behavioural block
        "iat_mean": float(iats.mean()) if iats.size else 0.0,
        "iat_std": float(iats.std()) if iats.size else 0.0,
        # Coefficient of variation separates uniform jitter (low CV) from
        # lognormal human think time (high CV) even when both look "random".
        "iat_cv": float(iats.std() / iats.mean()) if iats.size and iats.mean() > 0 else 0.0,
        "iat_min": float(iats.min()) if iats.size else 0.0,
        "iat_entropy": _entropy(iats),
        "burstiness": _burstiness(iats),
        # navigation graph
        "unique_paths": g["path"].nunique(),
        "path_repeat_ratio": 1.0 - g["path"].nunique() / n,
        "funnel_skips": skips,
        "funnel_entry_stage": entry_stage,
        "max_funnel_stage": max_stage,
        "browse_page_frac": float(np.mean([p in BROWSE_PAGES for p in paths])),
        "requests_before_checkout": requests_before_checkout,
        "empty_referer_frac": float((g["referer"] == "").mean()),
        # client attestation
        "telemetry_frac": float(g["has_mouse_telemetry"].mean()),
        "ua_tls_consistent": ua_tls_consistent,
        "is_datacenter": float(g["asn_type"].iloc[0] == DATACENTER),
        # server-side outcomes
        "error_rate": float(g["status"].isin([403, 429]).mean()),
        # onsale timing
        "start_offset_s": float(ts[0] - onsale),
        # carried for pool joins and evaluation, dropped before modelling
        "_ja3": g["ja3_hash"].iloc[0],
        "_asn_type": g["asn_type"].iloc[0],
        "_label": g["actor_class"].iloc[0],
    }


BURST_WINDOW_S = 0.25


def _pool_features(sessions: pd.DataFrame) -> pd.DataFrame:
    """Cross-session features: local launch synchronisation.

    An operator running parallel attempts is invisible one session at a time.
    In aggregate the tell is a burst of sessions starting within milliseconds
    of each other off a single NTP-aligned clock.

    Note what we deliberately do *not* use: the raw size of a fingerprint
    bucket. Millions of real visitors share a Chrome JA3, and a competent
    spoofer lands in that same bucket, so bucket size is not evidence of
    anything. Instead we ask a local question -- given how busy this
    fingerprint is overall, are there far more sessions starting right *here*
    than the background rate would predict?

    Eager humans refreshing at T-0 also raise this score. That overlap is the
    whole difficulty of the problem and is not an artifact to be tuned away.
    """
    out = np.zeros(len(sessions), dtype=float)

    for _, idx in sessions.groupby("_ja3").groups.items():
        pos = sessions.index.get_indexer(idx)
        starts = np.sort(sessions.loc[idx, "start_offset_s"].to_numpy())
        order = np.argsort(sessions.loc[idx, "start_offset_s"].to_numpy())
        n = starts.size
        span = float(starts[-1] - starts[0])
        if n < 2 or span <= 0:
            continue

        # Neighbours within +/- BURST_WINDOW_S, via the sorted array.
        left = np.searchsorted(starts, starts - BURST_WINDOW_S, side="left")
        right = np.searchsorted(starts, starts + BURST_WINDOW_S, side="right")
        observed = (right - left).astype(float)

        # What a uniform spread over this fingerprint's active span would give.
        expected = max(n * (2 * BURST_WINDOW_S) / span, 1e-6)

        ratio = observed / expected
        out[pos[order]] = ratio

    return pd.DataFrame({"launch_burst_ratio": out}, index=sessions.index)


def build(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Aggregate raw events into a modelling matrix.

    Returns (X, y, meta) where X holds only model-visible features, y is the
    binary bot label, and meta carries the label detail and pool keys used for
    per-tier evaluation.
    """
    rows = [_session_features(g) for _, g in events.groupby("session_id", sort=False)]
    sessions = pd.DataFrame(rows)

    sessions = pd.concat([sessions, _pool_features(sessions)], axis=1)

    meta = sessions[["session_id", "_label", "_ja3", "_asn_type"]].rename(
        columns={"_label": "actor_class", "_ja3": "ja3_hash", "_asn_type": "asn_type"}
    )
    y = (meta["actor_class"] != "human").astype(int)

    drop = ["session_id", "_ja3", "_asn_type", "_label"]
    X = sessions.drop(columns=drop)
    return X.reset_index(drop=True), y.reset_index(drop=True), meta.reset_index(drop=True)


FEATURE_NOTES = {
    "iat_cv": "uniform jitter reads low, human lognormal think time reads high",
    "burstiness": "-1 metronomic, 0 Poisson, +1 bursty",
    "telemetry_frac": "client beacon fired; headless clients fake it unreliably",
    "ua_tls_consistent": "0 when the TLS stack contradicts the User-Agent",
    "launch_burst_ratio": "sessions starting far above this fingerprint's background rate",
    "funnel_entry_stage": "0 = walked in the front door, 3+ = materialised at the cart",
}
