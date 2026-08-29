"""Synthetic request-log generator for a ticketing site during an onsale.

Emits request-level events for three actor classes: humans, naive bots, and
advanced bots. The advanced class is deliberately hard -- it spoofs the TLS
fingerprint, rides residential proxies, and jitters its timing -- so that the
detector's failure modes are visible rather than hidden behind an easy win.

Ground-truth labels are attached for evaluation only. Nothing downstream of
features.py is allowed to read them.
"""

from __future__ import annotations

import hashlib
import ipaddress
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# --- site model -------------------------------------------------------------

# The funnel a real visitor walks. Index position matters: skipping stages is
# one of the cheaper signals available to us.
FUNNEL = [
    "/",
    "/event/9182",
    "/event/9182/seats",
    "/cart",
    "/checkout",
    "/checkout/payment",
]
FUNNEL_INDEX = {path: i for i, path in enumerate(FUNNEL)}

# Pages humans wander into and bots have no reason to touch.
BROWSE_PAGES = ["/faq", "/venue/info", "/event/9182/reviews", "/help/delivery", "/account"]

# TLS stacks, and the browser each one is *actually* consistent with. A client
# presenting go-tls ciphers while claiming Chrome in its User-Agent is lying at
# one of the two layers, and that inconsistency is the signal -- not either
# field on its own.
TLS_STACKS = {
    "chrome-120": "Chrome",
    "chrome-121": "Chrome",
    "firefox-119": "Firefox",
    "safari-17": "Safari",
    "go-tls": None,
    "python-requests": None,
}

USER_AGENTS = {
    "Chrome": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Firefox": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) Gecko/20100101 Firefox/119.0",
    "Safari": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    ),
    "python-requests": "python-requests/2.31.0",
}


def _hash(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


def _ip(rng: np.random.Generator, kind: str) -> str:
    """Draw an address from a block that plausibly belongs to `kind`."""
    if kind == "datacenter":
        base = rng.choice([0x0D_00_00_00, 0x2D_20_00_00, 0x9F_2D_00_00])
    elif kind == "residential":
        base = rng.choice([0x4A_10_00_00, 0x57_80_00_00, 0xB0_40_00_00])
    else:  # mobile
        base = rng.choice([0x1F_60_00_00, 0x68_A0_00_00])
    return str(ipaddress.IPv4Address(int(base) + int(rng.integers(0, 2**16))))


@dataclass
class Event:
    ts: float
    session_id: str
    ip: str
    asn_type: str
    user_agent: str
    ua_claimed_browser: str
    ja3_hash: str
    tls_stack: str
    header_order_hash: str
    path: str
    method: str
    status: int
    has_mouse_telemetry: bool
    referer: str
    # ground truth -- evaluation only
    actor_class: str


# --- actor behaviours -------------------------------------------------------


def _human_session(rng: np.random.Generator, sid: str, t0: float) -> list[Event]:
    """A person: heavy-tailed think times, backtracking, browsing detours."""
    browser = rng.choice(["Chrome", "Firefox", "Safari"], p=[0.68, 0.20, 0.12])
    stack = {"Chrome": rng.choice(["chrome-120", "chrome-121"]), "Firefox": "firefox-119", "Safari": "safari-17"}[
        browser
    ]
    asn = rng.choice(["residential", "mobile"], p=[0.75, 0.25])
    ip = _ip(rng, asn)
    ja3 = _hash(stack, browser)
    hdr = _hash("hdr", browser)

    # Arrival is bimodal. A quarter of the audience is camped on the page
    # refreshing into the drop and lands in a tight spike at T-0; the rest
    # trickle in with a long right tail.
    #
    # The eager group is the reason this problem is hard. They are
    # indistinguishable from a bot pool on launch timing alone, so any rule
    # sharp enough to catch synchronised bots also blocks the most committed
    # fans -- who are exactly the customers you can least afford to block.
    if rng.random() < 0.25:
        t = t0 + float(rng.normal(0.0, 1.5))
    else:
        t = t0 + float(rng.exponential(90.0)) - 20.0
    events: list[Event] = []
    prev = ""
    # Real visitors do not all arrive at the homepage -- most follow a deep
    # link from an email, a social post, or search straight into the event
    # page. Without this spread, "entered below stage 0" would perfectly
    # identify every bot and the whole problem would collapse.
    stage = int(rng.choice([0, 1, 2], p=[0.35, 0.50, 0.15]))
    # Not everyone completes: most bail somewhere in the funnel.
    depth = int(np.clip(rng.normal(3.4, 1.6), 1, len(FUNNEL)))
    depth = max(depth, stage + 1)

    while stage < depth:
        path = FUNNEL[stage]
        # Detour into a browse page now and then.
        if rng.random() < 0.22:
            path = str(rng.choice(BROWSE_PAGES))
        # Occasional back-navigation -- bots essentially never do this.
        elif stage > 0 and rng.random() < 0.15:
            path = FUNNEL[stage - 1]

        status = 200
        if path.startswith("/checkout"):
            # Everyone queues at an onsale. Deliberately close to the advanced
            # bot's distribution: if humans sailed through while bots got
            # throttled, `error_rate` would be a proxy for a decision the
            # defence had already made, and the model would just relearn it.
            status = int(rng.choice([200, 429, 403], p=[0.55, 0.42, 0.03]))

        events.append(
            Event(
                ts=t,
                session_id=sid,
                ip=ip,
                asn_type=asn,
                user_agent=USER_AGENTS[browser],
                ua_claimed_browser=browser,
                ja3_hash=ja3,
                tls_stack=stack,
                header_order_hash=hdr,
                path=path,
                method="POST" if path.endswith("payment") else "GET",
                status=status,
                # Real browsers run our client-side beacon.
                has_mouse_telemetry=bool(rng.random() < 0.94),
                referer=prev,
                actor_class="human",
            )
        )
        prev = path
        if path == FUNNEL[stage]:
            stage += 1
        # Lognormal think time: mostly a few seconds, sometimes a long pause.
        t += float(rng.lognormal(1.2, 0.9))

    return events


def _naive_bot_session(rng: np.random.Generator, sid: str, t0: float) -> list[Event]:
    """Straight requests loop. Datacenter IP, honest UA, metronomic."""
    ip = _ip(rng, "datacenter")
    stack = "python-requests"
    ja3 = _hash(stack, "none")
    hdr = _hash("hdr", "requests")

    t = t0 + float(rng.normal(0.4, 0.15))
    events: list[Event] = []
    # Skips the funnel: goes for seats and checkout directly, then retries.
    plan = ["/event/9182/seats", "/cart", "/checkout", "/checkout/payment"]
    n_retries = int(rng.integers(3, 12))

    for i in range(n_retries):
        for path in plan:
            status = 200
            if path.startswith("/checkout"):
                status = int(rng.choice([200, 429, 403], p=[0.35, 0.45, 0.20]))
            events.append(
                Event(
                    ts=t,
                    session_id=sid,
                    ip=ip,
                    asn_type="datacenter",
                    user_agent=USER_AGENTS["python-requests"],
                    ua_claimed_browser="python-requests",
                    ja3_hash=ja3,
                    tls_stack=stack,
                    header_order_hash=hdr,
                    path=path,
                    method="POST" if path.endswith("payment") else "GET",
                    status=status,
                    has_mouse_telemetry=False,
                    referer="",
                    actor_class="naive_bot",
                )
            )
            # Fixed sleep with negligible jitter.
            t += 0.20 + float(rng.normal(0, 0.02))

    return events


def _advanced_bot_session(
    rng: np.random.Generator, sid: str, t0: float, pool_launch: float
) -> list[Event]:
    """Spoofed TLS, residential proxy, jittered timing, warm-up requests.

    This is the class that survives the obvious rules. Its fingerprint is
    computed exactly as a real Chrome session's is, so it lands in the same
    JA3 bucket as thousands of genuine visitors -- there is no small
    give-away cluster to key on.

    What it still cannot cheaply fake: a human-shaped think-time
    *distribution* (it jitters uniformly; humans are lognormal), full
    client-beacon coverage, and independence from its own pool -- every
    session fires off the same NTP-aligned clock.
    """
    browser = "Chrome"
    stack = str(rng.choice(["chrome-120", "chrome-121"]))  # spoofed to match the UA
    ip = _ip(rng, "residential")
    hdr = _hash("hdr", browser)  # header order replicated correctly
    # Identical construction to the human path: the pool blends in.
    ja3 = _hash(stack, browser)

    # NTP-aligned launch: sessions in a pool are suspiciously synchronised.
    t = pool_launch + float(rng.normal(0, 0.05))
    events: list[Event] = []

    # Warm-up: touch a couple of funnel pages so the path graph looks plausible.
    plan = ["/", "/event/9182", "/event/9182/seats", "/cart", "/checkout", "/checkout/payment"]
    if rng.random() < 0.35:
        plan = plan[2:]  # some skip the warm-up to shave latency

    # One clean pass. A pooled operator does NOT loop inside a session -- that
    # would leave an obvious repeat signature. Parallelism lives across
    # sessions instead, which is the whole reason it runs a pool at all. Only
    # the payment POST is retried, and only on a soft failure.
    prev = ""
    for path in plan:
        status = 200
        if path.startswith("/checkout"):
            status = int(rng.choice([200, 429, 403], p=[0.55, 0.35, 0.10]))
        attempts = 2 if (path.endswith("payment") and status == 429) else 1

        for _ in range(attempts):
            events.append(
                Event(
                    ts=t,
                    session_id=sid,
                    ip=ip,
                    asn_type="residential",
                    user_agent=USER_AGENTS[browser],
                    ua_claimed_browser=browser,
                    ja3_hash=ja3,  # indistinguishable from real Chrome
                    tls_stack=stack,
                    header_order_hash=hdr,
                    path=path,
                    method="POST" if path.endswith("payment") else "GET",
                    status=status,
                    # A patched headless browser fires the beacon most of the
                    # time. Close enough to human that this alone won't do it.
                    has_mouse_telemetry=bool(rng.random() < 0.80),
                    referer=prev,
                    actor_class="advanced_bot",
                )
            )
            # Uniform jitter: looks random, but the coefficient of variation is
            # far too low next to a human's lognormal think time.
            t += float(rng.uniform(0.4, 1.5))
        prev = path

    return events


# --- driver -----------------------------------------------------------------


def generate(
    n_humans: int = 4000,
    n_naive: int = 40,
    n_advanced: int = 60,
    onsale_ts: float = 1_760_000_000.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Build one onsale's worth of traffic.

    Defaults give roughly a 2% bot session rate -- low enough that accuracy is
    a useless metric, which is the point.
    """
    rng = np.random.default_rng(seed)
    events: list[Event] = []

    for i in range(n_humans):
        events += _human_session(rng, f"h{i:06d}", onsale_ts)

    for i in range(n_naive):
        events += _naive_bot_session(rng, f"n{i:06d}", onsale_ts)

    # Advanced bots run in pools -- one operator, many parallel sessions fired
    # off a single NTP-aligned clock. The fingerprint blends in; the timing
    # coincidence is what is left to find.
    n_pools = max(1, n_advanced // 12)
    pool_launches = [onsale_ts + float(rng.normal(0.3, 0.1)) for _ in range(n_pools)]
    for i in range(n_advanced):
        events += _advanced_bot_session(rng, f"a{i:06d}", onsale_ts, pool_launches[i % n_pools])

    df = pd.DataFrame([asdict(e) for e in events])
    df = df.sort_values("ts").reset_index(drop=True)
    df["onsale_ts"] = onsale_ts
    return df


if __name__ == "__main__":
    frame = generate()
    print(frame.groupby("actor_class").agg(requests=("ts", "size"), sessions=("session_id", "nunique")))
    print(f"\ntotal requests: {len(frame):,}")
