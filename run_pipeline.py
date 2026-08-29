"""End-to-end run: generate traffic, build features, score three detectors.

    python run_pipeline.py
    python run_pipeline.py --advanced 200 --seed 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import evaluate
from src.detect import IsolationForestDetector, RuleBaseline, SupervisedUpperBound
from src.features import FEATURE_NOTES, build
from src.generate import generate

DATA = Path(__file__).parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--humans", type=int, default=4000)
    ap.add_argument("--naive", type=int, default=40, help="unsophisticated bot sessions")
    ap.add_argument("--advanced", type=int, default=60, help="spoofed-fingerprint bot sessions")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--review-capacity", type=int, default=100, help="sessions an analyst can review")
    ap.add_argument("--save", action="store_true", help="write events and features to data/")
    args = ap.parse_args()

    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 40)

    print("=" * 74)
    print("ADVERSARIAL BOT DETECTION -- ticketing onsale")
    print("=" * 74)

    events = generate(
        n_humans=args.humans, n_naive=args.naive, n_advanced=args.advanced, seed=args.seed
    )
    print("\n[1] traffic generated")
    summary = events.groupby("actor_class").agg(
        requests=("ts", "size"), sessions=("session_id", "nunique")
    )
    print(summary.to_string())

    X, y, meta = build(events)
    bot_rate = y.mean()
    print(f"\n[2] features built -- {X.shape[0]:,} sessions x {X.shape[1]} features")
    print(f"    bot session rate: {bot_rate:.2%}  "
          f"(predicting 'human' for everything scores {1 - bot_rate:.2%} accuracy)")

    print("\n    what the behavioural features are doing:")
    for feat, note in FEATURE_NOTES.items():
        print(f"      {feat:<20} {note}")

    print("\n    separation on the key features (mean by class):")
    key = [
        "iat_cv",
        "burstiness",
        "telemetry_frac",
        "launch_burst_ratio",
        "funnel_entry_stage",
        "start_offset_s",
    ]
    sep = X[key].copy()
    sep["class"] = meta["actor_class"].to_numpy()
    print(sep.groupby("class").mean().round(3).to_string())

    do_nothing = evaluate.baseline_cost(y)
    print(f"\n[3] detectors  (FP cost {evaluate.COST_FALSE_POSITIVE:.0f}, "
          f"FN cost {evaluate.COST_FALSE_NEGATIVE:.0f} per session)")
    print(f"    doing nothing costs {do_nothing:.0f}\n")

    results = []

    rules = RuleBaseline().fit(X)
    results.append(evaluate.report(rules.name, y, rules.score(X), meta, args.review_capacity))

    iso = IsolationForestDetector(contamination=0.02, seed=args.seed).fit(X)
    results.append(evaluate.report(iso.name, y, iso.score(X), meta, args.review_capacity))

    gb = SupervisedUpperBound(seed=args.seed).fit(X, y)
    X_te, y_te = gb.holdout()
    meta_te = meta.loc[X_te.index].reset_index(drop=True)
    results.append(
        evaluate.report(gb.name, y_te.reset_index(drop=True), gb.score(X_te), meta_te, args.review_capacity)
    )

    for r in results:
        # The supervised model is scored on its holdout, so its do-nothing
        # reference is proportionally smaller.
        ref = do_nothing if "boosting" not in r["detector"] else evaluate.baseline_cost(y_te)
        print(evaluate.format_report(r, ref))
        print()

    print("[4] what the labelled model leans on")
    print(gb.importances(X).head(8).round(4).to_string())

    print("\n" + "=" * 74)
    print("READ THIS BEFORE TRUSTING THE HEADLINE NUMBERS")
    print("=" * 74)
    print(
        """
The aggregate scores flatter the models. Check the per-tier recall: naive bots
are trivially caught and inflate every metric, while the advanced tier -- the
one that actually clears inventory -- is far harder and sometimes barely
detected at the cost-optimal threshold.

Three things to try from here:
  1. Raise --advanced and drop --naive. Watch the headline PR-AUC collapse once
     the easy tier stops carrying it.
  2. Change COST_FALSE_POSITIVE in src/evaluate.py. At a high enough ratio the
     cost-optimal move is to flag nobody, which is a real operational answer.
  3. Weaken the generator's tells one at a time in _advanced_bot_session --
     the telemetry rate, then the uniform jitter, then the synchronised launch
     -- to simulate an operator adapting to your blocks. That is the drift you
     would be fighting in production, and each removal costs you real recall.
"""
    )

    if args.save:
        DATA.mkdir(exist_ok=True)
        events.to_parquet(DATA / "events.parquet", index=False)
        feats = X.copy()
        feats["actor_class"] = meta["actor_class"]
        feats.to_parquet(DATA / "sessions.parquet", index=False)
        print(f"wrote {DATA / 'events.parquet'} and {DATA / 'sessions.parquet'}")


if __name__ == "__main__":
    main()
