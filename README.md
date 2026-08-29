# Adversarial bot detection on ticketing traffic

Detecting automated purchasing on a ticketing site during an onsale — a
supervised/unsupervised comparison on synthetic request logs, built around the
part that actually matters: the sophisticated tier that defeats the obvious
rules.

This is a **defensive** project. It generates synthetic traffic and classifies
it. There is no client here that talks to any real site.

## The problem

Not "clean the bad rows". The adversary is *adaptive* — it reads your blocks
and changes. Three properties make it different from ordinary data quality
work:

| | data cleaning | this |
|---|---|---|
| bad records | errors, not hiding | engineered to look legitimate |
| class balance | usually fine | ~2% positives; 97.6% accuracy by predicting "human" |
| labels | knowable | absent on live traffic; proxies are weak |
| stability | fixed once | decays because someone is adapting to you |

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

python run_pipeline.py
python -m pytest tests/ -q
```

## What it does

**`src/generate.py`** — synthesises request logs for one onsale. Three actor
classes:

- **human** — bimodal arrival (a quarter camped on the page refreshing into
  T-0, the rest trickling in), lognormal think times, backtracking, browse
  detours, deep-link entry, client beacon fires ~94% of the time.
- **naive_bot** — `python-requests` off a datacenter IP, honest User-Agent,
  metronomic 200 ms loop, enters at the seat map. Caught by rules alone.
- **advanced_bot** — TLS fingerprint computed *identically to real Chrome*, so
  it lands in the same JA3 bucket as thousands of genuine visitors. Residential
  proxy, correct header order, uniform timing jitter, beacon fires 80% of the
  time, one clean pass through the funnel per session with parallelism across
  sessions. Launches off a shared NTP-aligned clock.

**`src/features.py`** — aggregates requests into session vectors. Behaviour,
not identity:

- `iat_cv` — coefficient of variation of inter-arrival times. Uniform jitter
  reads low; human lognormal think time reads high. Both look "random".
- `burstiness` — Goh–Barabási, −1 metronomic to +1 bursty.
- `iat_entropy`, `telemetry_frac`, `ua_tls_consistent`, `funnel_entry_stage`,
  `empty_referer_frac`, `path_repeat_ratio`, …
- `launch_burst_ratio` — the one cross-session feature: given how busy this
  fingerprint is overall, are far more sessions starting *right here* than the
  background rate predicts?

**`src/detect.py`** — three detectors, weakest to strongest: a hand-written
rule baseline, an isolation forest (unsupervised — the realistic deployable
setting), and gradient boosting on ground-truth labels (a cheating upper bound
that answers "is the signal in the features at all?").

**`src/evaluate.py`** — PR-AUC, precision@k sized to a review queue,
cost-weighted threshold sweep, and per-tier recall.

## The result

Default run — 4,000 human sessions, 40 naive bots, 60 advanced bots:

| detector | PR-AUC | naive recall | **advanced recall** | human FPR |
|---|---|---|---|---|
| rules (datacenter / UA-TLS mismatch) | 0.42 | 1.00 | **0.00** | 0.0000 |
| isolation forest (unsupervised) | 0.61 | 0.82 | **0.00** | 0.0000 |
| gradient boosting (labelled) | 1.00 | 1.00 | **0.94** | 0.0000 |

The headline metrics are flattered by the naive tier. Read the third column
instead: **the deployable unsupervised model catches none of the sophisticated
bots**, while the labelled model catches 94% of them from the same features.

That gap is the finding. The features contain the signal; unsupervised outlier
ranking cannot extract it, because a sophisticated bot session is not an
*outlier* — it was built specifically to sit inside the human distribution. It
is only anomalous *relative to a label you don't have*. The practical
consequence is that the money goes into label acquisition — chargebacks,
no-shows, resale-listing matchbacks, manual review — not into a better
unsupervised model.

## Design notes: four leaks, found and fixed

Every one of these produced a beautiful, meaningless ~1.000 before being
caught. They are the actual lesson of the project, so they're documented rather
than quietly fixed.

1. **Private fingerprint bucket.** Bot pools were given their own spoofed JA3,
   so "small JA3 bucket" *was* the label — 63% of model importance. Real
   spoofers collide with mainstream Chrome. Fixed by computing the bot's
   fingerprint through the identical code path as a human's.
2. **In-session retry loops.** Advanced bots replayed their whole path plan 2–5
   times, making `path_repeat_ratio` a giveaway. A pooled operator doesn't do
   that — that's the entire reason it runs 100 parallel sessions. Fixed to one
   clean pass.
3. **Circular error rates.** Humans got 429'd 10% of the time and bots 45%,
   which assumes the defence already works — the model just relearned a
   decision the server had already made. Fixed by making everyone queue.
4. **Uniform funnel entry.** Every human started at `/`, so "entered deeper
   than stage 0" identified every bot. Real visitors arrive on deep links from
   email, social, and search. Fixed with a realistic entry distribution.

A regression test (`test_no_single_feature_separates_the_advanced_tier`) now
fails the build if any single feature cleanly splits advanced bots from humans.
It caught leak-adjacent behaviour in `launch_burst_ratio` after the first three
were fixed, which is what prompted adding the T-0 human refresh spike.

## Why false positives dominate

`COST_FALSE_POSITIVE = 50` against `COST_FALSE_NEGATIVE = 5`. Blocking a real
customer costs roughly ten times what letting a bot through costs: they don't
come back, they complain publicly, and support has to unpick it. That ratio is
a business input — it cannot be learned from the data, and it drives everything
downstream of the model.

It also means *doing nothing* is a real baseline with a real cost, and a
detector can easily score worse than it. The threshold sweep always includes
"flag nobody" as a candidate for exactly this reason.

The 25% of humans who camp on the page refreshing into T-0 are the crux: on
launch timing alone they are indistinguishable from a synchronised bot pool.
Any rule sharp enough to catch the pool also blocks the most committed fans —
the customers you can least afford to block.

## Things to try

```bash
python run_pipeline.py --naive 0 --advanced 150   # headline PR-AUC collapses
python run_pipeline.py --review-capacity 25       # precision@k for a smaller queue
```

- Raise `COST_FALSE_POSITIVE` until the cost-optimal action becomes "flag
  nobody". That threshold is a genuinely useful number to know.
- Weaken the advanced bot's tells one at a time in `_advanced_bot_session` —
  telemetry rate, then uniform jitter, then the synchronised launch — to
  simulate an operator adapting. Each removal costs real recall, which is the
  drift you'd fight in production.
- Swap the isolation forest for a semi-supervised approach (positive-unlabelled
  learning, or a one-class model fitted on *confidently* human sessions). That's
  the natural next step given what the results say.

## Transfer

Same machinery as payment fraud, market-manipulation surveillance, and trade
anomaly monitoring: heavy class imbalance, adversarial drift, precision at a
fixed review capacity, and cost-asymmetric decisions where the threshold
matters more than the model.

For real labelled data with similar structure, look at CIC-IDS2017 or
UNSW-NB15.
