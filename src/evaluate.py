"""Evaluation for a heavily imbalanced, cost-asymmetric problem.

Accuracy is meaningless here -- predicting "human" for everything scores ~98%.
What matters is precision at the operating point you can actually afford, and
how the model does on the sophisticated tier specifically, since that is the
tier that takes the inventory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

# Blocking a real customer costs far more than missing a bot: they don't come
# back, they complain publicly, and support has to unpick it. The ratio is the
# single most important number in the whole pipeline and it is a business
# input, not something you can learn from the data.
COST_FALSE_POSITIVE = 50.0
COST_FALSE_NEGATIVE = 5.0


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Precision among the k highest-scoring sessions.

    This is the metric that matches how a review queue actually works: an
    analyst can look at k cases a day, and wants them to be worth looking at.
    """
    if k <= 0:
        return float("nan")
    top = np.argsort(-scores)[:k]
    return float(y[top].mean())


def cost_sweep(y: np.ndarray, scores: np.ndarray, n_steps: int = 200) -> pd.DataFrame:
    """Total expected cost across every threshold.

    "Flag nobody" is always included as a candidate. Without it a detector
    whose scores are all identical -- the rule baseline, once no session trips
    a rule -- would sweep a single degenerate threshold, flag the entire site,
    and report that as optimal. It also makes the genuinely correct answer at
    a high enough false-positive cost reachable: block no one.
    """
    thresholds = np.unique(np.quantile(scores, np.linspace(0.50, 0.9995, n_steps)))
    thresholds = np.append(thresholds, float(scores.max()) + 1.0)
    rows = []
    for t in thresholds:
        pred = scores >= t
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        rows.append(
            {
                "threshold": float(t),
                "flagged": int(pred.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "cost": fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE,
            }
        )
    return pd.DataFrame(rows)


def tier_recall(meta: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.Series:
    """Recall broken out by bot sophistication.

    The headline number hides the story: naive bots inflate it while the
    advanced tier walks through. Always read this table before believing a
    good aggregate score.
    """
    flagged = scores >= threshold
    out = {}
    for tier in ["naive_bot", "advanced_bot"]:
        mask = (meta["actor_class"] == tier).to_numpy()
        out[tier] = float(flagged[mask].mean()) if mask.any() else float("nan")
    human = (meta["actor_class"] == "human").to_numpy()
    out["human_false_positive_rate"] = float(flagged[human].mean()) if human.any() else float("nan")
    return pd.Series(out)


def report(name: str, y: pd.Series, scores: np.ndarray, meta: pd.DataFrame, review_capacity: int = 100) -> dict:
    """One detector's full scorecard."""
    y_arr = y.to_numpy()
    sweep = cost_sweep(y_arr, scores)
    best = sweep.loc[sweep["cost"].idxmin()]

    # A flat-score detector (like the rule baseline) has no meaningful ranking,
    # so AUC-style metrics are reported but should be read with that in mind.
    result = {
        "detector": name,
        "pr_auc": float(average_precision_score(y_arr, scores)),
        "roc_auc": float(roc_auc_score(y_arr, scores)),
        "review_capacity": review_capacity,
        "precision_at_k": precision_at_k(y_arr, scores, review_capacity),
        "best_threshold": float(best["threshold"]),
        "cost_at_best": float(best["cost"]),
        "precision_at_best": float(best["precision"]),
        "recall_at_best": float(best["recall"]),
        "flagged_at_best": int(best["flagged"]),
        "tiers": tier_recall(meta, scores, float(best["threshold"])),
        "sweep": sweep,
    }
    return result


def baseline_cost(y: pd.Series) -> float:
    """Cost of doing nothing -- every bot gets through, nobody is blocked.

    Any detector that scores worse than this is actively harmful, and at these
    cost weights that is easier to do than it sounds.
    """
    return float((y == 1).sum() * COST_FALSE_NEGATIVE)


def format_report(r: dict, do_nothing: float) -> str:
    if r["cost_at_best"] < do_nothing:
        verdict = "better"
    elif r["cost_at_best"] == do_nothing:
        # The sweep chose to flag nobody: the detector found nothing worth
        # acting on at these cost weights.
        verdict = "no better -- optimal action is to flag nobody"
    else:
        verdict = "WORSE"

    lines = [
        f"  {r['detector']}",
        f"    PR-AUC {r['pr_auc']:.3f}   ROC-AUC {r['roc_auc']:.3f}",
        f"    precision@{r['review_capacity']} {r['precision_at_k']:.3f}",
        f"    at min-cost threshold: flags {r['flagged_at_best']:>4}  "
        f"precision {r['precision_at_best']:.3f}  recall {r['recall_at_best']:.3f}",
        f"    cost {r['cost_at_best']:>8.0f}   vs {do_nothing:.0f} for doing nothing ({verdict})",
        f"    recall by tier: naive {r['tiers']['naive_bot']:.2f}   "
        f"advanced {r['tiers']['advanced_bot']:.2f}   "
        f"human FPR {r['tiers']['human_false_positive_rate']:.4f}",
    ]
    return "\n".join(lines)
