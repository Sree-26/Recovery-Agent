"""
recovery_engine.py
-------------------
Takes the root-cause classifier's predictions and, for each failed
transaction, decides:
  1. Which recovery action to take (rule-based mapping from predicted
     root cause -> action, gated by prediction confidence)
  2. Whether to act at all (stopping rules: retry caps, risk-hold
     escalation-only, minimum confidence threshold)
  3. Simulates the outcome against ground truth (for evaluation only)
  4. Writes a full audit trail row per transaction/attempt

Also runs a NAIVE BASELINE ("retry everything immediately, once") for
comparison, so the $-recovered uplift is measurable rather than asserted.

Outputs:
  - ../outputs/audit_trail.csv         (every decision + outcome, smart agent)
  - ../outputs/baseline_outcomes.csv   (naive retry-everything baseline)
  - ../outputs/recovery_summary.json   (headline $ recovered, per-cause stats)
"""

import json

import numpy as np
import pandas as pd

from generate_data import simulate_ground_truth_recovery

TEST_PRED_PATH = "../data/test_predictions.csv"
AUDIT_OUT = "../outputs/audit_trail.csv"
BASELINE_OUT = "../outputs/baseline_outcomes.csv"
SUMMARY_OUT = "../outputs/recovery_summary.json"

# --- Action policy: predicted root cause -> recovery action -------------
ACTION_FOR_CAUSE = {
    "insufficient_funds": "delayed_retry",
    "card_expired_invalid": "card_update_nudge",
    "bank_gateway_timeout": "immediate_retry",
    "risk_fraud_hold": "manual_escalation",
    "issuer_declined_generic": "immediate_retry",
    "network_technical_error": "immediate_retry",
}

# --- Guardrails / stopping rules -----------------------------------------
MIN_CONFIDENCE = 0.45          # below this, don't auto-act -> flag for manual review
# NOTE: prior_failures_7d reflects failures BEFORE this attempt (a classifier
# feature drawn from Poisson(1.2), so 2+ occurs ~35% of the time by chance).
# The stopping rule should only trigger for genuine repeat-failure abuse
# (i.e. we've already hammered this transaction), not normal pre-existing
# history -- hence a higher threshold than the classifier feature's mean.
MAX_RETRIES_PER_TXN = 4        # hard cap: stop auto-retrying after repeated real failures
NEVER_AUTO_RETRY_CAUSES = {"risk_fraud_hold"}  # compliance: always escalate, never auto-retry money movement
RETRY_COOLDOWN_HOURS = {       # minimum wait before a retry attempt, per action
    "immediate_retry": 0,
    "delayed_retry": 48,       # wait for likely next salary/balance cycle
    "card_update_nudge": 0,    # not a retry, a customer-facing nudge
    "manual_escalation": None,  # not a retry at all
}


def decide_action(pred_cause, confidence, prior_failures_7d):
    """Returns (action, escalated_reason_or_none)."""
    if pred_cause in NEVER_AUTO_RETRY_CAUSES:
        return "manual_escalation", "compliance_never_auto_retry"

    if confidence < MIN_CONFIDENCE:
        return "manual_review", "low_confidence_prediction"

    if prior_failures_7d >= MAX_RETRIES_PER_TXN:
        return "manual_review", "retry_cap_exceeded"

    return ACTION_FOR_CAUSE[pred_cause], None


def run_smart_agent(df, rng):
    rows = []
    for _, r in df.iterrows():
        pred_cause = r["pred_root_cause"]
        confidence = r[f"proba_{pred_cause}"]
        action, escalated_reason = decide_action(
            pred_cause, confidence, r["prior_failures_7d"]
        )

        if action in ("manual_review", "manual_escalation"):
            # No automated money-movement action taken. For risk_fraud_hold,
            # the case still gets resolved via a human, but we do NOT credit
            # the agent with an automated recovery -- conservative accounting.
            recovered = False
            if action == "manual_escalation":
                # Human-in-the-loop outcome, modeled at a lower, honest rate.
                recovered = simulate_ground_truth_recovery(
                    rng, r["true_root_cause"], r["_customer_quality"],
                    r["amount"], "manual_escalation"
                )
        else:
            recovered = simulate_ground_truth_recovery(
                rng, r["true_root_cause"], r["_customer_quality"],
                r["amount"], action
            )

        rows.append({
            "transaction_id": r["transaction_id"],
            "amount": r["amount"],
            "true_root_cause": r["true_root_cause"],
            "pred_root_cause": pred_cause,
            "prediction_confidence": round(float(confidence), 3),
            "action_taken": action,
            "escalation_reason": escalated_reason,
            "cooldown_hours": RETRY_COOLDOWN_HOURS.get(action),
            "recovered": bool(recovered),
            "amount_recovered": round(float(r["amount"]) if recovered else 0.0, 2),
        })
    return pd.DataFrame(rows)


def run_naive_baseline(df, rng):
    """Naive baseline: retry every failed transaction immediately, once,
    regardless of root cause. No escalation, no cooldown, no cap logic
    beyond a single attempt. This is the 'do nothing smart' comparator.
    """
    rows = []
    for _, r in df.iterrows():
        recovered = simulate_ground_truth_recovery(
            rng, r["true_root_cause"], r["_customer_quality"],
            r["amount"], "immediate_retry"  # naive: always immediate retry
        )
        rows.append({
            "transaction_id": r["transaction_id"],
            "amount": r["amount"],
            "true_root_cause": r["true_root_cause"],
            "action_taken": "immediate_retry_naive",
            "recovered": bool(recovered),
            "amount_recovered": round(float(r["amount"]) if recovered else 0.0, 2),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(TEST_PRED_PATH)
    rng = np.random.default_rng(123)  # separate rng for outcome simulation

    smart = run_smart_agent(df, rng)
    rng2 = np.random.default_rng(123)  # same seed -> fair comparison, same "luck"
    baseline = run_naive_baseline(df, rng2)

    smart.to_csv(AUDIT_OUT, index=False)
    baseline.to_csv(BASELINE_OUT, index=False)

    total_at_risk = df["amount"].sum()
    smart_recovered = smart["amount_recovered"].sum()
    baseline_recovered = baseline["amount_recovered"].sum()

    summary = {
        "total_transactions": int(len(df)),
        "total_amount_at_risk": round(float(total_at_risk), 2),
        "smart_agent": {
            "amount_recovered": round(float(smart_recovered), 2),
            "recovery_rate_pct": round(100 * smart_recovered / total_at_risk, 2),
            "txns_recovered": int(smart["recovered"].sum()),
            "txns_escalated_manual": int((smart["action_taken"] == "manual_escalation").sum()),
            "txns_flagged_review": int((smart["action_taken"] == "manual_review").sum()),
        },
        "naive_baseline": {
            "amount_recovered": round(float(baseline_recovered), 2),
            "recovery_rate_pct": round(100 * baseline_recovered / total_at_risk, 2),
            "txns_recovered": int(baseline["recovered"].sum()),
        },
        "uplift": {
            "extra_amount_recovered": round(float(smart_recovered - baseline_recovered), 2),
            "relative_uplift_pct": round(
                100 * (smart_recovered - baseline_recovered) / baseline_recovered, 2
            ) if baseline_recovered > 0 else None,
        },
        "per_root_cause_recovery_rate_smart": (
            smart.groupby("true_root_cause")["recovered"].mean().round(3).to_dict()
        ),
        "per_root_cause_recovery_rate_baseline": (
            baseline.groupby("true_root_cause")["recovered"].mean().round(3).to_dict()
        ),
    }

    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
