"""
generate_data.py
----------------
Generates a synthetic dataset of FAILED payment transactions for the
Payment Degradation -> Root Cause -> Recovery Action project.

Each row = one failed payment attempt, with:
  - gateway-level decline/error code
  - payment method, amount, customer history features
  - a TRUE root cause label (ground truth, used only for evaluation)
  - a TRUE "was it actually recoverable, and via what action" outcome
    (simulated with realistic conditional probabilities, used only to
    compute the final $-recovered comparison, never fed to the model
    as a feature)

Run:
    python3 generate_data.py --n 5000 --seed 42 --out ../data/failed_transactions.csv
"""

import argparse
import numpy as np
import pandas as pd

ROOT_CAUSES = [
    "insufficient_funds",
    "card_expired_invalid",
    "bank_gateway_timeout",
    "risk_fraud_hold",
    "issuer_declined_generic",
    "network_technical_error",
]

# Gateway/bank decline codes we *observe* per root cause. In reality, raw
# gateway codes are NOISY and ambiguous: the same code ("05" / DO_NOT_HONOR
# in particular) gets issued by different banks for genuinely different
# underlying reasons. We model this with a primary pool per root cause
# PLUS a shared pool of ambiguous "generic decline" codes that can show up
# across multiple root causes -- this is what makes the classification
# task realistic rather than a disguised lookup table.
DECLINE_CODE_POOL = {
    "insufficient_funds": ["51", "NSF", "INSUFF_BAL"],
    "card_expired_invalid": ["54", "14", "EXPIRED_CARD", "INVALID_CARD_NO"],
    "bank_gateway_timeout": ["68", "91", "GTW_TIMEOUT", "BANK_UNAVAILABLE"],
    "risk_fraud_hold": ["59", "61", "RISK_HOLD", "VELOCITY_BLOCK"],
    "issuer_declined_generic": ["05", "12", "DO_NOT_HONOR", "GENERIC_DECLINE"],
    "network_technical_error": ["96", "TIMEOUT", "5XX_ERROR", "CONN_RESET"],
}

# Ambiguous codes that different banks issue for genuinely different root
# causes -- e.g. "05 / DO_NOT_HONOR" is issued for insufficient funds by
# some issuers and for risk holds by others. Probability a transaction
# draws from this shared, ambiguous pool instead of its "clean" pool.
AMBIGUOUS_CODE_PROB = 0.30
AMBIGUOUS_CODE_POOL = {
    "insufficient_funds": ["05", "51"],
    "risk_fraud_hold": ["05", "59"],
    "issuer_declined_generic": ["05", "12", "51"],
    "card_expired_invalid": ["05", "14"],
    "bank_gateway_timeout": ["91", "96"],
    "network_technical_error": ["91", "96"],
}

PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet"]


def sample_customer_history(rng, n):
    """Simulate a customer 'quality' score and derived history features."""
    quality = rng.beta(2, 2, size=n)  # 0..1, general reliability proxy
    past_success_rate = np.clip(quality + rng.normal(0, 0.08, n), 0, 1)
    past_attempts_30d = rng.poisson(3, n) + 1
    account_age_days = rng.integers(1, 2000, n)
    avg_txn_amount = np.round(np.exp(rng.normal(6.5, 1.0, n)), 2)  # skewed, INR-ish
    return quality, past_success_rate, past_attempts_30d, account_age_days, avg_txn_amount


def sample_root_causes(rng, n, quality):
    """Root cause distribution, mildly influenced by customer quality.
    Lower quality customers skew towards insufficient_funds / risk_fraud_hold.
    """
    base_probs = np.array([0.28, 0.16, 0.14, 0.10, 0.20, 0.12])
    causes = np.empty(n, dtype=object)
    for i in range(n):
        p = base_probs.copy()
        q = quality[i]
        p[0] *= (1.6 - q)          # insufficient_funds more likely if low quality
        p[3] *= (1.4 - q)          # risk_fraud_hold more likely if low quality
        p[1] *= (0.6 + 0.4 * q)    # card_expired roughly independent
        p = p / p.sum()
        causes[i] = rng.choice(ROOT_CAUSES, p=p)
    return causes


def sample_decline_code(rng, root_cause):
    """Draw a decline code. With some probability, draw from the shared
    ambiguous pool instead of the "clean" pool for this root cause, so the
    same code can legitimately correspond to different true root causes.
    """
    if rng.random() < AMBIGUOUS_CODE_PROB:
        return rng.choice(AMBIGUOUS_CODE_POOL[root_cause])
    return rng.choice(DECLINE_CODE_POOL[root_cause])


def simulate_ground_truth_recovery(rng, root_cause, quality, amount, action):
    """
    Simulates whether a given recovery ACTION actually recovers the payment.
    This encodes domain assumptions about what works for each root cause,
    and is only used to score outcomes -- never exposed to the model.
    """
    # base recovery probability if the "correct" action is used
    correct_action_for = {
        "insufficient_funds": "delayed_retry",
        "card_expired_invalid": "card_update_nudge",
        "bank_gateway_timeout": "immediate_retry",
        "risk_fraud_hold": "manual_escalation",
        "issuer_declined_generic": "immediate_retry",
        "network_technical_error": "immediate_retry",
    }
    base_p = {
        "insufficient_funds": 0.55,
        "card_expired_invalid": 0.45,
        "bank_gateway_timeout": 0.70,
        "risk_fraud_hold": 0.20,   # escalation resolves the *case*, not always recovers $
        "issuer_declined_generic": 0.30,
        "network_technical_error": 0.80,
    }
    p = base_p[root_cause]
    p *= (0.5 + 0.5 * quality)  # higher quality customers recover more reliably

    if action == correct_action_for[root_cause]:
        p_final = p
    elif action == "no_action":
        p_final = 0.02  # money almost never comes back on its own
    else:
        # wrong action chosen: heavy penalty (esp. immediate retry on
        # insufficient_funds or risk hold, which can look like abuse)
        p_final = p * 0.25

    p_final = np.clip(p_final, 0.0, 0.97)
    return rng.random() < p_final


def generate(n, seed, out_path):
    rng = np.random.default_rng(seed)

    quality, past_success_rate, past_attempts_30d, account_age_days, avg_txn_amount = (
        sample_customer_history(rng, n)
    )
    root_cause = sample_root_causes(rng, n, quality)
    decline_code = np.array([sample_decline_code(rng, rc) for rc in root_cause])
    payment_method = rng.choice(PAYMENT_METHODS, size=n, p=[0.45, 0.35, 0.12, 0.08])
    amount = np.round(np.exp(rng.normal(6.8, 1.1, n)), 2)
    hour_of_day = rng.integers(0, 24, n)
    is_recurring = rng.random(n) < 0.35
    prior_failures_7d = rng.poisson(1.2, n)

    df = pd.DataFrame({
        "transaction_id": [f"txn_{i:06d}" for i in range(n)],
        "decline_code": decline_code,
        "payment_method": payment_method,
        "amount": amount,
        "hour_of_day": hour_of_day,
        "is_recurring": is_recurring,
        "prior_failures_7d": prior_failures_7d,
        "past_success_rate": np.round(past_success_rate, 3),
        "past_attempts_30d": past_attempts_30d,
        "account_age_days": account_age_days,
        "avg_txn_amount": avg_txn_amount,
        "true_root_cause": root_cause,       # ground truth (eval only)
        "_customer_quality": quality,        # latent var, eval/simulation only
    })

    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["true_root_cause"].value_counts(normalize=True).round(3))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="../data/failed_transactions.csv")
    args = ap.parse_args()
    generate(args.n, args.seed, args.out)
