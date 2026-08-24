"""
make_summary_chart.py
-----------------------
Builds the two headline figures for the README / pitch video:
  1. Overall $ recovered: smart agent vs naive baseline
  2. Per-root-cause recovery rate: smart agent vs naive baseline
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("../outputs/recovery_summary.json") as f:
    s = json.load(f)

# --- Figure 1: overall $ recovered ---------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
labels = ["Naive baseline\n(retry everything)", "Smart agent\n(root-cause routed)"]
values = [s["naive_baseline"]["amount_recovered"], s["smart_agent"]["amount_recovered"]]
colors = ["#9aa5b1", "#2f6fed"]
bars = ax.bar(labels, values, color=colors, width=0.55)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.02,
             f"₹{v:,.0f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Amount recovered (₹)")
ax.set_title(f"Money Recovered on Held-Out Batch (n={s['total_transactions']})\n"
             f"+{s['uplift']['relative_uplift_pct']}% relative uplift")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("../outputs/recovery_comparison.png", dpi=150)
print("Saved ../outputs/recovery_comparison.png")

# --- Figure 2: per root-cause recovery rate --------------------------------
causes = list(s["per_root_cause_recovery_rate_smart"].keys())
smart_rates = [s["per_root_cause_recovery_rate_smart"][c] * 100 for c in causes]
base_rates = [s["per_root_cause_recovery_rate_baseline"][c] * 100 for c in causes]

x = range(len(causes))
width = 0.38
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar([i - width / 2 for i in x], base_rates, width, label="Naive baseline", color="#9aa5b1")
ax.bar([i + width / 2 for i in x], smart_rates, width, label="Smart agent", color="#2f6fed")
ax.set_xticks(list(x))
ax.set_xticklabels([c.replace("_", "\n") for c in causes], fontsize=9)
ax.set_ylabel("Recovery rate (%)")
ax.set_title("Recovery Rate by Root Cause: Smart Routing vs Naive Retry")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("../outputs/recovery_by_cause.png", dpi=150)
print("Saved ../outputs/recovery_by_cause.png")
