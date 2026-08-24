"""
feature_importance.py
-----------------------
Extracts and plots feature importances from the trained root-cause
classifier, so the model isn't a black box in the writeup.
"""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("../outputs/root_cause_model.pkl", "rb") as f:
    pipe = pickle.load(f)

clf = pipe.named_steps["clf"]
preprocess = pipe.named_steps["preprocess"]

cat_features = list(preprocess.named_transformers_["cat"].get_feature_names_out(["decline_code", "payment_method"]))
num_features = ["amount", "hour_of_day", "is_recurring", "prior_failures_7d",
                 "past_success_rate", "past_attempts_30d", "account_age_days", "avg_txn_amount"]
all_features = cat_features + num_features

importances = clf.feature_importances_
order = np.argsort(importances)[::-1][:15]  # top 15

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([all_features[i] for i in order][::-1], importances[order][::-1], color="#2f6fed")
ax.set_xlabel("Feature importance")
ax.set_title("Root Cause Classifier — Top Feature Importances")
fig.tight_layout()
fig.savefig("../outputs/feature_importance.png", dpi=150)
print("Saved ../outputs/feature_importance.png")

print("\nTop features:")
for i in order[:10]:
    print(f"  {all_features[i]:30s} {importances[i]:.4f}")
