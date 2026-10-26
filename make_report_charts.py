"""Charts for the report / deck that come from the production scorer's
output (dashboard/risk_summary.json). Run after export_risk_summary.py."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open("dashboard/risk_summary.json", encoding="utf-8"))
SEM = {"LOW": "#2e8b57", "MEDIUM": "#c9a227", "HIGH": "#d9771e", "CRITICAL": "#c0392b"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# 15 - does the band mean anything?
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
b = S["bands"]
bars = ax.bar([f'{x["level"]}\n{x["n"]:,} orders' for x in b], [x["late_rate"] for x in b], color=[SEM[x["level"]] for x in b], width=0.6)
ax.axhline(S["base_late_rate"], color="#5d6876", ls="--", lw=1)
ax.text(-0.3, S["base_late_rate"] + 1.2, f"all orders: {S['base_late_rate']}% late", ha="left", color="#5d6876", fontsize=9)
for r, x in zip(bars, b):
    ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1, f"{x['late_rate']}%", ha="center", fontweight="bold")
ax.set_ylabel("Share actually late (%)")
ax.set_ylim(0, max(x["late_rate"] for x in b) * 1.18)
ax.set_title("Held-out test orders: how often each risk band was really late", loc="left", fontsize=12)
fig.tight_layout(); fig.savefig("dashboard/charts/15_risk_band_outcomes.png"); plt.close(fig)

# 16 - suggested actions
acts = S["actions"][:6][::-1]
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
short = {"Peak-season": "Peak season: prioritise + pad ETA", "Promise window": "Tight promise: send realistic ETA",
         "Book an express": "Long route: book express carrier", "Use the most reliable": "Risky region: best carrier + warn customer",
         "Payment/approval": "Slow approval: fast-track picking", "Contact the seller": "Slow seller: confirm dispatch",
         "High freight": "Hard route: confirm pickup slot", "Prioritise in": "Other: prioritise + monitor"}
lab = [next((v for k, v in short.items() if a["action"].startswith(k)), a["action"][:40]) for a in acts]
ax.barh(lab, [a["n"] for a in acts], color="#b8640f")
for i, a in enumerate(acts):
    ax.text(a["n"] + 15, i, f"{a['n']:,}", va="center", fontsize=9)
ax.set_xlabel("Flagged test orders")
ax.set_title(f"Suggested action for the {S['n_flagged']:,} flagged test orders", loc="left", fontsize=12)
fig.tight_layout(); fig.savefig("dashboard/charts/16_suggested_actions.png"); plt.close(fig)
print("saved 15, 16")

# 17 - final model feature importance with readable labels
FI = json.load(open("dashboard/feature_importance.json", encoding="utf-8"))[:12][::-1]
NAMES = {"purchase_month": "Month of purchase", "n_distinct_sellers": "Sellers in the order",
         "promised_delivery_days": "Promised delivery window", "customer_state_historical_delay_rate": "Region's past delay rate",
         "seller_historical_delay_rate": "Seller's past delay rate", "same_state": "Same-state shipment",
         "seller_customer_distance_km": "Shipping distance"}
def nice(f):
    if f in NAMES: return NAMES[f]
    for pre, lab in (("customer_state_", "Customer in "), ("seller_state_", "Seller in "), ("product_category_", "Category: ")):
        if f.startswith(pre): return lab + f[len(pre):].replace("_", " ")
    return f.replace("_", " ")
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
ax.barh([nice(x["feature"]) for x in FI], [x["importance"] for x in FI], color="#2f5d7c")
ax.set_xlabel("Gain importance")
ax.set_title("Tuned XGBoost: top 12 features", loc="left", fontsize=12)
fig.tight_layout(); fig.savefig("dashboard/charts/17_final_feature_importance_labeled.png"); plt.close(fig)
print("saved 17")

# 18 - delay patterns (clusters of late orders)
import textwrap
P = json.load(open("backend/app_data/patterns.json", encoding="utf-8"))["patterns"][::-1]
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
labels = [f"P{p['rank']}  " + "\n".join(textwrap.wrap(p["name"], 34)) for p in P]
ax.barh(labels, [p["share_of_late"] for p in P], color="#b8640f", height=0.62)
for i, p in enumerate(P):
    ax.text(p["share_of_late"] + 0.4, i, f"{p['share_of_late']}%  ({p['n']:,})", va="center", fontsize=9)
ax.set_xlabel("Share of all late deliveries (%)")
ax.set_xlim(0, max(p["share_of_late"] for p in P) * 1.35)
ax.tick_params(axis="y", labelsize=9)
ax.set_title("Six delay patterns among 7,826 late orders", loc="left", fontsize=12)
fig.tight_layout(); fig.savefig("dashboard/charts/18_delay_patterns.png"); plt.close(fig)
print("saved 18")
