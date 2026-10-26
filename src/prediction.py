"""
prediction.py
--------------
Turns a trained model + one order's features into the kind of output the
logistics team would actually act on: a probability, a risk band, and a
short list of the factors driving that score for this specific order
(not just the model's global feature importance).
"""

import numpy as np
import pandas as pd

from modeling import risk_bucket, NUMERIC, CATEGORICAL

# Human-readable labels for the reason-code output
FEATURE_LABELS = {
    "seller_historical_delay_rate": "Seller's track record of late deliveries",
    "category_historical_delay_rate": "Product category's typical delay rate",
    "customer_state_historical_delay_rate": "Delivery region's typical delay rate",
    "seller_customer_distance_km": "Long shipping distance",
    "promised_delivery_days": "Length of promised delivery window",
    "approval_delay_hours": "Slow payment/order approval",
    "total_freight": "High freight cost",
    "n_items": "Multiple items in one order",
    "n_distinct_sellers": "Order split across multiple sellers",
    "total_weight_g": "Heavy shipment",
    "is_weekend_purchase": "Order placed on a weekend",
    "same_state": "Seller and customer in different states",
    "purchase_month": "Seasonal timing of the order",
    "customer_state": "Delivery state",
    "avg_freight": "High freight cost per item",
    "purchase_dow": "Day of week the order was placed",
    "customer_state_historical_delay_rate": "Delivery region's typical delay rate",
}


def explain_order(pipe, row, feature_columns, top_n=4):
    """
    LEGACY / GLOBAL explanation, kept for the Random Forest in notebook 05.
    Returns the model's top features by GLOBAL importance together with
    this order's value for each. Because the ranking uses global
    importance only, the same few features come back for nearly every
    order -- it says what the model relies on overall, not what drove
    THIS order's score. For genuine per-order attribution use
    explain_orders() below (XGBoost contributions).
    """
    proba = pipe.predict_proba(row[feature_columns])[0, 1]
    level = risk_bucket(proba)

    clf = pipe.named_steps["clf"]
    pre = pipe.named_steps["pre"]
    if not hasattr(clf, "feature_importances_"):
        return proba, level, []

    num_names = NUMERIC
    cat_names = list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
    all_names = num_names + cat_names
    importances = dict(zip(all_names, clf.feature_importances_))

    # rank the *numeric* drivers by importance x how unusual this order's value is
    scores = []
    numeric_row = row[NUMERIC].iloc[0]
    for feat in NUMERIC:
        imp = importances.get(feat, 0)
        val = numeric_row[feat]
        scores.append((feat, imp, val))
    scores.sort(key=lambda x: x[1], reverse=True)

    reasons = []
    for feat, imp, val in scores[:top_n]:
        label = FEATURE_LABELS.get(feat, feat.replace("_", " "))
        reasons.append((feat, round(float(val), 2), label))

    return float(proba), level, reasons


# ---------------------------------------------------------------------
# Genuine per-order explanations (XGBoost native TreeSHAP contributions)
# ---------------------------------------------------------------------
def _transformed_to_original(pre):
    """Map each column of the fitted preprocessor's output back to the
    original feature it came from (one-hot columns -> their category
    feature), so contributions can be summed per original feature."""
    mapping = list(NUMERIC)
    for cat, cats in zip(CATEGORICAL, pre.named_transformers_["cat"].categories_):
        mapping.extend([cat] * len(cats))
    return mapping


def explain_orders(pipe, X, top_n=3):
    """
    Per-order attribution using XGBoost's built-in pred_contribs (exact
    TreeSHAP values, in log-odds). For each row returns the top_n
    features pushing the delay risk UP, as (feature, contribution,
    human_label). Contributions of one-hot columns are summed back onto
    their original feature, so 'product_category' appears once rather
    than as 70 separate columns.

    Unlike explain_order() this genuinely differs per order: two orders
    with the same probability can be risky for entirely different
    reasons, and this shows which.
    """
    import xgboost as xgb
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    if not hasattr(clf, "get_booster"):
        raise TypeError("explain_orders needs an XGBoost pipeline.")
    Xt = pre.transform(X)
    contribs = clf.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[:, :-1]  # drop bias column
    mapping = np.array([str(m) for m in _transformed_to_original(pre)])
    features = [str(f) for f in dict.fromkeys(mapping)]  # plain str, unique, order-preserving

    out = []
    for i in range(contribs.shape[0]):
        per_feat = {f: float(contribs[i, mapping == f].sum()) for f in features}
        up = sorted(((f, c) for f, c in per_feat.items() if c > 0), key=lambda t: t[1], reverse=True)[:top_n]
        out.append([(f, round(c, 3), FEATURE_LABELS.get(f, f.replace("_", " "))) for f, c in up])
    return out
