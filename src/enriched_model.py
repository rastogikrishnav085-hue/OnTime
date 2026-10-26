"""
enriched_model.py
------------------
The enriched CHECKOUT model: the original 33 order features plus the real-data history, network, promise
and place features from enriched_features.py. Members: XGBoost, LightGBM, histogram gradient boosting
(logistic regression only as a weak reference in the report, not as a member).

Everything here is built on top of the existing ensemble module (BlendModel, save/load with SHA-256
checks, OOF weights, thresholds), so the blend keeps the same probability scale the risk bands use.
"""
import os

import numpy as np
import pandas as pd
# LightGBM is needed to train the enriched model, but is not required to SERVE the
# shipped enriched artifact (which uses the portable XGBoost member).  Some user
# environments have an incompatible optional Dask installation; LightGBM imports
# Dask opportunistically and older Dask versions can fail during import with a
# TypeError instead of ImportError.  Do not let that optional training dependency
# prevent the web application from starting.
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
    LIGHTGBM_IMPORT_ERROR = None
except Exception as _lightgbm_exc:  # noqa: BLE001 - third-party import can fail in several ways
    LIGHTGBM_AVAILABLE = False
    LIGHTGBM_IMPORT_ERROR = f"{type(_lightgbm_exc).__name__}: {_lightgbm_exc}"

    class LGBMClassifier:  # type: ignore[no-redef]
        """Import-safe placeholder; training gives a clear dependency error."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "LightGBM is unavailable in this Python environment. "
                "The OnTime serving app can still run with the shipped model. "
                f"Import error: {LIGHTGBM_IMPORT_ERROR}. "
                "Run setup.bat to install the project dependencies in .venv."
            )
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

from enriched_features import EXTRA_COLUMNS, LABELS as EXTRA_LABELS, ServingContext
from features import FEATURE_COLUMNS
from modeling import CATEGORICAL, NUMERIC
from prediction import FEATURE_LABELS

ENRICHED_COLUMNS = FEATURE_COLUMNS + EXTRA_COLUMNS
NUMERIC_E = NUMERIC + EXTRA_COLUMNS
CATEGORICAL_E = list(CATEGORICAL)
ALL_LABELS = {**FEATURE_LABELS, **EXTRA_LABELS}

MEMBER_LABELS_E = {
    "xgb": "XGBoost (gradient boosting)",
    "lgbm": "LightGBM (gradient boosting)",
    "hgb": "Histogram gradient boosting",
}

ACTION_BY_DRIVER_EXTRA = {
    "seller_handling_h_hist": "This seller is usually slow to hand parcels over: ask for dispatch confirmation within 24 hours.",
    "seller_overdue_open": "This seller already has overdue orders: contact them before adding more work to the queue.",
    "seller_overdue_share": "This seller already has overdue orders: contact them before adding more work to the queue.",
    "seller_transit_d_hist": "This seller's parcels usually travel slowly: use a faster carrier option.",
    "lane_late_hist": "This route is often late: use the most reliable carrier and warn the customer early.",
    "lane_transit_d_hist": "This route is slow: use the most reliable carrier and warn the customer early.",
    "promise_minus_lane_transit": "The promise is tight for this route's usual transit time: send the customer a realistic date now.",
    "cstate_overdue_open": "Many orders to this state are already overdue: expect congestion and update the customer proactively.",
    "cstate_overdue_share": "Many orders to this state are already overdue: expect congestion and update the customer proactively.",
    "net_overdue_open": "The whole network has many overdue orders right now: prioritise this order in dispatch.",
    "net_overdue_share": "The whole network has many overdue orders right now: prioritise this order in dispatch.",
    "lane_overdue_open": "This route has overdue orders right now: use an alternative carrier if possible.",
    "lane_overdue_share": "This route has overdue orders right now: use an alternative carrier if possible.",
    "shipping_limit_days": "The seller was given a long handling window: confirm the dispatch date with them.",
    "holidays_in_promise": "Public holidays fall inside the promise window: pad the customer's date.",
    "promise_business_days": "Few business days inside the promise window: pad the customer's date.",
    "days_from_black_friday": "Order placed near Black Friday: expect carrier congestion and pad the customer's date.",
    "cust_zip3_late_hist": "This postcode area is often late: warn the customer and use the most reliable carrier.",
    "cust_city_late_hist": "This city is often late: warn the customer and use the most reliable carrier.",
}


class LGBMCat(LGBMClassifier):
    """LightGBM that reads the ordinal-encoded category columns (the last ones) as true categories."""

    def fit(self, X, y, **kw):
        n_num = len(NUMERIC_E)
        kw["categorical_feature"] = list(range(n_num, n_num + len(CATEGORICAL_E)))
        return super().fit(X, y, **kw)


def _tree_pre():
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC_E),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), CATEGORICAL_E),
    ], sparse_threshold=0)


def _onehot_pre():
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC_E),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_E),
    ], sparse_threshold=0)


def build_xgb_e(params=None, scale_pos_weight=1.0):
    p = dict(n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
             reg_lambda=10, eval_metric="auc", tree_method="hist", random_state=42, n_jobs=1)
    p.update(params or {})
    p["scale_pos_weight"] = scale_pos_weight
    return Pipeline([("pre", _onehot_pre()), ("clf", XGBClassifier(**p))])


def build_lgbm_e(params=None, scale_pos_weight=1.0):
    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError(
            "Cannot train the LightGBM member because LightGBM failed to import. "
            f"Import error: {LIGHTGBM_IMPORT_ERROR}. "
            "Run setup.bat in a clean project virtual environment."
        )
    p = dict(n_estimators=700, learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.6, reg_lambda=10, random_state=42, n_jobs=1, verbose=-1)
    p.update(params or {})
    p["scale_pos_weight"] = scale_pos_weight
    return Pipeline([("pre", _tree_pre()), ("clf", LGBMCat(**p))])


def build_hgb_e(params=None, balanced=False):
    p = dict(max_iter=500, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=5.0)
    p.update(params or {})
    mask = [False] * len(NUMERIC_E) + [True] * len(CATEGORICAL_E)
    clf = HistGradientBoostingClassifier(categorical_features=mask, class_weight="balanced" if balanced else None, random_state=42, **p)
    return Pipeline([("pre", _tree_pre()), ("clf", clf)])


def as_frame(X):
    """Model input in the exact column order, numbers as floats (blank history stays NaN)."""
    X = X[ENRICHED_COLUMNS].copy()
    for c in NUMERIC_E:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X


# ---------------------------------------------------------------- reasons for each order
def explain_rows(blend, X, top_n=3):
    """Top factors pushing each order's risk UP, as (feature, contribution, label), from the XGBoost member's
    exact TreeSHAP values (one-hot columns are summed back onto their feature)."""
    import xgboost as xgb
    member = blend.members[blend.explainer] if hasattr(blend, "members") else blend
    pre, clf = member.named_steps["pre"], member.named_steps["clf"]
    mapping = list(NUMERIC_E)
    for cat, cats in zip(CATEGORICAL_E, pre.named_transformers_["cat"].categories_):
        mapping.extend([cat] * len(cats))
    mapping = np.array([str(m) for m in mapping])
    Xt = pre.transform(as_frame(X))
    contrib = clf.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[:, :-1]
    feats = [str(f) for f in dict.fromkeys(mapping)]
    out = []
    for i in range(contrib.shape[0]):
        per = {f: float(contrib[i, mapping == f].sum()) for f in feats}
        up = sorted(((f, c) for f, c in per.items() if c > 0), key=lambda t: t[1], reverse=True)[:top_n]
        out.append([(f, round(c, 3), ALL_LABELS.get(f, f.replace("_", " "))) for f, c in up])
    return out


def action_for(risk_level, flagged, driver):
    from scoring import suggested_action
    if flagged and driver in ACTION_BY_DRIVER_EXTRA:
        text = ACTION_BY_DRIVER_EXTRA[driver]
        return f"URGENT: {text}" if risk_level == "CRITICAL" else text
    return suggested_action(risk_level, flagged, driver)


# ---------------------------------------------------------------- serving wrapper
class EnrichedModel:
    """Serves the 70-feature blend through the same interface the 33-feature models use, so it passes through
    scoring.score_orders (validation, guardrails, risk bands, confidence flags) unchanged.

    Where do the 37 extra features come from?
      * queue orders / rows that already carry them (net_open is present): the exact values computed at the
        order's own purchase time by build_dataset.py;
      * hand-typed orders and uploaded CSVs: the latest REAL history snapshot (models/enriched/context.json):
        the seller's and route's past behaviour, typical network state; postcode, city and coordinates stay
        blank (the models read blanks natively). The scorer says so in the row's warnings."""

    extra_columns = EXTRA_COLUMNS

    def __init__(self, blend, context_path=None):
        self.blend = blend
        self.context = ServingContext(context_path) if context_path and os.path.exists(context_path) else None

    # make it look like the pipeline the older explanation code expects
    @property
    def named_steps(self):
        return self.blend.named_steps

    @property
    def members(self):
        return self.blend.members

    @property
    def weights(self):
        return self.blend.weights

    def serving_extras(self, rows):
        """(DataFrame of EXTRA_COLUMNS with rows' index, list of warning-or-None per row)."""
        out, warns = [], []
        for _, r in rows.iterrows():
            exact = ("net_open" in r.index) and pd.notna(r["net_open"])
            if exact:
                out.append({c: (r[c] if c in r.index else np.nan) for c in EXTRA_COLUMNS})
                warns.append(None)
            elif self.context is not None:
                vals, known = self.context.fill(r.to_dict())
                out.append(vals)
                w = []
                if not known["seller_known"]:
                    w.append("enriched model: this seller has no recorded history, so its handling-time and speed history is blank")
                if not known["lane_known"]:
                    w.append("enriched model: no history for this seller-state to customer-state route")
                warns.append("; ".join(w) or None)
            else:
                out.append({c: np.nan for c in EXTRA_COLUMNS})
                warns.append("enriched model: history snapshot (models/enriched/context.json) not found, history features are blank")
        return pd.DataFrame(out, index=rows.index)[EXTRA_COLUMNS], warns

    def predict_proba(self, X):
        return self.blend.predict_proba(as_frame(X))

    def member_probabilities(self, X):
        return self.blend.member_probabilities(as_frame(X))

    def explain(self, X, top_n=3):
        return explain_rows(self.blend, X, top_n)

    def action_for(self, risk_level, flagged, driver):
        return action_for(risk_level, flagged, driver)
