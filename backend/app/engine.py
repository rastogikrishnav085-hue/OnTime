"""
engine.py - everything the API needs from the model side, loaded once.

The web app reuses the project's own modules (src/) rather than
re-implementing anything: the same integrity-checked model loader, the
same guardrails, the same scorer and the same TreeSHAP explanations that
the notebooks, tests and score.py use.
"""
import json
import os
import sys
import warnings
from functools import cached_property

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from features import FEATURE_COLUMNS                         # noqa: E402
from modeling import NUMERIC, risk_bucket       # noqa: E402
from artifacts import load_model, load_json                  # noqa: E402
from inference_validation import load_reference_stats        # noqa: E402
from scoring import score_orders, RATE_FEATURES, required_input_columns  # noqa: E402
from ensemble import load_ensemble, EnsembleError, apply_calibration         # noqa: E402
from enriched_model import (  # noqa: E402
    ENRICHED_COLUMNS, EnrichedModel, LIGHTGBM_AVAILABLE, LIGHTGBM_IMPORT_ERROR,
)

MODEL_DIR = os.environ.get("ONTIME_MODEL_DIR", os.path.join(ROOT, "models"))
APP_DATA = os.path.join(ROOT, "backend", "app_data")
DASH = os.path.join(ROOT, "dashboard")

# Fields the simulator form exposes; everything else is derived or filled
# from the typical order (and the response says which).
FORM_FIELDS = [
    "seller_id", "product_category", "customer_state", "seller_state", "seller_customer_distance_km",
    "promised_delivery_days", "purchase_month", "purchase_dow", "purchase_hour", "approval_delay_hours",
    "n_items", "n_distinct_sellers", "total_price", "total_freight", "total_weight_g",
    "payment_type", "n_payment_installments",
]


def _num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


MODEL_TITLES = {"xgboost": "Tuned XGBoost", "ensemble": "Weighted ensemble", "enriched": "Enriched chronological model", "synthetic_operational": "Dispatch operational model (synthetic benchmark)"}
SYNTHETIC_OPERATIONAL_FEATURES = ["route_km", "weather_severity_forecast", "congestion_forecast", "carrier_reliability_prior", "capacity_utilization_at_dispatch", "remaining_sla_slack_hours", "facility_backlog_at_dispatch", "route_disruption_alert_at_dispatch", "last_scan_age_hours", "estimated_truck_delay_hours_at_dispatch", "hour_at_dispatch", "weekday_at_dispatch", "price", "freight_value", "product_weight_g", "dispatch_lag_hours"]


class Engine:
    def __init__(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model, self.model_meta = load_model(
                MODEL_DIR, FEATURE_COLUMNS,
                data_path=os.path.join(ROOT, "data", "processed", "ontime_model_ready.csv"),
                log=lambda m: print(f"  [OnTime] {m}", flush=True))
        self.threshold = float(self.model_meta["threshold"])
        # ---- optional ensemble (a second, selectable model) ----
        self.models = {"xgboost": (self.model, self.threshold)}
        self.ensemble_meta, self.ensemble_error, self.calibration = None, None, None
        try:
            blend, emeta = load_ensemble(os.path.join(MODEL_DIR, "ensemble"), FEATURE_COLUMNS)
            self.models["ensemble"] = (blend, float(emeta["threshold"]))
            self.ensemble_meta = emeta
            self.calibration = emeta.get("calibration")
        except EnsembleError as ex:
            self.ensemble_error = str(ex)
        except Exception as ex:  # noqa: BLE001 - the ensemble is optional; never block the app on it
            self.ensemble_error = f"{ex.__class__.__name__}: {ex}"
        self.enriched_meta, self.enriched_error = None, None
        # LightGBM is only needed to retrain one candidate model, so a missing package is recorded
        # quietly (shown on the Model page) instead of printed at every launch. Training raises its own
        # clear error if it is ever asked for without the package.
        self.lightgbm_available = bool(LIGHTGBM_AVAILABLE)
        self.lightgbm_import_error = None if LIGHTGBM_AVAILABLE else LIGHTGBM_IMPORT_ERROR
        try:
            enriched_blend, enriched_meta = load_ensemble(os.path.join(MODEL_DIR, "enriched"), ENRICHED_COLUMNS)
            enriched_model = EnrichedModel(enriched_blend, os.path.join(MODEL_DIR, "enriched", "context.json"))
            self.models["enriched"] = (enriched_model, float(enriched_meta["threshold"]))
            self.enriched_meta = enriched_meta
            if enriched_meta.get("calibration"):
                self.calibration = enriched_meta["calibration"]
        except Exception as ex:  # optional enriched model must not prevent the app starting
            self.enriched_error = f"{ex.__class__.__name__}: {ex}"
        # The synthetic operational model is a separately trained dispatch-checkpoint model.
        # It is used for live scoring only when explicitly selected (or configured as default);
        # the historical risk queue remains scored by the real-Olist chronological model.
        self.synthetic_model = None
        self.synthetic_model_error = None
        try:
            import xgboost as xgb
            self.synthetic_model = xgb.Booster()
            self.synthetic_model.load_model(os.path.join(MODEL_DIR, "synthetic_operational", "xgboost_synthetic_operational.json"))
            with open(os.path.join(MODEL_DIR, "synthetic_operational", "metrics.json"), encoding="utf-8") as _mf:
                self.synthetic_metrics = json.load(_mf)
            _imp_path = os.path.join(MODEL_DIR, "synthetic_operational", "feature_importance.csv")
            _imp = pd.read_csv(_imp_path) if os.path.exists(_imp_path) else pd.DataFrame()
            self.synthetic_feature_importance = _imp.head(10).replace({np.nan: None}).to_dict("records") if not _imp.empty else []
            self.models["synthetic_operational"] = (self.synthetic_model, 0.5)
        except Exception as ex:  # optional to historical queue, required only when selected for live scoring
            self.synthetic_model = None
            self.synthetic_model_error = f"{ex.__class__.__name__}: {ex}"
            self.synthetic_metrics = {}
            self.synthetic_feature_importance = []
        self.stats = load_reference_stats(os.path.join(MODEL_DIR, "reference_stats.json"))
        self.lookup = load_json(os.path.join(MODEL_DIR, "rate_lookup.json"))
        self.meta = json.load(open(os.path.join(APP_DATA, "meta.json"), encoding="utf-8"))
        self.insights = json.load(open(os.path.join(APP_DATA, "insights.json"), encoding="utf-8"))
        self.patterns = json.load(open(os.path.join(APP_DATA, "patterns.json"), encoding="utf-8"))
        self.categories = json.load(open(os.path.join(APP_DATA, "categories.json"), encoding="utf-8"))
        self.network = json.load(open(os.path.join(APP_DATA, "network.json"), encoding="utf-8"))
        self.pattern_by_rank = {p["rank"]: p for p in self.patterns["patterns"]}
        q = pd.read_csv(os.path.join(APP_DATA, "queue.csv"), dtype={"order_id": str, "seller_id": str})
        q["flagged"] = q["flagged"].astype(bool)
        q["focus_tier"] = q["focus_tier"].fillna("")
        q["month"] = q["purchased_at"].str[:7]
        # which model scored the queue (probability / risk_level / flagged columns)
        self.serving = self.meta.get("serving", "xgboost")
        if self.serving not in self.models:
            self.serving = "xgboost"
        else:
            self.model, self.threshold = self.models[self.serving]
        # second opinion: the other model's probability, when the queue carries it
        for key, (_m, thr) in self.models.items():
            col = f"p_{key}"
            if col in q.columns:
                q[f"flag_{key}"] = q[col] >= thr
        self.queue = q
        self.queue_index = {oid: i for i, oid in enumerate(q["order_id"])}
        # The queue as shipped (scored by the serving model). Switching the scoring model in Settings
        # re-scores every queue-based view from the stored per-model probabilities (see set_view).
        self.base_serving, self.base_threshold, self.base_model = self.serving, self.threshold, self.model
        self._base_queue = q
        self._view_cache = {self.serving: q}
        self.view = self._view_info(self.serving, self.serving)

    # ---------- switching the model behind every queue view ----------
    # The newest 19,294 orders carry one stored probability per checkout-time model (p_<key>). The core
    # XGBoost and the ensemble were trained on a random 80% of ALL orders, so these orders were largely in
    # their training data: their queue numbers are optimistic and are labelled as such.
    IN_SAMPLE_MODELS = ("xgboost", "ensemble")

    def _view_info(self, requested, active):
        key = requested if requested not in (None, "", "auto") else self.base_serving
        note = None
        if key == "synthetic_operational":
            note = ("The dispatch model needs dispatch-time signals (weather, congestion, carrier capacity) that the "
                    "historical Olist orders do not contain, so it cannot re-score this queue. The queue, alerts and "
                    "risk bands stay on the real-Olist model; the dispatch model is used for single-order and batch scoring.")
        elif key not in self.models or f"p_{key}" not in self.queue.columns:
            key, note = active, f"The requested model is not available for the queue, so {MODEL_TITLES.get(active, active)} is shown."
        in_sample = active in self.IN_SAMPLE_MODELS
        if in_sample and note is None:
            note = (f"{MODEL_TITLES[active]} was trained on a random 80% of all orders, so most of these newest orders were in its "
                    "training data. Queue results below are optimistic and are not a test of real-world performance; "
                    "the chronological enriched model is the honest one.")
        return {"requested": requested or "auto", "active": active, "label": MODEL_TITLES.get(active, active),
                "in_sample": in_sample, "limited_by": "synthetic_operational" if requested == "synthetic_operational" else None,
                "note": note}

    def _build_view(self, key):
        """Queue re-scored by model `key`: probability, band, alert, priority and next action follow that model."""
        base = self._base_queue
        model, thr = self.models[key]
        q = base.copy()
        w = (base["priority"] / base["probability"].where(base["probability"] > 0)).fillna(1.0)
        p = base[f"p_{key}"].astype(float)
        q["probability"] = p
        q["risk_level"] = p.map(risk_bucket)
        q["flagged"] = p >= thr
        q["priority"] = (p * w).round(4)
        if key != self.base_serving:
            # reasons from THIS model (exact TreeSHAP, vectorised) so the "why" matches the score shown
            try:
                cached = self._stored_reasons(key, q["order_id"])
                if cached is not None:
                    reasons = [json.loads(x) for x in cached]
                else:                       # file missing: compute (slow, 1-2 minutes) rather than show wrong reasons
                    reasons = [[{"feature": f, "label": lab, "contribution": c} for f, c, lab in r] for r in self._explain_fast(model, q)]
                q["reasons_json"] = [json.dumps(r) for r in reasons]
                q["main_driver"] = [r[0]["label"] if r else "" for r in reasons]
                q["_driver_key"] = [r[0]["feature"] if r else None for r in reasons]
            except Exception:  # noqa: BLE001 - keep the base explanations rather than fail the switch
                q["_driver_key"] = None
        from scoring import suggested_action as _act
        drv = q["_driver_key"] if "_driver_key" in q.columns else None
        if drv is None or key == self.base_serving:
            keys = [None] * len(q)
        else:
            keys = list(drv)
        q["suggested_action"] = [_act(rl, bool(fl), dk) for rl, fl, dk in zip(q["risk_level"], q["flagged"], keys)] \
            if key != self.base_serving else base["suggested_action"].values
        if "_driver_key" in q.columns:
            q = q.drop(columns=["_driver_key"])
        q = q.sort_values("probability", ascending=False).reset_index(drop=True)
        for k2, (_m, t2) in self.models.items():
            if f"p_{k2}" in q.columns:
                q[f"flag_{k2}"] = q[f"p_{k2}"] >= t2
        return q

    def _stored_reasons(self, key, order_ids):
        path = os.path.join(APP_DATA, "queue_explanations.csv.gz")
        col = f"reasons_{key}"
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, dtype={"order_id": str})
        if col not in df.columns:
            return None
        m = dict(zip(df["order_id"], df[col]))
        vals = [m.get(str(o)) for o in order_ids]
        return vals if all(v is not None and v == v for v in vals) else None

    def _explain_fast(self, model, q, top_n=3):
        import xgboost as xgb
        from prediction import _transformed_to_original, FEATURE_LABELS
        X = q[FEATURE_COLUMNS].copy()
        for c in NUMERIC:
            X[c] = X[c].astype(float)
        pre, clf = model.named_steps["pre"], model.named_steps["clf"]
        contribs = clf.get_booster().predict(xgb.DMatrix(pre.transform(X)), pred_contribs=True)[:, :-1]
        mapping = np.array([str(m) for m in _transformed_to_original(pre)])
        feats = list(dict.fromkeys(mapping))
        pos = {f: i for i, f in enumerate(feats)}
        agg = np.zeros((contribs.shape[0], len(feats)))
        for j, m in enumerate(mapping):
            agg[:, pos[m]] += contribs[:, j]
        out = []
        for row in agg:
            idx = np.argsort(-row)[:top_n]
            out.append([(feats[i], round(float(row[i]), 3), FEATURE_LABELS.get(feats[i], feats[i].replace("_", " "))) for i in idx if row[i] > 0])
        return out

    def set_view(self, requested):
        """Make every queue-based view (Overview, queue, bands, patterns, briefing, evaluation) follow `requested`."""
        req = requested if requested not in (None, "", "auto") else self.base_serving
        active = req if (req in self.models and req != "synthetic_operational" and f"p_{req}" in self._base_queue.columns) else self.base_serving
        if getattr(self, "_active_view", None) == (requested, active) and self.serving == active:
            return self.view
        if active not in self._view_cache:
            self._view_cache[active] = self._build_view(active)
        self.queue = self._view_cache[active]
        self.queue_index = {oid: i for i, oid in enumerate(self.queue["order_id"])}
        self.serving = active
        self.model, self.threshold = self.models[active] if active != self.base_serving else (self.base_model, self.base_threshold)
        self.view = self._view_info(requested, active)
        self._active_view = (requested, active)
        self.__dict__.pop("model_report", None)   # cached_property: rebuild for the new model
        return self.view

    # ---------- static content ----------
    @cached_property
    def ensemble_report(self):
        path = os.path.join(DASH, "ensemble_report.json")
        return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None

    @cached_property
    def literature(self):
        return json.load(open(os.path.join(APP_DATA, "literature.json"), encoding="utf-8"))

    @cached_property
    def model_report(self):
        m = json.load(open(os.path.join(DASH, "model_metrics.json"), encoding="utf-8"))
        fi = json.load(open(os.path.join(DASH, "feature_importance.json"), encoding="utf-8"))
        risk = json.load(open(os.path.join(DASH, "risk_summary.json"), encoding="utf-8"))
        if self.serving == "enriched" and self.enriched_meta:
            report_path = os.path.join(DASH, "verified_training_report.json")
            report = json.load(open(report_path, encoding="utf-8")) if os.path.exists(report_path) else {}
            split = report.get("split", {})
            test_metrics = self.enriched_meta.get("metrics_on_test", {})
            candidate_rows = report.get("candidates", {})
            pretty = {"xgboost_enriched": "XGBoost · enriched 70 features", "lightgbm_enriched": "LightGBM · enriched 70 features", "hist_gradient_enriched": "Histogram gradient boosting · enriched 70 features"}
            enriched_models = []
            for key in ("xgboost_enriched", "lightgbm_enriched", "hist_gradient_enriched"):
                row = candidate_rows.get(key, {}).get("test", {})
                if row:
                    enriched_models.append({"key": key, "label": pretty[key], "roc_auc": float(row.get("roc_auc", 0)), "pr_auc": float(row.get("pr_auc", 0)),
                        "precision": float(row.get("precision", 0)), "recall": float(row.get("recall", 0)), "f1": float(row.get("f1", 0)),
                        "recall_at_top10pct": float(row.get("recall_top10pct", 0))})
            enriched_models.sort(key=lambda r: r["roc_auc"], reverse=True)
            backtest_path = os.path.join(ROOT, "data", "processed", "rolling_backtest_enriched.csv")
            backtest = []
            if os.path.exists(backtest_path):
                bt = pd.read_csv(backtest_path)
                backtest = [{"month": str(r.month), "auc": float(r.roc_auc), "n": int(r.n_test), "late_rate": float(r.late_rate) * 100} for r in bt.itertuples()]
            imp_path = os.path.join(DASH, "enriched_feature_importance.json")
            enriched_fi = json.load(open(imp_path, encoding="utf-8"))[:12] if os.path.exists(imp_path) else fi[:12]
            return {
                "models": enriched_models, "selected": "xgboost_enriched", "threshold": float(self.enriched_meta.get("threshold", self.threshold)),
                "base_rate": float(split.get("test", {}).get("late_rate", 0.0529)),
                "n_train": int(split.get("train", {}).get("n", 62705)), "n_test": int(split.get("test", {}).get("n", 19294)),
                "n_validation": int(split.get("validation", {}).get("n", 14471)),
                "backtest": backtest, "backtest_mean_auc": float(np.mean([b["auc"] for b in backtest])) if backtest else None,
                "static_split_auc": float(test_metrics.get("roc_auc", 0.0)), "feature_importance": enriched_fi,
                "risk": self.risk_from_queue(), "serving": self.serving, "calibration": self.enriched_meta.get("calibration_report"),
                "ensemble": None, "models_info": self.models_info(), "agreement": self.agreement(),
                "library_versions": self.enriched_meta.get("library_versions", {}), "saved_at": self.enriched_meta.get("saved_at_utc"),
                "test_late_rate": float(split.get("test", {}).get("late_rate", 0.0529)),
                "paired_test_comparison": json.load(open(os.path.join(ROOT, "data", "processed", "paired_test_comparison.json"), encoding="utf-8")) if os.path.exists(os.path.join(ROOT, "data", "processed", "paired_test_comparison.json")) else None,
                "calibration_note": "Validation-only calibration shows temporal drift on the newest test period. Treat the model output as a risk ranking; recalibrate with recent labelled operational data before interpreting it as a literal probability.",
            }
        bt_path = os.path.join(ROOT, "data", "processed", "rolling_backtest_results.csv")
        backtest = []
        if os.path.exists(bt_path):
            bt = pd.read_csv(bt_path)
            backtest = [{"month": r.test_month, "auc": round(float(r.roc_auc), 3), "n": int(r.n_test),
                         "late_rate": round(float(r.delay_rate) * 100, 1)} for r in bt.itertuples()]
        models = [{"key": k, **{kk: round(float(vv), 4) for kk, vv in m[k].items()}}
                  for k in ["logistic_regression", "decision_tree", "random_forest", "random_forest_tuned", "xgboost", "xgboost_tuned"]
                  if k in m]
        return {
            "models": models, "selected": m.get("selected_model"), "threshold": self.threshold,
            "base_rate": m.get("base_delay_rate"), "n_train": m.get("n_train"), "n_test": m.get("n_test"),
            "backtest": backtest, "backtest_mean_auc": m.get("rolling_backtest_mean_auc"),
            "static_split_auc": m.get("static_time_split_auc_legacy"),
            "feature_importance": fi[:12], "risk": self.risk_from_queue(),
            "serving": self.serving,
            "calibration": (self.ensemble_meta or {}).get("calibration_report"), "ensemble": self.ensemble_report, "models_info": self.models_info(), "agreement": self.agreement(),
            "library_versions": self.model_meta.get("library_versions", {}),
            "saved_at": self.model_meta.get("saved_at_utc"),
        }

    def risk_from_queue(self):
        """Same summary as dashboard/risk_summary.json, computed from the queue the app is showing."""
        q = self.queue
        late = q.is_delayed == 1
        fl = q[q.flagged]
        bands = [{"level": b, "n": int((q.risk_level == b).sum()),
                  "late_rate": round(float(q[q.risk_level == b].is_delayed.mean()) * 100, 1) if (q.risk_level == b).any() else None}
                 for b in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]]
        drivers = fl.main_driver.value_counts()
        acts = fl.suggested_action.str.replace("URGENT: ", "", regex=False).value_counts()
        return {"bands": bands, "n_test": int(len(q)), "n_flagged": int(len(fl)), "flagged_share": round(len(fl) / len(q) * 100, 1),
                "flagged_late_rate": round(float(fl.is_delayed.mean()) * 100, 1) if len(fl) else 0.0,
                "late_orders_caught": round(float((q.flagged & late).sum() / max(late.sum(), 1)) * 100, 1),
                "base_late_rate": round(float(late.mean()) * 100, 1),
                "top_drivers": [{"driver": k, "n": int(v)} for k, v in drivers.head(8).items()],
                "actions": [{"action": k, "n": int(v)} for k, v in acts.head(8).items()],
                "n_urgent": int(fl.suggested_action.str.startswith("URGENT").sum())}

    # ---------- queue ----------
    COMPACT = ["order_id", "seller_id", "purchased_at", "promised_by", "product_category", "customer_state", "customer_city",
               "seller_state", "seller_customer_distance_km", "promised_delivery_days", "total_order_value",
               "probability", "risk_level", "flagged", "main_driver", "suggested_action", "pattern", "focus_tier", "priority"]

    def compact(self, row):
        d = {k: row[k] for k in self.COMPACT}
        d["flagged"] = bool(d["flagged"])
        d["pattern"] = int(d["pattern"])
        d["pattern_name"] = self.pattern_by_rank[d["pattern"]]["name"]
        for k in ("seller_customer_distance_km", "promised_delivery_days", "total_order_value", "probability", "priority"):
            d[k] = None if pd.isna(d[k]) else round(float(d[k]), 4 if k == "probability" else 1)
        for k in ("customer_city", "main_driver", "seller_id"):
            d[k] = "" if pd.isna(d[k]) else str(d[k])
        d["customer_city"] = d["customer_city"].title()
        return d

    def _comparison_models(self):
        """Return the serving model and the best available independent comparison model."""
        if self.serving == "enriched":
            other = "xgboost"
        elif self.serving == "xgboost":
            other = "ensemble" if "ensemble" in self.models else None
        else:
            other = "xgboost" if "xgboost" in self.models else None
        return self.serving, other

    def _disagreement_mask(self, frame):
        serving, other = self._comparison_models()
        a, b = f"flag_{serving}", f"flag_{other}" if other else None
        if other and a in frame.columns and b in frame.columns:
            return frame[a] != frame[b]
        return pd.Series(False, index=frame.index)

    def list_orders(self, band=None, state=None, category=None, q=None, flagged=None, sort="probability",
                    order="desc", offset=0, limit=50, pattern=None, focus=None, disagree=None):
        df = self.queue
        if disagree:
            df = df[self._disagreement_mask(df)]
        if pattern:
            df = df[df.pattern == int(pattern)]
        if focus:
            df = df[df.focus_tier != ""]
        if band:
            df = df[df.risk_level.isin([b.strip().upper() for b in band.split(",") if b.strip()])]
        if state:
            df = df[df.customer_state == state]
        if category:
            df = df[df.product_category == category]
        if flagged is not None:
            df = df[df.flagged == flagged]
        if q:
            ql = q.strip().lower()
            df = df[df.order_id.str.startswith(ql) | df.seller_id.fillna("").str.startswith(ql)
                    | df.customer_city.fillna("").str.lower().str.contains(ql, regex=False)]
        if sort not in {"probability", "priority", "total_order_value", "seller_customer_distance_km", "promised_delivery_days", "purchased_at"}:
            sort = "probability"
        df = df.sort_values(sort, ascending=(order == "asc"), na_position="last")
        total = len(df)
        page = df.iloc[offset: offset + limit]
        return {"total": int(total), "offset": offset, "limit": limit, "items": [self.compact(r) for _, r in page.iterrows()]}

    def get_order(self, order_id):
        i = self.queue_index.get(order_id)
        if i is None:
            return None
        row = self.queue.iloc[i]
        d = self.compact(row)
        labels = self.meta["feature_labels"]
        d["reasons"] = json.loads(row["reasons_json"])
        d["features"] = {c: (None if pd.isna(row[c]) else (row[c].item() if hasattr(row[c], "item") else row[c])) for c in FEATURE_COLUMNS}
        d["feature_labels"] = {c: labels.get(c, c.replace("_", " ")) for c in FEATURE_COLUMNS}
        d["confidence"] = None if pd.isna(row["confidence"]) else str(row["confidence"])
        d["seller_city"] = "" if pd.isna(row["seller_city"]) else str(row["seller_city"]).title()
        seller = self.lookup["seller"].get(d["seller_id"])
        d["seller_history"] = seller
        d["simulated"] = {k.replace("sim_", ""): (row[k].item() if hasattr(row[k], "item") else row[k])
                          for k in row.index if k.startswith("sim_")}
        d["customer_region"], d["seller_region"] = row["customer_region"], row["seller_region"]
        # What actually happened - never shown to the model, shown here so the
        # team can judge the alerts against reality.
        d["second_opinion"] = self.second_opinion(row)
        d["calibrated_probability"] = (float(row["p_calibrated"]) if self.serving in ("ensemble", "enriched") and "p_calibrated" in row.index and pd.notna(row["p_calibrated"]) else None)
        d["outcome"] = {
            "was_late": bool(row["is_delayed"]),
            "delivered_at": None if pd.isna(row["delivered_at"]) else str(row["delivered_at"]),
            "days_vs_promise": None if pd.isna(row["delay_days"]) else round(float(row["delay_days"]), 1),
        }
        return d

    def second_opinion(self, row):
        """What the other model says about this queue order (None when not available)."""
        other = "ensemble" if self.serving == "xgboost" else "xgboost"
        col = f"p_{other}"
        if other not in self.models or col not in row.index or pd.isna(row[col]):
            return None
        p = float(row[col])
        flag = bool(p >= self.models[other][1])
        return {"model": other, "label": MODEL_TITLES[other], "probability": round(p, 4), "risk_level": risk_bucket(p),
                "flagged": flag, "agrees": flag == bool(row["flagged"]),
                "primary_model": self.serving, "primary_label": MODEL_TITLES[self.serving]}

    def agreement(self):
        q = self.queue
        serving, other = self._comparison_models()
        a_col, b_col = f"flag_{serving}", f"flag_{other}" if other else None
        if not other or a_col not in q.columns or b_col not in q.columns:
            return None
        a, b = q[a_col].astype(bool), q[b_col].astype(bool)
        either = int((a | b).sum())
        # Preserve legacy counters for old clients, and expose neutral names for the current comparison.
        only_xgb = int((q["flag_xgboost"].astype(bool) & ~q["flag_ensemble"].astype(bool)).sum()) if "flag_xgboost" in q and "flag_ensemble" in q else int((b & ~a).sum())
        only_ensemble = int((~q["flag_xgboost"].astype(bool) & q["flag_ensemble"].astype(bool)).sum()) if "flag_xgboost" in q and "flag_ensemble" in q else int((a & ~b).sum())
        return {"both": int((a & b).sum()), "only_xgboost": only_xgb, "only_ensemble": only_ensemble,
                "serving_model": serving, "other_model": other,
                "serving_only": int((a & ~b).sum()), "other_only": int((b & ~a).sum()),
                "agree_share": round(float((a & b).sum() / max(either, 1)) * 100, 1), "n_disagree": int((a != b).sum())}

    def chance(self, p, key):
        """Estimated real chance of being late, calibrated on the validation block where available."""
        if key == "enriched" and self.enriched_meta:
            cr = self.enriched_meta.get("calibration_report", {})
            test_rate = float(self.enriched_meta.get("split", {}).get("test_late_rate", 0.0))
            base_brier = test_rate * (1.0 - test_rate)
            # Disable a literal chance estimate if it failed held-out calibration checks.
            if float(cr.get("test_ece", 1.0)) > 0.03 or float(cr.get("test_brier", 1.0)) > base_brier:
                return None
            cal = self.enriched_meta.get("calibration")
            return round(float(apply_calibration(cal, p)), 4) if cal else None
        if key == "ensemble" and self.calibration:
            return round(float(apply_calibration(self.calibration, p)), 4)
        return None

    def resolve_model(self, key=None):
        """(key, model, threshold) for 'xgboost' / 'ensemble' / None (= whatever scored the queue)."""
        if key in (None, "", "auto"):
            key = self.serving
        if key not in self.models:
            key = "xgboost"
        m, thr = self.models[key]
        return key, m, thr

    def models_info(self):
        out = [{"key": "xgboost", "label": MODEL_TITLES["xgboost"], "available": True, "threshold": self.models["xgboost"][1],
                "serving_queue": self.serving == "xgboost", "detail": "Tuned XGBoost, the original production model."}]
        e = self.ensemble_meta
        out.append({"key": "ensemble", "label": MODEL_TITLES["ensemble"], "available": "ensemble" in self.models,
                    "threshold": self.models["ensemble"][1] if "ensemble" in self.models else None,
                    "serving_queue": self.serving == "ensemble",
                    "detail": (e and "Members: " + ", ".join(f"{m['label']} ({self.ensemble_meta['weights'].get(m['name'], 0) * 100:.0f}%)"
                                                             for m in e["members"] if self.ensemble_meta["weights"].get(m["name"], 0) > 0))
                    or (self.ensemble_error or "Not trained yet. Run train_ensemble.py.")})
        out.append({"key": "enriched", "label": MODEL_TITLES["enriched"], "available": "enriched" in self.models,
                    "threshold": self.models["enriched"][1] if "enriched" in self.models else None,
                    "serving_queue": self.serving == "enriched",
                    "detail": ("70 checkout-time features; trained on real Olist data with chronological validation. "
                               "No synthetic weather/tracking inputs.") if "enriched" in self.models else (self.enriched_error or "Not trained yet.")})
        out.append({"key": "synthetic_operational", "label": MODEL_TITLES["synthetic_operational"],
                    "available": self.synthetic_model is not None, "threshold": 0.5 if self.synthetic_model is not None else None,
                    "serving_queue": False,
                    "benchmark_roc_auc": self.synthetic_metrics.get("heldout_chronological_test_roc_auc"),
                    "benchmark_pr_auc": self.synthetic_metrics.get("heldout_chronological_test_pr_auc"),
                    "benchmark_train_rows": self.synthetic_metrics.get("train_rows"),
                    "benchmark_validation_rows": self.synthetic_metrics.get("validation_rows"),
                    "benchmark_test_rows": self.synthetic_metrics.get("test_rows"),
                    "benchmark_type": "synthetic dispatch-time simulation",
                    "benchmark_top_features": self.synthetic_feature_importance,
                    "detail": (f"Dispatch-checkpoint model; synthetic held-out ROC-AUC {self.synthetic_metrics.get('heldout_chronological_test_roc_auc', 0):.3f}. Requires operational inputs; synthetic benchmark only, not real-world validation.") if self.synthetic_model is not None else (self.synthetic_model_error or "Synthetic operational model not available.")})
        return out

    def summary(self):
        q = self.queue
        counts = q.risk_level.value_counts().to_dict()
        return {
            "n_queue": int(len(q)), "n_flagged": int(q.flagged.sum()),
            "bands": {b: int(counts.get(b, 0)) for b in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
            "threshold": self.threshold,
            "risk": self.model_report["risk"],
            "monthly": self.insights["monthly"],
            "base_rate": self.insights["base_rate"],
            "n_orders": self.insights["n_orders"],
            "top_queue": [self.compact(r) for _, r in q.head(40).iterrows()],
        }

    # ---------- scoring ----------
    def complete_order(self, fields):
        """Build a full feature row from the form. Returns (row, filled, derived)."""
        typical = self.meta["typical_order"]
        row, filled = {}, []
        given = {k: v for k, v in (fields or {}).items() if v not in (None, "")}
        for c in required_input_columns():
            if c in given:
                row[c] = given[c]
            else:
                row[c] = typical.get(c)
                filled.append(c)
        row["seller_id"] = given.get("seller_id")
        for c in NUMERIC:
            if isinstance(row.get(c), str):
                n = _num(row[c])
                if n is not None:
                    row[c] = n  # otherwise leave the junk in place so the guardrail rejects it
        derived = []
        # keep the order internally consistent instead of mixing form values with medians
        n_items = _num(row.get("n_items")) or 1
        if "total_price" in given:
            tp = _num(row["total_price"])
            if tp is not None:
                for c in ("avg_item_price", "max_item_price"):  # equal-priced items assumed
                    if c not in given:
                        row[c] = round(tp / n_items, 2)
                        derived.append(c)
        if "total_freight" in given and "avg_freight" not in given:
            tf = _num(row["total_freight"])
            if tf is not None:
                row["avg_freight"] = round(tf / n_items, 2); derived.append("avg_freight")
        if "total_weight_g" in given and "avg_weight_g" not in given:
            tw = _num(row["total_weight_g"])
            if tw is not None:
                row["avg_weight_g"] = round(tw / n_items, 1); derived.append("avg_weight_g")
        tp, tf = _num(row.get("total_price")), _num(row.get("total_freight"))
        if tp is not None and tf is not None:
            if "total_order_value" not in given:
                row["total_order_value"] = round(tp + tf, 2); derived.append("total_order_value")
            if "total_payment_value" not in given:
                row["total_payment_value"] = round(tp + tf, 2); derived.append("total_payment_value")
        if "n_distinct_products" not in given:
            row["n_distinct_products"] = max(1.0, min(n_items, _num(row.get("n_distinct_products")) or 1)); derived.append("n_distinct_products")
        if "same_state" not in given and isinstance(row.get("customer_state"), str) and isinstance(row.get("seller_state"), str):
            row["same_state"] = 1.0 if row["customer_state"] == row["seller_state"] else 0.0; derived.append("same_state")
        dow = _num(row.get("purchase_dow"))
        if dow is not None and "is_weekend_purchase" not in given:
            row["is_weekend_purchase"] = 1.0 if dow >= 5 else 0.0; derived.append("is_weekend_purchase")
        filled = [c for c in filled if c not in derived]
        return row, filled, derived

    def _synthetic_operational_row(self, fields):
        """Build the dispatch model's exact feature schema from explicit operational inputs.

        If a dispatch feature is omitted, derive a documented estimate from the order form;
        the response explicitly lists every estimated/defaulted input.
        """
        f = dict(fields or {})
        def num(key, default):
            value = _num(f.get(key))
            return default if value is None else value
        n_items = max(1.0, num("n_items", 1.0))
        purchase_hour = num("purchase_hour", 15.0)
        purchase_dow = num("purchase_dow", 3.0)
        price_total = num("total_price", 86.9)
        freight_total = num("total_freight", 17.16)
        weight_total = num("total_weight_g", 750.0)
        promised_days = num("promised_delivery_days", 23.0)
        dispatch_lag = num("dispatch_lag_hours", max(4.0, num("approval_delay_hours", 0.5) + 8.0))
        values = {
            "route_km": num("route_km", num("seller_customer_distance_km", 434.0)),
            "weather_severity_forecast": num("weather_severity_forecast", 1.0),
            "congestion_forecast": num("congestion_forecast", 1.5),
            "carrier_reliability_prior": num("carrier_reliability_prior", 0.85),
            "capacity_utilization_at_dispatch": num("capacity_utilization_at_dispatch", 0.78),
            "remaining_sla_slack_hours": num("remaining_sla_slack_hours", promised_days * 24.0 - dispatch_lag),
            "facility_backlog_at_dispatch": num("facility_backlog_at_dispatch", 14.0),
            "route_disruption_alert_at_dispatch": num("route_disruption_alert_at_dispatch", 0.0),
            "last_scan_age_hours": num("last_scan_age_hours", 2.0),
            "estimated_truck_delay_hours_at_dispatch": num("estimated_truck_delay_hours_at_dispatch", 2.0),
            "hour_at_dispatch": num("hour_at_dispatch", (purchase_hour + int(dispatch_lag)) % 24),
            "weekday_at_dispatch": num("weekday_at_dispatch", (purchase_dow + int(dispatch_lag // 24)) % 7),
            "price": num("price", price_total / n_items),
            "freight_value": num("freight_value", freight_total / n_items),
            "product_weight_g": num("product_weight_g", weight_total / n_items),
            "dispatch_lag_hours": dispatch_lag,
        }
        # These fields have no trustworthy checkout-time proxy; when absent they are
        # estimates from documented defaults. Other absent fields are derived from order fields.
        operational_only = ["weather_severity_forecast", "congestion_forecast", "carrier_reliability_prior",
                            "capacity_utilization_at_dispatch", "facility_backlog_at_dispatch",
                            "route_disruption_alert_at_dispatch", "last_scan_age_hours",
                            "estimated_truck_delay_hours_at_dispatch", "dispatch_lag_hours"]
        defaults = [k for k in operational_only if f.get(k) in (None, "")]
        x = pd.DataFrame([[values[k] for k in SYNTHETIC_OPERATIONAL_FEATURES]], columns=SYNTHETIC_OPERATIONAL_FEATURES)
        return x, defaults

    def _score_synthetic_operational(self, fields):
        if self.synthetic_model is None:
            raise ValueError(f"Synthetic operational model unavailable: {self.synthetic_model_error or 'model artifact missing'}")
        fields = dict(fields or {})
        bounds = {
            "seller_customer_distance_km": (0, 5000), "route_km": (0, 5000),
            "promised_delivery_days": (0, 365), "total_price": (0, 1e8), "price": (0, 1e8),
            "total_freight": (0, 1e8), "freight_value": (0, 1e8), "total_weight_g": (0, 1e8),
            "product_weight_g": (0, 1e8), "weather_severity_forecast": (0, 5), "congestion_forecast": (0, 5),
            "carrier_reliability_prior": (0, 1), "capacity_utilization_at_dispatch": (0, 3),
            "remaining_sla_slack_hours": (-720, 10000), "facility_backlog_at_dispatch": (0, 1e7),
            "route_disruption_alert_at_dispatch": (0, 1), "last_scan_age_hours": (0, 10000),
            "estimated_truck_delay_hours_at_dispatch": (0, 10000), "dispatch_lag_hours": (0, 10000),
            "hour_at_dispatch": (0, 23), "weekday_at_dispatch": (0, 6),
        }
        reject = []
        for key, (lo, hi) in bounds.items():
            if key in fields and fields[key] not in (None, ""):
                value = _num(fields[key])
                if value is None:
                    reject.append(f"{key} must be numeric")
                elif not lo <= value <= hi:
                    reject.append(f"{key} must be between {lo} and {hi}")
        if reject:
            return {"status": "REJECTED", "probability": None, "risk_level": None, "flagged": None,
                    "confidence": "LOW_CONFIDENCE", "suggested_action": "Correct the input before scoring.",
                    "warnings": [], "notes": [], "reject_reasons": reject, "filled_from_typical": [],
                    "derived": [], "threshold": 0.5, "model": "synthetic_operational",
                    "model_label": MODEL_TITLES["synthetic_operational"], "calibrated_probability": None,
                    "reasons": [], "rates_used": {"seller_historical_delay_rate": None, "category_historical_delay_rate": None, "customer_state_historical_delay_rate": None},
                    "pattern": None, "simulated": None}
        import xgboost as xgb
        x, estimated = self._synthetic_operational_row(fields)
        arr = x.to_numpy(dtype=float)
        dmat = xgb.DMatrix(arr, feature_names=SYNTHETIC_OPERATIONAL_FEATURES)
        probability = float(self.synthetic_model.predict(dmat)[0])
        probability = min(1.0, max(0.0, probability))
        threshold = 0.5
        level = "LOW" if probability < .3 else "MEDIUM" if probability < .6 else "HIGH" if probability < .8 else "CRITICAL"
        contrib = self.synthetic_model.predict(dmat, pred_contribs=True)[0][:-1]
        reasons = []
        labels = {
            "route_km": "Long delivery route", "weather_severity_forecast": "Forecast weather severity",
            "congestion_forecast": "Forecast traffic congestion", "carrier_reliability_prior": "Carrier reliability prior",
            "capacity_utilization_at_dispatch": "Dispatch capacity utilisation", "remaining_sla_slack_hours": "Remaining delivery SLA slack",
            "facility_backlog_at_dispatch": "Facility backlog", "route_disruption_alert_at_dispatch": "Route disruption alert",
            "last_scan_age_hours": "Age of last tracking scan", "estimated_truck_delay_hours_at_dispatch": "Estimated truck delay",
            "hour_at_dispatch": "Dispatch hour", "weekday_at_dispatch": "Dispatch weekday", "price": "Item price",
            "freight_value": "Freight value", "product_weight_g": "Product weight", "dispatch_lag_hours": "Purchase-to-dispatch lag",
        }
        for i in np.argsort(contrib)[::-1]:
            if contrib[i] > 0 and len(reasons) < 5:
                reasons.append({"feature": SYNTHETIC_OPERATIONAL_FEATURES[i], "label": labels[SYNTHETIC_OPERATIONAL_FEATURES[i]], "contribution": round(float(contrib[i]), 3)})
        notes = ["Dispatch-time operational model trained on simulated labels; its benchmark AUC is not a real-world performance estimate."]
        if estimated:
            notes.append("Estimated/defaulted dispatch inputs: " + ", ".join(estimated) + ". Supply actual dispatch-time values for a meaningful operational score.")
        return {"status": "SCORED", "probability": probability, "risk_level": level, "flagged": bool(probability >= threshold),
                "confidence": "SYNTHETIC_BENCHMARK", "suggested_action": ("URGENT: Check carrier, route disruption and remaining SLA; escalate to dispatch lead." if probability >= .8 else "Review dispatch capacity, route conditions and remaining SLA." if probability >= .5 else "Continue monitoring; refresh operational inputs at the next scan."),
                "warnings": ["Synthetic benchmark model: do not interpret the score as a calibrated real-world probability."], "notes": notes,
                "reject_reasons": [], "filled_from_typical": [], "derived": estimated, "threshold": threshold,
                "model": "synthetic_operational", "model_label": MODEL_TITLES["synthetic_operational"], "calibrated_probability": None,
                "reasons": reasons, "rates_used": {"seller_historical_delay_rate": None, "category_historical_delay_rate": None, "customer_state_historical_delay_rate": None},
                "pattern": None, "simulated": None}

    def score_one(self, fields, model_key=None):
        key, model, threshold = self.resolve_model(model_key)
        if key == "synthetic_operational":
            return self._score_synthetic_operational(fields)
        row, filled, derived = self.complete_order(fields)
        df = pd.DataFrame([row])
        scored = score_orders(df, model, threshold, self.stats, self.lookup, top_n=5)
        r = scored.iloc[0]
        result = {
            "status": r["status"],
            "probability": None if pd.isna(r["probability"]) else float(r["probability"]),
            "risk_level": r["risk_level"], "flagged": None if r["flagged"] is None else bool(r["flagged"]),
            "confidence": r["confidence"], "suggested_action": r["suggested_action"],
            "warnings": [w for w in str(r["warnings"]).split(" | ") if w],
            "notes": [w for w in str(r["notes"]).split(" | ") if w],
            "reject_reasons": [w for w in str(r["reject_reasons"]).split(" | ") if w],
            "filled_from_typical": filled, "derived": derived, "threshold": threshold,
            "model": key, "model_label": MODEL_TITLES[key],
        }
        result["calibrated_probability"] = self.chance(result["probability"], key) if result["probability"] is not None else None
        if result["status"] == "SCORED":
            # explain every scored order (not just flagged ones) so the simulator
            # shows what moves the score either way
            from prediction import explain_orders
            from scoring import fill_rate_features
            filled_df, _, _ = fill_rate_features(df, self.lookup)
            X = filled_df[FEATURE_COLUMNS].copy()
            for c in NUMERIC:
                X[c] = X[c].astype(float)
            if getattr(model, "extra_columns", None):
                extras, _extra_warns = model.serving_extras(filled_df)
                X = pd.concat([X, extras], axis=1)
                why = model.explain(X, top_n=5)[0]
            else:
                why = explain_orders(model, X, top_n=5)[0]
            result["reasons"] = [{"feature": f, "label": lab, "contribution": c} for f, c, lab in why]
            result["rates_used"] = {k: round(float(filled_df[k].iloc[0]), 4) for k in RATE_FEATURES}
            # which known late-delivery pattern does this order resemble?
            from clustering import assign
            from simulation import region, simulate_logistics
            pdf = filled_df.copy()
            pdf["customer_region"] = pdf["customer_state"].map(region)
            for c in ("purchase_month", "total_weight_g", "total_freight", "total_order_value", "seller_customer_distance_km",
                      "approval_delay_hours", "promised_delivery_days", "seller_historical_delay_rate", "same_state"):
                pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
            rank = self.patterns["id_to_rank"][str(int(assign(pdf, self.patterns["params"])[0]))]
            pat = self.pattern_by_rank[rank]
            result["pattern"] = {"rank": rank, "name": pat["name"], "recommendation": pat["recommendations"][0]}
            pdf["order_id"] = "simulator"
            sim = simulate_logistics(pdf, seller_col="seller_id").iloc[0]
            result["simulated"] = {k.replace("sim_", ""): (v.item() if hasattr(v, "item") else v) for k, v in sim.items()}
        else:
            result["reasons"] = []
        return result

    def score_batch(self, df, model_key=None):
        key, model, threshold = self.resolve_model(model_key)
        if key == "synthetic_operational":
            from scoring import required_input_columns
            expected_order = set(required_input_columns())
            expected_dispatch = set(SYNTHETIC_OPERATIONAL_FEATURES)
            if not (expected_order.issubset(set(df.columns)) or expected_dispatch.issubset(set(df.columns))):
                missing_order = sorted(expected_order - set(df.columns))
                missing_dispatch = sorted(expected_dispatch - set(df.columns))
                raise ValueError("CSV is missing required order or dispatch feature columns. Missing order columns: " + ", ".join(missing_order[:12]) + "; missing dispatch columns: " + ", ".join(missing_dispatch[:12]))
            rows = []
            for _, row in df.iterrows():
                fields = row.to_dict()
                # Batch files commonly use Olist/application names rather than dispatch feature names.
                alias = {"seller_customer_distance_km": "seller_customer_distance_km", "promised_delivery_days": "promised_delivery_days",
                         "purchase_hour": "purchase_hour", "purchase_dow": "purchase_dow", "total_price": "total_price",
                         "total_freight": "total_freight", "total_weight_g": "total_weight_g"}
                for source, target in alias.items():
                    if target not in fields and source in row.index: fields[target] = row[source]
                result = self._score_synthetic_operational(fields)
                rows.append({"order_id": fields.get("order_id"), "status": result["status"], "probability": result["probability"],
                             "risk_level": result["risk_level"], "flagged": result["flagged"], "confidence": result["confidence"],
                             "suggested_action": result["suggested_action"], "warnings": " | ".join(result["warnings"]),
                             "notes": " | ".join(result["notes"]), "reasons": "; ".join(f"{r['label']} (+{r['contribution']})" for r in result["reasons"]),
                             "reject_reasons": "", "threshold": result["threshold"], "model": key})
            return pd.DataFrame(rows)
        return score_orders(df, model, threshold, self.stats, self.lookup)


class EngineUnavailable(RuntimeError):
    """The model or app data could not be loaded; the message says why and what to do."""


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        try:
            _engine = Engine()
        except FileNotFoundError as e:
            raise EngineUnavailable(
                f"A required file is missing: {e.filename or e}. Re-extract the OnTime zip "
                "into a fresh folder (don't copy individual files) and start again.") from e
        except Exception as e:  # noqa: BLE001
            raise EngineUnavailable(
                f"The model could not be loaded on this computer ({e.__class__.__name__}: {e}). "
                "Run setup.bat again, then restart start_ontime.bat.") from e
    _sync_view(_engine)
    return _engine


def _sync_view(eng):
    """Keep every queue view on the model chosen in Settings (cheap no-op when nothing changed)."""
    try:
        from . import assistant as _ai
        eng.set_view(_ai.load_settings().get("scoring_model"))
    except Exception as ex:  # noqa: BLE001 - a settings problem must never take the app down
        print(f"  [OnTime] model view not switched: {ex.__class__.__name__}: {ex}", flush=True)


# ---------------------------------------------------------------- briefings & planning
def _period_label(m):
    import calendar
    y, mo = m.split("-")
    return f"{calendar.month_name[int(mo)]} {y}"


def briefing_periods(eng, min_orders=150):
    vc = eng.queue["month"].value_counts()
    return sorted([m for m, n in vc.items() if n >= min_orders], reverse=True)


def build_briefing(eng, period):
    from narrative import briefing_story
    q = eng.queue
    periods = briefing_periods(eng)
    if period not in periods:
        raise KeyError(period)
    g = q[q.month == period]
    fl = g[g.flagged]

    def stats(df):
        return {"n_orders": int(len(df)), "n_flagged": int(df.flagged.sum()),
                "flagged_share": round(float(df.flagged.mean()) * 100, 1) if len(df) else 0.0}
    b = {"period": period, "period_label": _period_label(period), **stats(g),
         "n_critical": int((g.risk_level == "CRITICAL").sum())}
    i = periods.index(period)
    if i + 1 < len(periods):
        prev = q[q.month == periods[i + 1]]
        b["prev"] = {"period": periods[i + 1], "period_label": _period_label(periods[i + 1]), **stats(prev)}
    pc = fl.pattern.value_counts()
    b["top_patterns"] = [{"rank": int(r), "name": eng.pattern_by_rank[int(r)]["name"], "n": int(n),
                          "recommendation": eng.pattern_by_rank[int(r)]["recommendations"][0]} for r, n in pc.head(3).items()]
    b["top_drivers"] = [{"driver": k, "n": int(v)} for k, v in fl.main_driver.value_counts().head(5).items()]
    focus = []
    for key, c in eng.categories["focus"].items():
        gg = g[g.product_category == key]
        if len(gg):
            focus.append({"key": key, "label": c["label"], "tier": c["tier"], "n": int(len(gg)), "alerts": int(gg.flagged.sum()),
                          "flagged_share": round(float(gg.flagged.mean()) * 100, 1)})
    b["focus"] = sorted(focus, key=lambda r: -r["flagged_share"])
    late = g.is_delayed == 1
    b["outcome"] = {"flagged_late_rate": round(float(fl.is_delayed.mean()) * 100, 1) if len(fl) else 0.0,
                    "caught": round(float((g.flagged & late).sum() / max(late.sum(), 1)) * 100, 1),
                    "late_rate": round(float(late.mean()) * 100, 1)}
    actions = [p["recommendation"] for p in b["top_patterns"][:2]]
    if b["focus"]:
        w = b["focus"][0]
        actions.append(f"{w['label']}: {eng.categories['focus'][w['key']]['playbook'][0]}")
    b["actions"] = actions
    b["critical_orders"] = [eng.compact(r) for _, r in g[g.risk_level == "CRITICAL"].head(8).iterrows()]
    b["story"] = briefing_story(b)
    b["note"] = ("Built from the held-out test sample (about 20% of all orders), so volumes are a sample of the month; "
                 "rates and patterns are representative.")
    return b


def build_planner(eng, cost_check=15.0, cost_late=120.0, effectiveness=0.5):
    """Expected cost of running OnTime at each alert threshold.
    cost_check:    cost of a coordinator checking one alerted order (R$)
    cost_late:     cost of one late delivery (refund, support, lost repeat business) (R$)
    effectiveness: share of truly-late alerted orders that a check actually rescues
    Uses the held-out queue, where the real outcome is known."""
    p = eng.queue["probability"].values
    y = eng.queue["is_delayed"].values.astype(bool)
    n, n_late = len(p), int(y.sum())
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        a = p >= t
        tp = int((a & y).sum())
        alerts = int(a.sum())
        saving = tp * cost_late * effectiveness - alerts * cost_check
        rows.append({"threshold": float(t), "alerts": alerts, "alert_share": round(alerts / n * 100, 2),
                     "precision": round(tp / alerts * 100, 1) if alerts else None,
                     "recall": round(tp / n_late * 100, 1), "late_caught": tp, "net_saving": round(float(saving), 0)})
    best = max(rows, key=lambda r: r["net_saving"])
    cur = min(rows, key=lambda r: abs(r["threshold"] - eng.threshold))
    return {"rows": rows, "best": best, "current": cur, "n_orders": n, "n_late": n_late,
            "baseline_cost": round(n_late * cost_late, 0),
            "inputs": {"cost_check": cost_check, "cost_late": cost_late, "effectiveness": effectiveness}}
