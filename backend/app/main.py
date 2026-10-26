"""
OnTime web app - FastAPI backend that also serves the frontend.

    python run.py            -> http://localhost:8000
    API docs (Swagger)       -> http://localhost:8000/docs
"""
import os
import math
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .engine import ROOT, FORM_FIELDS, EngineUnavailable, get_engine, briefing_periods, build_briefing, build_planner
from . import assistant as ai
from . import handoff as hand
from . import tracking as trk
from . import weather as wx
from . import adapt_service as adapt
from . import monitoring as mon
from .store import BoardStore, STAGES, STAGE_LABELS, STAGE_INFO, SUPABASE_SCHEMA_SQL

FRONTEND = os.path.join(ROOT, "frontend")
DB_PATH = os.environ.get("ONTIME_DB", os.path.join(ROOT, "backend", "ontime.db"))
MAX_BATCH_ROWS = int(os.environ.get("ONTIME_MAX_BATCH_ROWS", "5000"))


def _json_safe(value):
    """Convert NumPy/pandas scalars and non-finite numbers to strict JSON values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


class SafeJSONResponse(JSONResponse):
    """Strict JSON response: never emit invalid NaN/Infinity tokens to clients."""
    def render(self, content):
        return super().render(_json_safe(content))

@asynccontextmanager
async def lifespan(_app):
    # Load the model once at startup, so a one-off local rebuild (if this
    # machine can't read the saved model) happens before the first page,
    # with its progress shown in the console window.
    try:
        e = get_engine()
        print(f"  [OnTime] Model ready - {len(e.queue):,} orders in the queue.", flush=True)
    except EngineUnavailable as ex:
        print(f"  [OnTime] PROBLEM: {ex}", flush=True)
    yield


app = FastAPI(title="OnTime API", version="2.1", lifespan=lifespan, default_response_class=SafeJSONResponse,
              description="Delivery delay early warning: scored order queue, action board, single and batch scoring.")


@app.exception_handler(EngineUnavailable)
async def engine_unavailable(_request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})
_board = None


def board():
    global _board
    if _board is None or _board.path != DB_PATH:
        _board = BoardStore(DB_PATH)
    return _board


def set_db_path(path):
    """Used by the tests to point the board at a throwaway database."""
    global DB_PATH, _board
    DB_PATH, _board = path, None


# ------------------------------------------------------------------ meta
def _effective_live_scoring_model(e=None):
    e = e or get_engine()
    key = ai.load_settings().get("scoring_model", "enriched") or "enriched"
    return e.serving if key in ("", "auto") else key


@app.get("/api/health")
def health():
    e = get_engine()
    saved = e.enriched_meta.get("saved_at_utc") if e.serving == "enriched" and e.enriched_meta else e.model_meta.get("saved_at_utc")
    live_key = _effective_live_scoring_model(e)
    live_info = next((m for m in e.models_info() if m["key"] == live_key), None)
    return {"status": "ok", "orders_in_queue": int(len(e.queue)), "model_saved_at": saved,
            "threshold": e.threshold, "serving_model": e.serving,
            "live_scoring_model": live_key, "live_scoring_label": (live_info or {}).get("label", live_key),
            "live_scoring_available": bool((live_info or {}).get("available", False))}


@app.get("/api/health/diagnostics")
def health_diagnostics():
    """Report local readiness separately from external connectivity and real-world validation."""
    e = get_engine()
    live_key = _effective_live_scoring_model(e)
    live_info = next((m for m in e.models_info() if m["key"] == live_key), None)
    frontend_ok = os.path.isfile(os.path.join(FRONTEND, "index.html")) and os.path.isfile(os.path.join(FRONTEND, "assets", "app.js"))
    synthetic = live_key == "synthetic_operational"
    return {
        "status": "ready" if bool((live_info or {}).get("available", False)) and frontend_ok and len(e.queue) > 0 else "degraded",
        "checks": {
            "api": {"status": "ready"},
            "frontend": {"status": "ready" if frontend_ok else "missing_required_files"},
            "historical_queue": {"status": "ready" if len(e.queue) else "empty", "orders": int(len(e.queue)), "source": "prepared historical Olist-derived queue"},
            "live_scoring_model": {"status": "ready" if bool((live_info or {}).get("available", False)) else "unavailable", "key": live_key, "label": (live_info or {}).get("label", live_key)},
            "benchmark_provenance": {"status": "simulated" if synthetic else "see_model_metadata", "note": "Synthetic dispatch benchmark is not real-world evidence." if synthetic else "Historical evaluation and live scoring are separate workflows."},
            "weather_provider": {"status": "not_checked", "check_endpoint": "/api/weather/forecast?latitude=-23.55&longitude=-46.63", "note": "External connectivity is not tested by this local readiness endpoint."},
            "real_world_validation": {"status": "not_configured", "required_evidence": ["predictions made before outcomes were known", "observed delivery outcomes keyed by order_id", "prediction timestamp scored_at"], "tool": "scripts/validate_operational_outcomes.py"},
        },
        "note": "Local readiness is not the same as external-service availability or real-world model validation.",
    }


@app.get("/api/meta")
def meta():
    e = get_engine()
    live_key = _effective_live_scoring_model(e)
    live_info = next((m for m in e.models_info() if m["key"] == live_key), None)
    return {**e.meta, "serving": e.serving, "queue_model": e.serving, "base_serving": e.base_serving, "queue_model_label": e.view["label"], "queue_view": e.view,
            "live_scoring_model": live_key,
            "live_scoring_label": (live_info or {}).get("label", live_key),
            "live_scoring_available": bool((live_info or {}).get("available", False)),
            "ensemble_available": "ensemble" in e.models, "enriched_available": "enriched" in e.models,
            "synthetic_operational_available": "synthetic_operational" in e.models,
            "form_fields": FORM_FIELDS, "stages": [{"key": s, "label": STAGE_LABELS[s]} for s in STAGES]}


@app.get("/api/summary")
def summary():
    e = get_engine()
    s = e.summary()
    on_board = board().ids()
    s["board"] = board().counts()
    s["needs_attention"] = [o for o in s.pop("top_queue") if o["order_id"] not in on_board][:8]
    return s


# ------------------------------------------------------------------ queue
@app.get("/api/orders")
def orders(band: Optional[str] = None, state: Optional[str] = None, category: Optional[str] = None,
           q: Optional[str] = None, flagged: Optional[bool] = None, sort: str = "probability",
           pattern: Optional[int] = None, focus: Optional[bool] = None, disagree: Optional[bool] = None,
           order: str = Query("desc", pattern="^(asc|desc)$"),
           offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    res = get_engine().list_orders(band, state, category, q, flagged, sort, order, offset, limit, pattern, focus, disagree)
    on_board = board().ids()
    for it in res["items"]:
        it["on_board"] = it["order_id"] in on_board
    return res


@app.get("/api/orders/{order_id}")
def order_detail(order_id: str):
    d = get_engine().get_order(order_id)
    if d is None:
        raise HTTPException(404, f"No order {order_id!r} in the queue.")
    d["board"] = board().get(order_id)
    return d


@app.get("/api/orders/{order_id}/story")
def order_story(order_id: str):
    from narrative import order_story as story, customer_message
    e = get_engine()
    o = e.get_order(order_id)
    if o is None:
        raise HTTPException(404, f"No order {order_id!r} in the queue.")
    pat = e.pattern_by_rank.get(o["pattern"])
    cat = e.categories["focus"].get(o["product_category"])
    return {"order_id": order_id, "paragraphs": story(o, pat if o["flagged"] else None, cat),
            "customer_message": customer_message(o)}


# ------------------------------------------------------------------ patterns, categories, network
@app.get("/api/patterns")
def patterns():
    e = get_engine()
    counts = e.queue[e.queue.flagged].pattern.value_counts().to_dict()
    pats = [{**{k: p[k] for k in ("rank", "name", "n", "share_of_late", "median_distance_km", "median_promised_days",
                                  "median_actual_days", "median_days_late", "peak_share", "distinguishing", "regions",
                                  "states", "categories", "recommendations", "logistics", "z")},
             "queue_alerts": int(counts.get(p["rank"], 0))} for p in e.patterns["patterns"]]
    return {"n_late": e.patterns["n_late"], "k": e.patterns["k"], "silhouette_by_k": e.patterns["silhouette_by_k"], "patterns": pats}


@app.get("/api/patterns/{rank}")
def pattern_detail(rank: int):
    from narrative import pattern_story
    e = get_engine()
    p = e.pattern_by_rank.get(rank)
    if p is None:
        raise HTTPException(404, f"No pattern {rank}.")
    qq = e.queue[(e.queue.pattern == rank) & e.queue.flagged]
    return {**p, "story": pattern_story(p, e.patterns["n_late"]), "queue_alerts": int(len(qq)),
            "orders": [e.compact(r) for _, r in qq.head(10).iterrows()]}


@app.get("/api/categories")
def categories():
    e = get_engine()
    out = []
    for key, c in e.categories["focus"].items():
        qq = e.queue[e.queue.product_category == key]
        out.append({**{k: c[k] for k in ("key", "label", "tier", "weight", "why", "impact", "playbook")},
                    "stats": c["stats"], "queue": {"orders": int(len(qq)), "alerts": int(qq.flagged.sum()),
                                                   "critical": int((qq.risk_level == "CRITICAL").sum())}})
    return {"focus": out, "roadmap_note": e.categories["roadmap_note"], "patterns": {
        p["rank"]: p["name"] for p in e.patterns["patterns"]}}


@app.get("/api/categories/{key}")
def category_detail(key: str):
    from narrative import category_story
    e = get_engine()
    c = e.categories["focus"].get(key)
    if c is None:
        raise HTTPException(404, f"{key!r} is not one of the focus categories.")
    qq = e.queue[(e.queue.product_category == key) & e.queue.flagged]
    return {**c, "story": category_story(key, c, c["stats"]), "orders": [e.compact(r) for _, r in qq.head(10).iterrows()]}


@app.get("/api/network")
def network():
    return get_engine().network


# ------------------------------------------------------------------ briefing & planner
@app.get("/api/briefing/periods")
def periods():
    from .engine import _period_label
    return {"periods": [{"key": m, "label": _period_label(m)} for m in briefing_periods(get_engine())]}


@app.get("/api/briefing")
def briefing(period: Optional[str] = None):
    e = get_engine()
    ps = briefing_periods(e)
    try:
        return build_briefing(e, period or ps[0])
    except KeyError:
        raise HTTPException(404, f"No briefing for {period!r}. Available: {', '.join(ps)}")


@app.get("/api/planner")
def planner(cost_check: float = Query(15.0, ge=0, le=10000), cost_late: float = Query(120.0, ge=0, le=100000),
            effectiveness: float = Query(0.5, ge=0, le=1)):
    return build_planner(get_engine(), cost_check, cost_late, effectiveness)



# ------------------------------------------------------------------ database (SQLite / Supabase)
@app.get("/api/database/status")
def database_status():
    return {
        **board().status(),
        "schema_sql": SUPABASE_SCHEMA_SQL
    }


class DbTestRequest(BaseModel):
    url: Optional[str] = None
    key: Optional[str] = None


@app.post("/api/database/test")
def database_test(req: Optional[DbTestRequest] = None):
    url = (req.url if req and req.url else None) or os.environ.get("SUPABASE_URL", "")
    key = (req.key if req and req.key else None) or os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        s = ai.load_settings()
        url = url or s.get("supabase_url", "")
        key = key or s.get("supabase_key", "")
    if not url or not key:
        return {"ok": False, "mode": "sqlite", "detail": "No Supabase credentials provided. Running in local SQLite mode."}
    from .store import SupabaseBackend
    sb = SupabaseBackend(url, key)
    return sb.test_connection()

# ------------------------------------------------------------------ assistant & settings
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    engine: Optional[str] = None
    context: Optional[dict] = None      # what the previous answer was about (order, pattern, state)
    history: Optional[list] = None      # last few {role, text} turns, for the AI engines


class RewriteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=6000)
    style: str = "executive"
    engine: Optional[str] = None


class SettingsPatch(BaseModel):
    engine: Optional[str] = None
    timeout_s: Optional[float] = None
    ollama_url: Optional[str] = None
    ollama_model: Optional[str] = None
    gemini_model: Optional[str] = None
    gemini_api_key: Optional[str] = None
    scoring_model: Optional[str] = None
    database_mode: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None


@app.get("/api/settings")
def get_settings():
    return ai.public_settings(ai.load_settings())


@app.post("/api/settings")
def set_settings(req: SettingsPatch):
    global _board
    try:
        s = ai.save_settings({k: v for k, v in req.model_dump().items() if v is not None})
        _board = None  # Re-initialize board store with new database configuration
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    return ai.public_settings(s)


@app.get("/api/assistant/status")
def assistant_status():
    s = ai.load_settings()
    return {"settings": ai.public_settings(s), "engines": ai.engine_status(s), "examples": ai.EXAMPLES}


@app.post("/api/assistant/ask")
def assistant_ask(req: AskRequest):
    if req.engine and req.engine not in ai.ENGINES:
        raise HTTPException(400, f"Unknown engine '{req.engine}'.")
    return ai.ask(req.question, get_engine(), board().counts(), engine=req.engine,
                  context=req.context, history=(req.history or [])[-6:])


@app.post("/api/assistant/rewrite")
def assistant_rewrite(req: RewriteRequest):
    return ai.rewrite(req.text, req.style, engine=req.engine)


@app.post("/api/assistant/test")
def assistant_test(engine: str = Query(..., pattern="^(laptop|gemini)$")):
    return ai.test_engine(engine)


@app.get("/api/sellers/{seller_id}")
def seller(seller_id: str):
    e = get_engine()
    hist = e.lookup["seller"].get(seller_id)
    q = e.queue[e.queue.seller_id == seller_id]
    if hist is None and q.empty:
        raise HTTPException(404, f"No seller {seller_id!r} in the history or the queue.")
    return {"seller_id": seller_id, "history": hist, "platform_rate": e.lookup["global_rate"],
            "queue": [e.compact(r) for _, r in q.head(50).iterrows()], "n_in_queue": int(len(q))}


# ------------------------------------------------------------------ scoring
MAX_SCORE_FIELDS = 200
MAX_BATCH_BYTES = int(os.environ.get("ONTIME_MAX_BATCH_BYTES", str(10 * 1024 * 1024)))

_PROVENANCE = {
    "xgboost": ("checkout", "Historical Olist orders (real)", "ranking score; see calibrated_probability for a probability estimate"),
    "ensemble": ("checkout", "Historical Olist orders (real)", "ranking score; see calibrated_probability for a probability estimate"),
    "enriched": ("checkout", "Historical Olist orders (real)", "ranking score; see calibrated_probability for a probability estimate"),
    "synthetic_operational": ("dispatch", "Olist-adapted GENERATED operational dataset (simulated dispatch signals and labels)",
                              "ranking score from a synthetic benchmark; not a measured probability on real deliveries"),
}


def _provenance(model_key):
    """Where a score comes from, attached to every scoring response."""
    key = model_key if model_key in _PROVENANCE else "xgboost"
    stage, data, meaning = _PROVENANCE[key]
    return {"model": key, "stage": stage, "data_origin": data, "score_meaning": meaning,
            "synthetic": key == "synthetic_operational"}


def _effective_model_key(requested):
    """The model key that will really score a request (handles 'auto' and the dispatch model)."""
    key = _scoring_model(requested)
    if key in (None, "", "auto"):
        key = get_engine().serving
    return key


async def _read_limited(file, limit, what="file"):
    """Read an upload but refuse anything above `limit` bytes instead of loading it all."""
    raw = await file.read(limit + 1)
    if len(raw) > limit:
        raise HTTPException(413, f"The {what} is larger than {limit // (1024 * 1024)} MB. Split it into smaller files.")
    return raw


class ScoreRequest(BaseModel):
    fields: dict = Field(default_factory=dict, description="Any subset of the order fields; the rest come from a typical order.")
    model: Optional[str] = Field(None, description="'xgboost' or 'ensemble'; default is the model chosen in Settings.")

    @field_validator("fields")
    @classmethod
    def _limit_fields(cls, v):
        if len(v) > MAX_SCORE_FIELDS:
            raise ValueError(f"Too many fields ({len(v)}); the limit is {MAX_SCORE_FIELDS}.")
        for k, val in v.items():
            if len(str(k)) > 80:
                raise ValueError("A field name is longer than 80 characters.")
            if isinstance(val, (dict, list)):
                raise ValueError(f"Field '{k}' must be a single value, not a nested object or list.")
        return v


def _scoring_model(requested):
    if requested and requested not in ("auto", "xgboost", "ensemble", "enriched", "synthetic_operational"):
        raise HTTPException(400, f"Unknown model '{requested}'. Use xgboost, ensemble, enriched, or synthetic_operational.")
    return requested or ai.load_settings().get("scoring_model", "auto")


@app.post("/api/score")
def score(req: ScoreRequest):
    result = get_engine().score_one(req.fields, _scoring_model(req.model))
    result["order_id"] = req.fields.get("order_id")
    result["provenance"] = _provenance(result.get("model") or _effective_model_key(req.model))
    result["scored_at"] = datetime.now(timezone.utc).isoformat()
    return result


@app.get("/api/template.csv", response_class=PlainTextResponse)
def template():
    e = get_engine()
    from scoring import required_input_columns
    cols = ["order_id", "seller_id"] + required_input_columns()
    # a realistic mix: a few alerts plus ordinary orders
    risky = e.queue[e.queue.flagged].sample(4, random_state=7)
    rest = e.queue[~e.queue.flagged].sample(6, random_state=7)
    sample = pd.concat([risky, rest]).sample(frac=1, random_state=1)[cols].copy()
    # Explicitly synthetic dispatch examples let the template exercise either model.
    n = len(sample)
    sample["route_km"] = pd.to_numeric(sample.get("seller_customer_distance_km", 434), errors="coerce").fillna(434)
    sample["weather_severity_forecast"] = [4.2 if i < 4 else 0.8 for i in range(n)]
    sample["congestion_forecast"] = [4.0 if i < 4 else 1.0 for i in range(n)]
    sample["carrier_reliability_prior"] = [0.55 if i < 4 else 0.93 for i in range(n)]
    sample["capacity_utilization_at_dispatch"] = [1.12 if i < 4 else 0.68 for i in range(n)]
    sample["remaining_sla_slack_hours"] = [2 if i < 4 else 48 for i in range(n)]
    sample["facility_backlog_at_dispatch"] = [50 if i < 4 else 5 for i in range(n)]
    sample["route_disruption_alert_at_dispatch"] = [1 if i < 4 else 0 for i in range(n)]
    sample["last_scan_age_hours"] = [12 if i < 4 else 1 for i in range(n)]
    sample["estimated_truck_delay_hours_at_dispatch"] = [10 if i < 4 else 0.5 for i in range(n)]
    sample["hour_at_dispatch"] = 16
    sample["weekday_at_dispatch"] = 2
    sample["price"] = pd.to_numeric(sample.get("avg_item_price", 79), errors="coerce").fillna(79)
    sample["freight_value"] = pd.to_numeric(sample.get("avg_freight", 16), errors="coerce").fillna(16)
    sample["product_weight_g"] = pd.to_numeric(sample.get("avg_weight_g", 700), errors="coerce").fillna(700)
    sample["dispatch_lag_hours"] = 12
    return PlainTextResponse(sample.to_csv(index=False), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=ontime_template.csv"})


@app.post("/api/score/batch")
async def score_batch(file: UploadFile = File(...), model: Optional[str] = Query(None)):
    raw = await _read_limited(file, MAX_BATCH_BYTES, "CSV")
    if not raw.strip():
        raise HTTPException(400, "The file is empty.")
    try:
        from scoring import read_orders_csv
        df = read_orders_csv(raw)
    except Exception as ex:
        raise HTTPException(400, f"Could not read the file as CSV ({ex.__class__.__name__}). Save it as comma-separated .csv and try again.")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(400, f"The file has {len(df):,} rows; the limit is {MAX_BATCH_ROWS:,}. Split it into smaller files.")
    try:
        scored = get_engine().score_batch(df, _scoring_model(model))
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    scored_at = datetime.now(timezone.utc).isoformat()
    scored["scored_at"] = scored_at
    rows = scored.where(pd.notna(scored), None).to_dict("records")
    ok = scored[scored.status == "SCORED"]
    rej = scored[scored.status == "REJECTED"]
    reason_counts = {}
    for txt in rej.get("reject_reasons", []):
        for r in str(txt).split(" | "):
            if r:
                reason_counts[r] = reason_counts.get(r, 0) + 1
    row_errors = [{"row": int(r["row"]) + 1 if "row" in rej.columns and pd.notna(r.get("row")) else int(i) + 1,
                   "order_id": None if pd.isna(r.get("order_id")) else str(r.get("order_id")),
                   "reasons": [x for x in str(r.get("reject_reasons", "")).split(" | ") if x]}
                  for i, r in rej.head(200).iterrows()]
    return {
        "filename": file.filename, "rows": rows, "model": _scoring_model(model), "scored_at": scored_at,
        "provenance": _provenance(_effective_model_key(model)),
        "row_errors": row_errors, "error_reasons": dict(sorted(reason_counts.items(), key=lambda kv: -kv[1])),
        "summary": {
            "total": int(len(scored)), "scored": int(len(ok)), "rejected": int((scored.status == "REJECTED").sum()),
            "flagged": int(ok.flagged.astype(bool).sum()) if len(ok) else 0,
            "low_confidence": int((ok.confidence == "LOW_CONFIDENCE").sum()) if len(ok) else 0,
            "bands": {b: int((ok.risk_level == b).sum()) for b in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
        },
    }


# ------------------------------------------------------------------ board
class BoardAdd(BaseModel):
    order_id: str = Field(min_length=1, max_length=120)
    owner: str = Field("", max_length=80)
    note: str = Field("", max_length=2000)


class BoardPatch(BaseModel):
    stage: Optional[str] = None
    owner: Optional[str] = Field(None, max_length=80)
    note: Optional[str] = Field(None, max_length=2000)


def _stages():
    return [{"key": s, "label": STAGE_LABELS[s], **STAGE_INFO[s]} for s in STAGES]


def _board_items():
    e = get_engine()
    out = []
    for it in board().items():
        o = e.get_order(it["order_id"])
        if o is None:
            continue
        out.append({**it, "order": {k: o[k] for k in ("order_id", "probability", "risk_level", "product_category", "customer_state",
                                                       "customer_city", "seller_state", "main_driver", "suggested_action",
                                                       "promised_by", "seller_id")},
                    "outcome": o["outcome"]})
    return out


@app.get("/api/board")
def board_list():
    items = _board_items()
    items.sort(key=lambda x: (x["stage"] == "resolved", -float(x["order"]["probability"])))
    return {"stages": _stages(), "items": items, "counts": board().counts()}


@app.post("/api/board", status_code=201)
def board_add(req: BoardAdd):
    if get_engine().get_order(req.order_id) is None:
        raise HTTPException(404, f"No order {req.order_id!r} in the queue.")
    if not board().add(req.order_id, req.owner, req.note):
        raise HTTPException(409, "That order is already on the board.")
    return board().get(req.order_id)


@app.patch("/api/board/{order_id}")
def board_patch(order_id: str, req: BoardPatch):
    try:
        r = board().update(order_id, req.stage, req.owner, req.note)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    if r is None:
        raise HTTPException(404, "That order is not on the board.")
    return r


@app.delete("/api/board/{order_id}")
def board_delete(order_id: str):
    if not board().remove(order_id):
        raise HTTPException(404, "That order is not on the board.")
    return {"removed": order_id}


@app.post("/api/board/autofill")
def board_autofill(n: int = Query(10, ge=1, le=50), band: str = "CRITICAL"):
    if band.upper() not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        raise HTTPException(400, f"Unknown risk band '{band}'. Use LOW, MEDIUM, HIGH or CRITICAL.")
    e = get_engine()
    have = board().ids()
    cand = e.queue[(e.queue.risk_level == band.upper()) & (~e.queue.order_id.isin(have))].head(n)
    for oid in cand.order_id:
        board().add(oid, auto=True)
    return {"added": int(len(cand))}


@app.get("/api/board/activity")
def board_activity(limit: int = Query(40, ge=1, le=200)):
    return {"items": board().activity(limit)}


# ------------------------------------------------------------------ analysis
@app.get("/api/insights")
def insights():
    return get_engine().insights


@app.get("/api/model")
def model():
    """Return the configured live-scoring model and its matching benchmark metadata.

    Historical queue evaluation stays in the offline reports and is not mixed into
    this live-scoring status payload.
    """
    e = get_engine()
    key = _effective_live_scoring_model(e)
    info = next((m for m in e.models_info() if m["key"] == key), None)
    is_synthetic = key == "synthetic_operational"
    return {
        "live_scoring": {
            "key": key,
            "label": (info or {}).get("label", key),
            "available": bool((info or {}).get("available", False)),
            "threshold": (info or {}).get("threshold"),
            "benchmark_type": (info or {}).get("benchmark_type"),
            "benchmark_roc_auc": (info or {}).get("benchmark_roc_auc") if is_synthetic else None,
            "benchmark_pr_auc": (info or {}).get("benchmark_pr_auc") if is_synthetic else None,
            "benchmark_train_rows": (info or {}).get("benchmark_train_rows") if is_synthetic else None,
            "benchmark_validation_rows": (info or {}).get("benchmark_validation_rows") if is_synthetic else None,
            "benchmark_test_rows": (info or {}).get("benchmark_test_rows") if is_synthetic else None,
            "benchmark_top_features": (info or {}).get("benchmark_top_features", []) if is_synthetic else [],
        },
        "serving_queue": e.serving,
        "historical_queue_available": bool(len(e.queue)),
        "configuration_source": "application settings",
        "interpretation": (
            "Synthetic dispatch benchmark metrics describe a simulated task only; "
            "scores are prioritisation signals, not calibrated real-world probabilities."
            if is_synthetic else
            "The selected model is configured for live scoring. Validate its outputs against current labelled outcomes."
        ),
    }


class HandoffReq(BaseModel):
    order_id: str = Field(..., min_length=1, max_length=120)
    handed_over_at: str = Field(..., min_length=8, max_length=40)         # ISO date-time the carrier collected the parcel, e.g. 2018-04-02T14:30


@app.get("/api/handoff/model")
def handoff_model():
    return hand.info()


@app.post("/api/score/handoff")
def score_handoff(req: HandoffReq):
    """Second score for a queued order, once the carrier has collected it."""
    e = get_engine()
    rows = e.queue[e.queue.order_id == req.order_id]
    if rows.empty:
        raise HTTPException(404, f"No order {req.order_id!r} in the queue.")
    try:
        when = pd.Timestamp(req.handed_over_at)
    except (ValueError, TypeError):
        raise HTTPException(400, "Use a date and time such as 2018-04-02T14:30.")
    if pd.isna(when):
        raise HTTPException(400, "Use a date and time such as 2018-04-02T14:30.")
    try:
        out = hand.score(rows.iloc[0], when)
    except hand.HandoffUnavailable as ex:
        raise HTTPException(503, str(ex))
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    r = rows.iloc[0]
    out["checkout_probability"] = float(r["probability"])
    out["checkout_risk_level"] = str(r["risk_level"])
    out["change"] = out["probability"] - out["checkout_probability"]
    out["provenance"] = {"stage": "after carrier handoff", "data_origin": "Historical Olist orders (real); handoff time supplied by the user",
                         "synthetic": False, "score_meaning": "ranking score; compare with the checkout score for the same order"}
    return out


@app.get("/api/model/thresholds")
def model_thresholds():
    """Precision, recall, accuracy and F1 at several alert sizes, measured on the SAME queue the
    Overview shows (the newest 20% of orders, which the serving model never saw in training)."""
    import numpy as np
    q = get_engine().queue
    if "is_delayed" not in q.columns or "probability" not in q.columns or not len(q):
        raise HTTPException(404, "The scored queue is not available.")
    ok = q["probability"].notna()
    p = q.loc[ok, "probability"].to_numpy(dtype=float)
    y = q.loc[ok, "is_delayed"].to_numpy().astype(int)
    rows = []
    for share in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):          # alert the riskiest X% of orders
        cut = float(np.quantile(p, 1 - share))
        a = p >= cut
        tp = int((a & (y == 1)).sum())
        rows.append({"alert_share": share, "cutoff": round(cut, 3), "alerts": int(a.sum()),
                     "precision": tp / max(int(a.sum()), 1), "recall": tp / max(int(y.sum()), 1),
                     "accuracy": float((a == (y == 1)).mean()),
                     "f1": 2 * tp / max(int(a.sum()) + int(y.sum()), 1)})
    base = float(y.mean())
    e = get_engine()
    note = f"Accuracy looks high at any cut-off because only about {base * 100:.0f}% of orders are late; judge by precision and recall."
    if e.view["in_sample"]:
        note = (f"{e.view['label']} was trained on a random 80% of all orders, so most of these orders were in its training data: "
                "precision and recall here are optimistic and do not predict real-world performance. " + note)
    return {"rows": rows, "base_rate": base, "n": int(len(y)), "model": e.serving, "model_label": e.view["label"],
            "in_sample": bool(e.view["in_sample"]), "note": note}


@app.get("/api/model/evaluation")
def model_evaluation():
    """Every score's ROC-AUC next to the data and test protocol it was measured on, so numbers from
    real data, optimistic random splits and simulations are never mixed up."""
    import json as _json
    import numpy as np
    from .engine import MODEL_DIR

    def load(*parts):
        try:
            with open(os.path.join(MODEL_DIR, *parts), encoding="utf-8") as f:
                return _json.load(f)
        except (OSError, ValueError):
            return {}

    e = get_engine()
    q = e.queue
    enr, xgb, ens = load("enriched", "ensemble_meta.json"), load("model_metadata.json"), load("ensemble", "ensemble_meta.json")
    hand, trk, syn = load("handoff_meta.json"), load("tracking_meta.json"), load("synthetic_operational", "metrics.json")
    split = enr.get("split", {})
    served_flagged = q["flagged"].astype(bool) if "flagged" in q.columns else None
    y = q["is_delayed"].astype(int) if "is_delayed" in q.columns else None
    served = {}
    if served_flagged is not None and y is not None and len(q):
        tp = int((served_flagged & (y == 1)).sum())
        served = {"alerts": int(served_flagged.sum()), "alert_share": float(served_flagged.mean()),
                  "precision": tp / max(int(served_flagged.sum()), 1), "recall": tp / max(int(y.sum()), 1),
                  "late_rate": float(y.mean())}
    mt = enr.get("metrics_on_test", {})
    rows = [
        {"key": "enriched", "label": "Enriched chronological model (real-Olist checkout score)", "stage": "At checkout",
         "data": "Real Olist orders", "real_data": True, "serving_queue": e.serving == "enriched",
         "protocol": f"Chronological: trained on the oldest 65% of orders, tested on the newest 20% ({str(split.get('test_start', ''))[:10]} to {str(split.get('test_end', ''))[:10]}), which it never saw.",
         "roc_auc": mt.get("roc_auc"), "pr_auc": mt.get("pr_auc"), "precision": mt.get("precision"),
         "recall": mt.get("recall"),
         "note": "The expected performance on real future orders: a chronological test on the newest orders, which the model never saw during training. Results on generated stand-in data are reported separately."},
        {"key": "xgboost", "label": "Original tuned XGBoost", "stage": "At checkout", "data": "Real Olist orders", "real_data": True,
         "serving_queue": e.serving == "xgboost",
         "protocol": "Random 80/20 split. Past and future orders are mixed, so this is optimistic compared with a real future.",
         "roc_auc": xgb.get("metrics_on_test", {}).get("roc_auc"), "pr_auc": None,
         "precision": xgb.get("metrics_on_test", {}).get("precision"), "recall": xgb.get("metrics_on_test", {}).get("recall"),
         "note": "Not comparable with the chronological score above."},
        {"key": "ensemble", "label": "Weighted ensemble", "stage": "At checkout", "data": "Real Olist orders", "real_data": True,
         "serving_queue": e.serving == "ensemble",
         "protocol": "Random 80/20 split (same as the original XGBoost).",
         "roc_auc": ens.get("metrics_on_test", {}).get("roc_auc"), "pr_auc": ens.get("metrics_on_test", {}).get("pr_auc"),
         "precision": ens.get("metrics_on_test", {}).get("precision"), "recall": ens.get("metrics_on_test", {}).get("recall"),
         "note": "Not comparable with the chronological score above."},
        {"key": "handoff", "label": "After the carrier collects the parcel", "stage": "After carrier handoff", "data": "Real Olist orders", "real_data": True,
         "serving_queue": False,
         "protocol": "Random split; month-by-month backtest "
                     f"{hand.get('monthly_backtest', {}).get('handoff', {}).get('mean_auc', 0):.2f} vs {hand.get('monthly_backtest', {}).get('checkout', {}).get('mean_auc', 0):.2f} at checkout.",
         "roc_auc": hand.get("random_split", {}).get("roc_auc"), "pr_auc": hand.get("random_split", {}).get("pr_auc"),
         "precision": hand.get("random_split", {}).get("precision"), "recall": hand.get("random_split", {}).get("recall"),
         "note": "A later stage with more information, so it is a separate score, never a replacement."},
        {"key": "tracking", "label": "In transit, with tracking", "stage": "In transit", "data": "Real orders + SIMULATED tracking", "real_data": False,
         "serving_queue": False,
         "protocol": "Newest 20% of orders; carrier scans, weather and strikes were invented for these orders.",
         "roc_auc": trk.get("with_tracking", {}).get("roc_auc"), "pr_auc": trk.get("with_tracking", {}).get("pr_auc"),
         "precision": trk.get("with_tracking", {}).get("precision"), "recall": trk.get("with_tracking", {}).get("recall"),
         "note": f"Describes the simulation only. Without tracking the same model reaches {trk.get('without_tracking', {}).get('roc_auc', 0):.2f}."},
        {"key": "synthetic_operational", "label": "Dispatch operational model (synthetic benchmark; opt-in)", "stage": "At dispatch", "data": "SIMULATED dispatch data", "real_data": False,
         "serving_queue": False,
         "protocol": f"Chronological split of {syn.get('synthetic_samples', 0):,} simulated orders; test = newest {syn.get('test_rows', 0):,}.",
         "roc_auc": syn.get("heldout_chronological_test_roc_auc"), "pr_auc": syn.get("heldout_chronological_test_pr_auc"),
         "precision": syn.get("test_precision_at_0_5"), "recall": syn.get("test_recall_at_0_5"),
         "note": "The labels were generated from the same simulated signals, so this proves the pipeline works, not real-world accuracy."},
    ]
    for r in rows:
        for k in ("roc_auc", "pr_auc", "precision", "recall"):
            v = r.get(k)
            r[k] = None if v is None or not np.isfinite(float(v)) else float(v)
    # --- the headline block follows the model selected in Settings ---
    from sklearn.metrics import roc_auc_score, average_precision_score
    key = e.serving
    by_key = {r["key"]: r for r in rows}
    active = dict(by_key.get(key, rows[0]))
    queue_scores = {}
    if y is not None and "probability" in q.columns and len(q) and y.nunique() > 1:
        queue_scores = {"roc_auc": float(roc_auc_score(y, q["probability"])), "pr_auc": float(average_precision_score(y, q["probability"]))}
    active.update({"key": key, "label": e.view["label"], "in_sample": bool(e.view["in_sample"]), "view_note": e.view["note"],
                   "queue_precision": served.get("precision"), "queue_recall": served.get("recall"),
                   "queue_roc_auc": queue_scores.get("roc_auc"), "queue_pr_auc": queue_scores.get("pr_auc"),
                   "queue_alerts": served.get("alerts"), "queue_alert_share": served.get("alert_share")})
    if key == "xgboost":
        active["protocol"] = "Random 80/20 split of all orders. Past and future orders are mixed, so this is optimistic compared with a real future."
    if e.view["in_sample"]:
        active["note"] = ("These are the model's own held-out random-split figures. Scored on the newest 19,294 orders below, the same model "
                          "reads higher only because most of those orders were in its training data.")
    synthetic_selected = e.view.get("limited_by") == "synthetic_operational"
    return {"rows": rows, "queue": {"n": int(len(q)), **served}, "active": active, "view": e.view,
            "dispatch_selected": synthetic_selected,
            "dispatch": by_key.get("synthetic_operational") if synthetic_selected else None,
            "reading_guide": "The checkout-time chronological row is the primary real-Olist retrospective result. Handoff is a separate later-stage experiment. In-transit and dispatch rows using simulated/generated signals are not verified real-world performance."}


# ------------------------------------------------------------------ models, literature, adapting to other data


# Plain-language description of every algorithm family the project uses.
ALGORITHM_FAMILIES = [
    {"key": "xgb", "name": "XGBoost", "kind": "Boosted trees",
     "how": "Builds many small decision trees one after another; each new tree corrects the mistakes of the ones before.",
     "strength": "Strong on tabular data; handles non-linear effects and interactions."},
    {"key": "hgb", "name": "Histogram gradient boosting", "kind": "Boosted trees",
     "how": "scikit-learn's boosted trees. Similar idea to XGBoost with a different implementation, so its errors differ a little.",
     "strength": "Fast; adds diversity to the blend."},
    {"key": "lr", "name": "Logistic regression", "kind": "Linear model",
     "how": "Adds up weighted inputs and squashes the total into a 0-1 risk. One straight-line weight per feature.",
     "strength": "Simple and easy to explain; a sanity check against the tree models."},
    {"key": "rf", "name": "Random forest", "kind": "Bagged trees",
     "how": "Trains many trees independently on random samples and averages their votes.",
     "strength": "Stable and hard to overfit; was tested but received 0% weight in the blend."},
    {"key": "lgbm", "name": "LightGBM", "kind": "Boosted trees",
     "how": "A third boosted-tree library, tried as a candidate when training the enriched model.",
     "strength": "Optional: it is only needed for retraining, not for running the app."},
]


@app.get("/api/models/compare")
def models_compare():
    """One payload for the Model comparison section: algorithms, ensemble weights, candidates, scoreboard."""
    e = get_engine()
    ens = e.ensemble_meta or {}
    weights = ens.get("weights", {})
    fams = []
    for f in ALGORITHM_FAMILIES:
        w = weights.get(f["key"])
        fam = {**f, "ensemble_weight": w, "in_ensemble": f["key"] in weights}
        if f["key"] == "lgbm" and not getattr(e, "lightgbm_available", True):
            fam["strength"] += " Not installed on this machine, so retraining it is unavailable; scoring is unaffected."
        fams.append(fam)
    enr = getattr(e, "enriched_meta", None) or {}
    cand = enr.get("validation_auc_by_candidate") or {}
    names = {"xgboost_enriched": "XGBoost", "lightgbm_enriched": "LightGBM", "hist_gradient_enriched": "Histogram gradient boosting"}
    candidates = sorted(({"name": names.get(k, k), "validation_auc": v, "selected": k == "xgboost_enriched"} for k, v in cand.items()),
                        key=lambda r: -r["validation_auc"])
    try:
        scoreboard = model_evaluation().get("rows", [])
    except Exception:
        scoreboard = []
    return {"families": fams, "ensemble": {"weights": weights, "decision": ens.get("decision"),
                                           "roc_auc": (ens.get("metrics_on_test") or {}).get("roc_auc")},
            "candidates": candidates, "scoreboard": scoreboard}


@app.get("/api/models")
def models():
    e = get_engine()
    live_key = _effective_live_scoring_model(e)
    live_info = next((m for m in e.models_info() if m["key"] == live_key), None)
    return {"models": e.models_info(), "serving_queue": e.serving,
            "scoring_setting": ai.load_settings().get("scoring_model", "synthetic_operational"),
            "live_scoring_model": live_key, "live_scoring_label": (live_info or {}).get("label", live_key),
            "live_scoring_available": bool((live_info or {}).get("available", False)),
            "agreement": e.agreement(), "ensemble_error": e.ensemble_error, "report": e.ensemble_report}


@app.get("/api/literature")
def literature():
    return get_engine().literature


MAX_UPLOAD_BYTES = 40 * 1024 * 1024


@app.post("/api/adapt/preview")
async def adapt_preview(file: UploadFile = File(...)):
    raw = await _read_limited(file, MAX_UPLOAD_BYTES, "file")
    try:
        return adapt.preview(get_engine(), raw, file.filename)
    except adapt.AdaptError as ex:
        raise HTTPException(400, str(ex))


class AdaptRun(BaseModel):
    dataset_id: str
    mapping: dict
    train: bool = False
    name: Optional[str] = Field(None, max_length=60)
    model: Optional[str] = None


@app.post("/api/adapt/run")
def adapt_run(req: AdaptRun):
    try:
        # Adapt-to-data learns checkout-time schemas, so absent an explicit choice it
        # keeps using the historical checkout model rather than the dispatch model default.
        adapt_model = _scoring_model(req.model) if req.model else "auto"
        if adapt_model == "synthetic_operational":
            raise adapt.AdaptError("The dispatch operational model expects dispatch-time telemetry. Choose a checkout-time model for Adapt to your data.")
        return adapt.run(get_engine(), req.dataset_id, req.mapping, train=req.train, name=req.name, model_key=adapt_model)
    except adapt.AdaptError as ex:
        raise HTTPException(400, str(ex))


@app.get("/api/adapt/models")
def adapt_models():
    return {"models": adapt.list_models()}


@app.delete("/api/adapt/models/{model_id}")
def adapt_delete(model_id: str):
    try:
        adapt.delete_model(model_id)
    except adapt.AdaptError as ex:
        raise HTTPException(404, str(ex))
    return {"removed": model_id}


@app.post("/api/adapt/models/{model_id}/score")
async def adapt_score(model_id: str, file: UploadFile = File(...)):
    raw = await _read_limited(file, MAX_UPLOAD_BYTES, "file")
    try:
        return adapt.score_with(get_engine(), model_id, raw)
    except adapt.AdaptError as ex:
        raise HTTPException(400, str(ex))


# ------------------------------------------------------------------ monitoring
@app.get("/api/monitoring/summary")
def monitoring_summary(alert_share: Optional[float] = Query(None, gt=0, lt=1), bootstrap: int = Query(300, ge=50, le=1000)):
    """Confidence intervals, calibration, monthly stability and early-vs-late drift on the held-out real queue."""
    e = get_engine()
    q = e.queue
    y, p = q["is_delayed"].astype(int).to_numpy(), q["probability"].astype(float).to_numpy()
    if alert_share is None:      # default: the share of orders the ACTIVE model actually alerts on
        alert_share = min(max(float(q["flagged"].mean()), 0.01), 0.99)
    in_sample = bool(e.view["in_sample"])
    return {
        "model": e.serving, "model_label": e.view["label"], "in_sample": in_sample, "view_note": e.view["note"],
        "provenance": {"data_origin": (f"Newest 20% of real Olist orders, scored by {e.view['label']}. This model was trained on a random 80% of all orders, so most of these orders were in its training data and the figures are optimistic"
                                       if in_sample else "Held-out newest 20% of real Olist orders (chronological); never used in training or selection"),
                       "synthetic": False},
        "confidence_intervals": mon.bootstrap_ci(y, p, share=alert_share, n_boot=bootstrap),
        "calibration": {
            "served_score": mon.calibration(y, p),
            "calibrated_ensemble": mon.calibration(y, q["p_calibrated_ensemble"].astype(float).to_numpy()) if "p_calibrated_ensemble" in q.columns and q["p_calibrated_ensemble"].notna().all() else None,
            "reading": "The served score is a class-weighted ranking score and over-states the chance of being late; the calibrated ensemble score is the one to read as a probability.",
        },
        "monthly_stability": mon.monthly_stability(q),
        "drift_early_vs_late": mon.early_vs_late_drift(q),
        "retraining_criteria": ["Any feature or the score reaches PSI > 0.25 against the training period",
                                "Rolling 30-day ROC-AUC on matched outcomes falls below the lower 95% bound reported here",
                                "Alert precision falls below the base late rate for two consecutive weeks"],
        "rollback": "Keep the previous model files; set Settings > scoring model back to the earlier model key and restart.",
    }


@app.post("/api/monitoring/drift")
async def monitoring_drift(file: UploadFile = File(...)):
    """Compare an uploaded batch of orders with the held-out queue (population stability index per input)."""
    raw = await _read_limited(file, MAX_BATCH_BYTES, "CSV")
    try:
        from scoring import read_orders_csv
        df = read_orders_csv(raw)
    except Exception as ex:
        raise HTTPException(400, f"Could not read the file as CSV ({ex.__class__.__name__}).")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(400, f"The file has {len(df):,} rows; the limit is {MAX_BATCH_ROWS:,}.")
    if len(df) < 20:
        raise HTTPException(400, "Need at least 20 rows to measure drift.")
    return mon.drift_report(get_engine().queue, df, score_col="__none__")


@app.post("/api/monitoring/outcomes")
async def monitoring_outcomes(file: UploadFile = File(...)):
    """Upload real outcomes (order_id, is_delayed) to measure how OnTime's earlier scores performed."""
    raw = await _read_limited(file, MAX_BATCH_BYTES, "CSV")
    e = get_engine()
    try:
        return mon.outcomes_report(e.queue, raw, float(e.threshold))
    except ValueError as ex:
        raise HTTPException(400, str(ex))


# ------------------------------------------------------------------ frontend
if os.path.isdir(FRONTEND):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND, "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(FRONTEND, "index.html"))


@app.get("/api/weather/forecast")
def live_weather_forecast(latitude: float = Query(..., ge=-90, le=90), longitude: float = Query(..., ge=-180, le=180)):
    """Live Open-Meteo weather for explicit coordinates. Separate from simulated historical tracking and excluded from model inputs."""
    try:
        return wx.forecast(latitude, longitude)
    except wx.WeatherUnavailable as ex:
        raise HTTPException(503, str(ex))
    except ValueError as ex:
        raise HTTPException(400, str(ex))


class TrackingReq(BaseModel):
    order_id: str = Field(..., min_length=1, max_length=120)
    as_of: Optional[str] = Field(None, max_length=40)                    # ISO date-time, e.g. 2018-04-05T09:00
    days_after_purchase: Optional[float] = None    # alternative to as_of


def _tracking_row(order_id):
    e = get_engine()
    rows = e.queue[e.queue.order_id == order_id]
    if rows.empty:
        raise HTTPException(404, f"No order {order_id!r} in the queue.")
    return rows.iloc[0]


@app.get("/api/tracking/model")
def tracking_model():
    """Metrics of the SIMULATED-tracking score (Model page)."""
    return trk.info()


@app.get("/api/tracking/{order_id}")
def tracking_timeline(order_id: str, days_after_purchase: float = Query(7.0), as_of: Optional[str] = None):
    """Simulated scans, weather and strikes known for one order at a chosen moment."""
    row = _tracking_row(order_id)
    try:
        return trk.timeline(row, trk.resolve_time(row, as_of, None if as_of else days_after_purchase))
    except trk.TrackingUnavailable as ex:
        raise HTTPException(503, str(ex))
    except ValueError as ex:
        raise HTTPException(400, str(ex))


@app.post("/api/score/tracking")
def score_tracking(req: TrackingReq):
    """Third score for a queued order that is in transit, from SIMULATED carrier scans, weather and strikes."""
    row = _tracking_row(req.order_id)
    try:
        out = trk.score(row, trk.resolve_time(row, req.as_of, req.days_after_purchase))
    except trk.TrackingUnavailable as ex:
        raise HTTPException(503, str(ex))
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    out["checkout_probability"] = float(row["probability"])
    out["checkout_risk_level"] = str(row["risk_level"])
    out["change"] = out["probability"] - out["checkout_probability"]
    out["provenance"] = {"stage": "in transit", "data_origin": "SIMULATED carrier scans, weather and strikes layered on real Olist orders",
                         "synthetic": True, "score_meaning": "ranking score from a simulated-tracking experiment; not validated on live carrier feeds"}
    return out
