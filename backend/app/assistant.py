"""
assistant.py - "Ask OnTime", a question-answering assistant with three engines.

  builtin  Works offline, always available. Recognises what the question is
           about (an order, a pattern, a category, a state, the model...)
           and answers from the app's own numbers.
  laptop   A local model on this computer through Ollama
           (https://ollama.com, e.g. `ollama pull llama3.2`). Nothing leaves
           the machine.
  gemini   Google's Gemini API with your own key.

Grounding: whichever engine is chosen, the built-in engine first collects
the relevant FACTS from the app's data. The AI engines only get those facts
plus the question, and are told to use nothing else, so they word the
answer better without inventing numbers.

Time limit: every AI call runs against a hard deadline (5-120 seconds,
set in Settings). If the engine is unreachable, errors or runs out of time,
the built-in answer is returned instead and the reply says why.
"""
import concurrent.futures as cf
import json
import os
import re
import time

import httpx

MAX_TIMEOUT_S = 120
SETTINGS_PATH = os.environ.get("ONTIME_SETTINGS", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json"))
GEMINI_BASE = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
ENGINES = ("builtin", "laptop", "gemini", "auto")   # auto = Gemini if a key is set, else Ollama if running, else built-in

DEFAULTS = {
    "engine": "auto",      # Auto: cloud AI if a key is set, else Ollama if running, else built-in; falls back by itself
    "timeout_s": 30,
    "ollama_url": os.environ.get("OLLAMA_URL", "http://localhost:11434"),
    "ollama_model": os.environ.get("OLLAMA_MODEL", "llama3.2"),
    "gemini_model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    "gemini_api_key": "",
    "scoring_model": "enriched",   # real-Olist checkout-time model by default; synthetic dispatch benchmark is opt-in
    "scoring_model_migrated_v3": True,
    "database_mode": os.environ.get("DATABASE_MODE", "sqlite"),  # sqlite or supabase
    "supabase_url": os.environ.get("SUPABASE_URL", ""),
    "supabase_key": os.environ.get("SUPABASE_KEY", ""),
}
SCORING_MODELS = ("auto", "xgboost", "ensemble", "enriched", "synthetic_operational")

SYSTEM_PROMPT = (
    "You are the assistant inside OnTime, a delivery-delay early-warning tool for an e-commerce logistics team "
    "(Olist orders, Brazil, 2016-2018). Answer the user's question using ONLY the FACTS provided. "
    "If the facts do not contain the answer, say so plainly and suggest what to look at in the app. "
    "Never invent numbers, orders, sellers or causes. Simulated logistics fields must be called simulated. "
    "Write plain, direct English for a logistics manager: short paragraphs, no markdown tables, at most 170 words. "
    "Quote figures exactly as given. If RECENT CONVERSATION is provided, use it only to understand what the question refers to."
)


# ------------------------------------------------------------------ settings
def _is_placeholder(v):
    """Template values from .env.example ("YOUR_GEMINI_API_KEY_HERE", "your-supabase-...") are not real keys."""
    v = (v or "").strip().lower()
    return (not v) or v.startswith("your") or v.startswith("https://yourproject") or v in ("changeme", "xxx")


def load_settings(path=None):
    path = path or SETTINGS_PATH
    s = dict(DEFAULTS)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
                s.update({k: v for k, v in saved.items() if k in DEFAULTS})
                # Migrate the old automatic/default choice to the real-Olist checkout model.
                # Preserve explicit model selections; the synthetic dispatch benchmark remains opt-in.
                if saved.get("scoring_model") == "auto" and "scoring_model_migrated_v3" not in saved:
                    s["scoring_model"] = "enriched"
                    s["scoring_model_migrated_v3"] = True
        except (OSError, ValueError):
            pass
    if not s.get("gemini_api_key"):
        s["gemini_api_key"] = os.environ.get("GEMINI_API_KEY", "")
    if not s.get("supabase_url"):
        s["supabase_url"] = os.environ.get("SUPABASE_URL", "")
    if not s.get("supabase_key"):
        s["supabase_key"] = os.environ.get("SUPABASE_KEY", "")
    if not s.get("database_mode"):
        s["database_mode"] = os.environ.get("DATABASE_MODE", "sqlite")
    for k in ("gemini_api_key", "supabase_key", "supabase_url"):
        if _is_placeholder(s.get(k)):
            s[k] = ""
    s["timeout_s"] = clamp_timeout(s.get("timeout_s"))
    if s.get("engine") not in ENGINES:
        s["engine"] = "builtin"
    if s.get("scoring_model") not in SCORING_MODELS:
        s["scoring_model"] = "auto"
    return s


def clamp_timeout(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = DEFAULTS["timeout_s"]
    return int(max(5, min(MAX_TIMEOUT_S, v)))


def save_settings(changes, path=None):
    path = path or SETTINGS_PATH
    cur = load_settings(path)
    for k, v in changes.items():
        if k not in DEFAULTS:
            continue
        if k == "gemini_api_key" and (v is None or v == "__keep__"):
            continue
        cur[k] = v
    cur["timeout_s"] = clamp_timeout(cur["timeout_s"])
    if cur["engine"] not in ENGINES:
        raise ValueError(f"Unknown engine '{cur['engine']}'. Use one of: {', '.join(ENGINES)}.")
    if cur.get("scoring_model") not in SCORING_MODELS:
        raise ValueError(f"Unknown scoring model '{cur.get('scoring_model')}'. Use one of: {', '.join(SCORING_MODELS)}.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cur, f, indent=2)
    return cur


def public_settings(s):
    key = s.get("gemini_api_key") or ""
    sb_key = s.get("supabase_key") or ""
    out = {k: v for k, v in s.items() if k not in ("gemini_api_key", "supabase_key", "scoring_model_migrated_v2", "scoring_model_migrated_v3")}
    out["supabase_key_set"] = bool(sb_key)
    out["supabase_key_hint"] = (sb_key[:6] + "..." + sb_key[-4:]) if len(sb_key) >= 14 else ("set" if sb_key else "")
    out["gemini_key_set"] = bool(key)
    out["gemini_key_hint"] = (key[:4] + "…" + key[-4:]) if len(key) >= 12 else ("set" if key else "")
    out["max_timeout_s"] = MAX_TIMEOUT_S
    return out


# ------------------------------------------------------------------ AI engines
class EngineError(RuntimeError):
    pass


def _pick_ollama_model(s):
    """The configured model if installed, otherwise the first installed one (so a wrong name still works)."""
    want = s["ollama_model"]
    try:
        r = httpx.get(s["ollama_url"].rstrip("/") + "/api/tags", timeout=1.5)
        have = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:  # noqa: BLE001
        return want
    if not have or want in have or any(m.split(":")[0] == want.split(":")[0] for m in have):
        return want
    return have[0]


def _call_ollama(s, system, user, timeout):
    url = s["ollama_url"].rstrip("/") + "/api/chat"
    model = _pick_ollama_model(s)
    body = {"model": model, "stream": False, "keep_alive": "10m", "options": {"temperature": 0.2, "num_predict": 600},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    try:
        r = httpx.post(url, json=body, timeout=httpx.Timeout(timeout, connect=min(5, timeout)))
    except httpx.ConnectError:
        raise EngineError(f"Ollama is not running at {s['ollama_url']}. Install it from ollama.com and run "
                          f"`ollama pull {s['ollama_model']}`.")
    if r.status_code == 404:
        raise EngineError(f"Ollama has no model '{s['ollama_model']}'. Run `ollama pull {s['ollama_model']}` or pick an "
                          "installed model in Settings.")
    if r.status_code >= 400:
        raise EngineError(f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")
    try:
        text = r.json()["message"]["content"]
    except (ValueError, KeyError, TypeError):
        raise EngineError("Ollama sent a reply in an unexpected format.")
    if not text.strip():
        raise EngineError("Ollama returned an empty answer.")
    return text.strip()


def _call_gemini(s, system, user, timeout):
    key = s.get("gemini_api_key")
    if not key:
        raise EngineError("No cloud AI API key set. Add one in Settings (get it free at aistudio.google.com).")
    url = f"{GEMINI_BASE}/models/{s['gemini_model']}:generateContent"
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 900}}
    if "2.5" in s["gemini_model"] and "pro" not in s["gemini_model"]:
        body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}   # answer tokens only
    r = None
    for attempt in range(2):                                                  # one retry if Google is busy
        try:
            r = httpx.post(url, json=body, headers={"x-goog-api-key": key},
                           timeout=httpx.Timeout(timeout, connect=min(10, timeout)))
        except httpx.ConnectError:
            raise EngineError("Could not reach the cloud AI service. Check the internet connection.")
        if r.status_code in (429, 500, 503) and attempt == 0:
            time.sleep(1.5)
            continue
        break
    if r.status_code in (400, 401, 403):
        msg = ""
        try:
            msg = r.json().get("error", {}).get("message", "")
        except ValueError:
            pass
        raise EngineError(f"The cloud AI rejected the request ({r.status_code}). {msg or 'Check the API key and model name in Settings.'}")
    if r.status_code == 404:
        raise EngineError(f"The cloud AI has no model '{s['gemini_model']}'. Check the model name in Settings.")
    if r.status_code == 429:
        raise EngineError("Cloud AI rate limit reached (free tier). Wait a minute or use another engine.")
    if r.status_code >= 400:
        raise EngineError(f"The cloud AI returned HTTP {r.status_code}.")
    try:
        parts = r.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
    except (ValueError, KeyError, IndexError, TypeError):
        raise EngineError("The cloud AI sent no answer (it may have been blocked by its safety filters).")
    if not text.strip():
        raise EngineError("The cloud AI returned an empty answer.")
    return text.strip()


_pool = cf.ThreadPoolExecutor(max_workers=4)


def run_engine(engine, s, system, user, timeout_s=None):
    """Call an AI engine with a hard deadline. Returns text or raises EngineError."""
    timeout = clamp_timeout(timeout_s if timeout_s is not None else s["timeout_s"])
    fn = {"laptop": _call_ollama, "gemini": _call_gemini}.get(engine)
    if fn is None:
        raise EngineError(f"Unknown engine '{engine}'.")
    fut = _pool.submit(fn, s, system, user, timeout)
    try:
        return fut.result(timeout=timeout + 1)
    except cf.TimeoutError:
        raise EngineError(f"No answer within the {timeout}-second limit.")
    except httpx.TimeoutException:
        raise EngineError(f"No answer within the {timeout}-second limit.")
    except EngineError:
        raise
    except Exception as e:  # noqa: BLE001
        raise EngineError(f"{e.__class__.__name__}: {e}")


def engine_status(s):
    out = {"builtin": {"available": True, "detail": "Always available, works offline."}}
    try:
        r = httpx.get(s["ollama_url"].rstrip("/") + "/api/tags", timeout=1.5)
        models = [m.get("name", "") for m in r.json().get("models", [])]
        has = any(m == s["ollama_model"] or m.split(":")[0] == s["ollama_model"].split(":")[0] for m in models)
        out["laptop"] = {"available": bool(models) and has, "models": models,
                         "detail": (f"Ollama running with {s['ollama_model']}." if has else
                                    f"Ollama is running but '{s['ollama_model']}' is not installed." if models else
                                    "Ollama is running but has no models yet.")}
    except Exception:  # noqa: BLE001
        out["laptop"] = {"available": False, "models": [],
                         "detail": f"Ollama not found at {s['ollama_url']}. Install it from ollama.com to use this engine."}
    out["gemini"] = {"available": bool(s.get("gemini_api_key")),
                     "detail": f"Key set, model {s['gemini_model']}." if s.get("gemini_api_key") else "No API key set."}
    pick = "gemini" if out["gemini"]["available"] else "laptop" if out["laptop"]["available"] else "built-in"
    out["auto"] = {"available": True, "detail": f"Will use {pick} right now; falls back by itself if it fails."}
    return out


# ------------------------------------------------------------------ built-in engine
STATE_NAMES = {
    "acre": "AC", "alagoas": "AL", "amapa": "AP", "amazonas": "AM", "bahia": "BA", "ceara": "CE", "distrito federal": "DF",
    "brasilia": "DF", "espirito santo": "ES", "goias": "GO", "maranhao": "MA", "mato grosso do sul": "MS", "mato grosso": "MT",
    "minas gerais": "MG", "para": "PA", "paraiba": "PB", "parana": "PR", "pernambuco": "PE", "piaui": "PI",
    "rio de janeiro": "RJ", "rio grande do norte": "RN", "rio grande do sul": "RS", "rondonia": "RO", "roraima": "RR",
    "santa catarina": "SC", "sao paulo": "SP", "sergipe": "SE", "tocantins": "TO",
}
FOCUS_WORDS = {"health": "health_beauty", "beauty": "health_beauty", "baby": "baby", "watch": "watches_gifts",
               "gift": "watches_gifts", "computer": "computers_accessories", "furniture": "furniture_decor",
               "decor": "furniture_decor", "bed": "bed_bath_table", "bath": "bed_bath_table"}
EXAMPLES = ["Why is the riskiest order risky?", "What is pattern 2?", "Which states have the most late deliveries?",
            "What should we do about baby products?", "How reliable is the model?", "Is the ensemble better than XGBoost?",
            "What did tuning change?", "Which papers support this approach?", "Can I use my own company's data?",
            "How many critical orders are there?", "What does the simulated hub data show?"]


def _pct(x, d=1):
    return f"{x:.{d}f}%"


def builtin_answer(q, eng, board_counts=None):
    """Returns dict(paragraphs, facts, sources, follow_ups, topic)."""
    from narrative import order_story, pattern_story, category_story
    text = q.strip()
    ql = text.lower()
    norm = re.sub(r"[^a-z0-9 ]", " ", ql.replace("ã", "a").replace("á", "a").replace("é", "e").replace("í", "i")
                  .replace("ó", "o").replace("ô", "o").replace("ç", "c"))
    pats = eng.patterns["patterns"]
    by_rank = {p["rank"]: p for p in pats}

    # 1. an order or seller ID
    m = re.search(r"\b([0-9a-f]{6,32})\b", ql)
    if m:
        tok = m.group(1)
        hit = eng.queue[eng.queue.order_id.str.startswith(tok)]
        if len(hit):
            o = eng.get_order(hit.order_id.iloc[0])
            pat = by_rank.get(o.get("pattern"))
            cat = eng.categories["focus"].get(o["product_category"])
            paras = order_story(o, pat if o["flagged"] else None, cat)
            if re.search(r"what should|do about|action|next step|fix|recommend|what now|how (do|can) (we|i)", ql):
                paras = [f"Suggested action for order {o['order_id'][:10]}…: " + re.sub(r"^URGENT: ", "URGENT - ", o["suggested_action"] or "No action needed.")]
                if pat and o["flagged"]:
                    paras.append(f"It resembles pattern {pat['rank']} ({pat['name']}). Pattern-level fix: {pat['recommendations'][0]}")
                if cat and o["flagged"]:
                    paras.append(f"{cat['label']} is a {cat['tier'].lower()} category: {cat['playbook'][0]}")
            return dict(topic="order", paragraphs=paras, sources=[f"Order {o['order_id'][:10]}…", "Risk model (TreeSHAP)"],
                        facts={"order": {k: o[k] for k in ("order_id", "probability", "risk_level", "flagged", "product_category",
                                                           "customer_state", "seller_state", "suggested_action", "main_driver")},
                               "reasons": o["reasons"], "outcome": o["outcome"], "pattern": pat and pat["name"]},
                        follow_ups=["What should we do about it?", f"What is pattern {o.get('pattern')}?"], order_id=o["order_id"])
        sel = eng.queue[eng.queue.seller_id.fillna("").str.startswith(tok)]
        hist = next((v for k, v in eng.lookup["seller"].items() if k.startswith(tok)), None)
        if len(sel) or hist:
            sid = sel.seller_id.iloc[0] if len(sel) else next(k for k in eng.lookup["seller"] if k.startswith(tok))
            n_flag = int(sel.flagged.sum())
            paras = [f"Seller {sid[:10]}… has {hist['n_orders']:,} delivered orders in the history, {_pct(hist['rate'] * 100)} of "
                     f"them late (platform average {_pct(eng.lookup['global_rate'] * 100)})." if hist else f"Seller {sid[:10]}…",
                     f"In the current queue this seller has {len(sel)} orders, {n_flag} of them with an alert."]
            if hist and hist["rate"] >= 0.15:
                paras.append("That record is well above average: a 48-hour dispatch SLA and a weekly scorecard are the usual fix.")
            return dict(topic="seller", paragraphs=paras, sources=["Seller history"], facts={"seller": sid, "history": hist,
                        "queue_orders": int(len(sel)), "queue_alerts": n_flag}, follow_ups=["Which sellers are the worst?"])


    # 1b. new topics: literature, ensemble / tuning, adapting to other data
    if re.search(r"\b(literature|papers?|related work|research|cite|citations?|references?|studies|study|bibliograph|who else)\b", ql) \
            and not re.search(r"\bthis (study|research)\b", ql):
        lit = eng.literature
        words = [w for w in re.findall(r"[a-z]{4,}", norm) if w not in {"which", "papers", "paper", "literature", "support", "about", "related", "work",
                                                                       "research", "studies", "study", "cite", "this", "that", "approach", "what", "does",
                                                                       "show", "there", "have", "with", "from", "into", "your", "their", "reference", "references"}]
        scored = []
        for th in lit["themes"]:
            for pp in th["papers"]:
                hay = " ".join([pp["title"], pp["authors"], pp["what"], pp["used"], th["title"]]).lower()
                sc = sum(2 if w in th["title"].lower() else 1 for w in words if w in hay)
                if sc:
                    scored.append((sc, th, pp))
        scored.sort(key=lambda t: -t[0])
        pick = scored[:3]
        if not pick:
            pick = [(0, th, th["papers"][0]) for th in lit["themes"][:3]]
        paras = [f"The Literature page lists {sum(len(t['papers']) for t in lit['themes'])} sources in {len(lit['themes'])} themes. The closest to your question:"]
        for _, th, pp in pick:
            paras.append(f"{pp['authors'].split(',')[0]} et al. ({pp['year']}), \"{pp['title']}\": {pp['what']} In OnTime: {pp['used']}")
        return dict(topic="literature", paragraphs=paras, sources=["Literature page"], facts={"papers": [{"authors": pp["authors"], "year": pp["year"],
                    "title": pp["title"], "what": pp["what"], "used": pp["used"]} for _, _, pp in pick]},
                    follow_ups=["Is the ensemble better than XGBoost?", "How reliable is the model?"])

    if re.search(r"ensemble|blend|stack|voting|second opinion|combine (the )?models|xgboost vs|compare (the )?models|which model", ql):
        rep = eng.ensemble_report
        if not rep:
            return dict(topic="ensemble", paragraphs=["No ensemble has been trained in this copy. Run train_ensemble.py (see the README) to compare tuned models and a weighted ensemble."],
                        sources=["Model page"], facts={}, follow_ups=["How reliable is the model?"])
        rows = {r["key"]: r for r in rep["models"]}
        prod, bl = rows["production_xgb"], rows["blend"]
        w = ", ".join(f"{rep['member_labels'].get(k, k)} {v * 100:.0f}%" for k, v in rep["weights"].items() if v > 0)
        pb = rep.get("paired_backtest") or {}
        paras = [f"The ensemble is a weighted average of {w}. Weights were fitted on out-of-fold training predictions, so the test orders never influenced them.",
                 f"On the held-out test orders the ensemble scored ROC-AUC {bl['roc_auc']:.3f} against {prod['roc_auc']:.3f} for the production XGBoost "
                 f"(PR-AUC {bl['pr_auc']:.3f} vs {prod['pr_auc']:.3f})."]
        if rep.get("backtest_mean_auc"):
            bm = rep["backtest_mean_auc"]
            paras.append(f"On the month-by-month backtest (retrain monthly, predict the next month) the mean ROC-AUC was {bm.get('blend')} for the ensemble and "
                         f"{bm.get('production_xgb')} for the production XGBoost" + (f", with the ensemble ahead in {pb['months_blend_better']} of {pb['months']} months." if pb.get("months") else "."))
        serving_label = {"enriched": "the enriched chronological model", "ensemble": "the weighted ensemble", "xgboost": "the original tuned XGBoost"}.get(eng.serving, eng.serving)
        paras.append(f"The risk queue is currently served by {serving_label}. The ensemble and XGBoost figures above come from a random split of past orders, "
                     "which is optimistic; the queue model is judged on the newest 20% of orders it never saw, so its ROC-AUC is lower and is the better guide to real use (ask: How reliable is the model?).")
        ag = eng.agreement()
        if ag:
            paras.append(f"On the queue, the two models both alert on {ag['both']:,} orders; {ag['n_disagree']:,} orders get an alert from only one of them (filter: Models disagree).")
        return dict(topic="ensemble", paragraphs=paras, sources=["Model page", "Ensemble report"], facts={"models": [{k: r[k] for k in ("label", "roc_auc", "pr_auc", "f1")} for r in rep["models"]],
                    "weights": rep["weights"], "backtest_mean_auc": rep.get("backtest_mean_auc"), "decision": rep["decision"], "agreement": ag},
                    follow_ups=["What did tuning change?", "Which papers support this approach?"])

    if re.search(r"tun(e|ed|ing)|hyper.?param|grid|random search|optimi[sz]", ql):
        rep = eng.ensemble_report
        if not rep:
            return dict(topic="tuning", paragraphs=["The original model was tuned with a 20-configuration random search (notebook 06). Run train_ensemble.py for the wider, cross-validated search."],
                        sources=["Model page"], facts={}, follow_ups=["How reliable is the model?"])
        t = rep["tuning"]["xgb"]
        paras = [f"Tuning searched {t['trials']} XGBoost settings (the production settings were trial 0) with 3-fold cross-validation on the training orders only. "
                 f"The production settings scored a mean CV ROC-AUC of {t['baseline_cv_auc']:.4f}; the best found scored {t['best_cv_auc']:.4f}.",
                 f"Histogram gradient boosting got {rep['tuning']['hgb']['trials']} trials (best CV ROC-AUC {rep['tuning']['hgb']['best_cv_auc']:.4f}).",
                 "The differences between good settings are small, which is typical when the data, not the settings, limits accuracy. "
                 "That is why the ensemble and the monthly backtest matter more than a fourth decimal of AUC."]
        return dict(topic="tuning", paragraphs=paras, sources=["Model page", "Tuning report"], facts={"xgb": {k: t[k] for k in ("trials", "baseline_cv_auc", "best_cv_auc")},
                    "hgb": {k: rep["tuning"]["hgb"][k] for k in ("trials", "best_cv_auc")}}, follow_ups=["Is the ensemble better than XGBoost?"])

    if re.search(r"(my|our|own|other|another|different|new)\b.{0,25}\b(data ?set|data|csv|file|company|shop|store|orders)\b|adapt|column (names|mapping)|map (the )?columns|upload", ql) \
            and not re.search(r"batch", ql):
        paras = ["Yes. Open Adapt to your data, upload a CSV with your own column names, and OnTime matches the columns to what the model needs (order date, promised date, "
                 "state or distance, weight, price, and so on), converting units and dates and showing which inputs came from your file and which were filled with typical values.",
                 "It then scores your orders with the Olist model and, if your file says which orders were late (an outcome flag, a status, or delivered versus promised dates), "
                 "measures how well that transfers. If transfer is weak and you have enough history (at least 300 orders with 30 late), it trains a model on your own data and compares the two on the same recent orders."]
        return dict(topic="adapt", paragraphs=paras, sources=["Adapt to your data"], facts={}, follow_ups=["Which papers support this approach?", "How reliable is the model?"])

    # 2. patterns
    m = re.search(r"(pattern|cluster|group)\s*#?\s*(\d)", ql)
    if m and int(m.group(2)) in by_rank:
        p = by_rank[int(m.group(2))]
        return dict(topic="pattern", paragraphs=pattern_story(p, eng.patterns["n_late"]) + ["Recommended actions: " + " ".join(p["recommendations"])],
                    sources=[f"Pattern {p['rank']}"], facts={"pattern": {k: p[k] for k in ("rank", "name", "n", "share_of_late",
                    "median_distance_km", "median_promised_days", "median_actual_days", "distinguishing", "states", "categories",
                    "recommendations")}}, follow_ups=["Which pattern is the biggest?", "What should we do about baby products?"])
    if re.search(r"\b(patterns?|clusters?|groups?)\b", ql):
        paras = [f"Late deliveries fall into {len(pats)} patterns (k-means on {eng.patterns['n_late']:,} late orders):"]
        paras += [f"{p['rank']}. {p['name']}: {p['n']:,} orders ({p['share_of_late']}%)." for p in pats]
        paras.append("Ask about any of them, for example \"What is pattern 2?\".")
        return dict(topic="patterns", paragraphs=paras, sources=["Patterns"], facts={"patterns": [
            {"rank": p["rank"], "name": p["name"], "n": p["n"], "share_of_late": p["share_of_late"],
             "top_recommendation": p["recommendations"][0]} for p in pats]}, follow_ups=["What is pattern 1?"])

    # 3. focus categories
    for w, key in FOCUS_WORDS.items():
        if re.search(rf"\b{w}", norm):
            c = eng.categories["focus"][key]
            qq = eng.queue[eng.queue.product_category == key]
            paras = category_story(key, c, c["stats"])
            paras.append(f"In the current queue: {len(qq):,} orders, {int(qq.flagged.sum())} alerts, "
                         f"{int((qq.risk_level == 'CRITICAL').sum())} critical.")
            return dict(topic="category", paragraphs=paras, sources=[f"Focus category: {c['label']}"],
                        facts={"category": c["label"], "tier": c["tier"], "stats": {k: v for k, v in c["stats"].items() if k != "monthly"},
                               "playbook": c["playbook"], "queue_alerts": int(qq.flagged.sum())},
                        follow_ups=["Which focus category is most at risk?"])

    # 4. states
    code = None
    for name, c in sorted(STATE_NAMES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{name}\b", norm):
            code = c
            break
    if not code:
        m = re.search(r"\b([A-Z]{2})\b", text)
        if m and m.group(1) in STATE_NAMES.values():
            code = m.group(1)
    if code:
        st = next((s for s in eng.insights["state"] if s["key"] == code), None)
        qq = eng.queue[eng.queue.customer_state == code]
        in_pats = [p for p in pats if any(s["key"] == code and s["lift"] >= 1.3 for s in p["states"])]
        paras = [f"Deliveries to {code}: {_pct(st['rate'])} late across {st['n']:,} orders (platform {_pct(eng.insights['base_rate'])})."
                 if st else f"{code} has too few orders for a reliable late rate.",
                 f"In the queue: {len(qq):,} orders to {code}, {int(qq.flagged.sum())} with an alert."]
        if in_pats:
            paras.append(f"{code} is over-represented in " + "; ".join(f"pattern {p['rank']} ({p['name']})" for p in in_pats) + ".")
        return dict(topic="state", paragraphs=paras, sources=["Insights by state", "Patterns"],
                    facts={"state": code, "late_rate": st, "queue_orders": int(len(qq)), "queue_alerts": int(qq.flagged.sum()),
                           "patterns": [p["name"] for p in in_pats]}, follow_ups=["Which states have the most late deliveries?"])

    # 5. rankings
    if re.search(r"\b(worst|highest|most|top|riskiest)\b", ql) and not re.search(r"\bwhen\b|\bmonths?\b|season|\bpeak\b", ql):
        if "seller" in ql:
            w = eng.insights["seller_watchlist"][:5]
            paras = ["Sellers with the highest late rate (30+ orders):"] + [
                f"{s['seller_id'][:10]}… in {s['city']}, {s['state']}: {s['rate']}% of {s['n']} orders ({s['category'].replace('_', ' ')})." for s in w]
            return dict(topic="sellers", paragraphs=paras, sources=["Seller watchlist"], facts={"sellers": w}, follow_ups=[])
        if "categor" in ql or "product" in ql:
            focus = sorted(eng.categories["focus"].values(), key=lambda c: -c["stats"]["late_rate"])
            paras = ["Focus categories by late rate:"] + [f"{c['label']}: {c['stats']['late_rate']:.1f}% ({c['tier'].lower()})." for c in focus]
            return dict(topic="categories", paragraphs=paras, sources=["Focus categories"],
                        facts={"focus": [{c["label"]: c["stats"]["late_rate"]} for c in focus]}, follow_ups=["What should we do about baby products?"])
        w = eng.insights["state"][:5]
        paras = ["States with the highest late-delivery rate (100+ orders):"] + [f"{s['key']}: {s['rate']}% of {s['n']:,} orders." for s in w]
        paras.append("Most are in the North-East, far from the São Paulo seller base.")
        return dict(topic="states", paragraphs=paras, sources=["Insights by state"], facts={"states": w}, follow_ups=["What is pattern 3?"])

    # 6. simulated network
    if re.search(r"simulat|\bhub|carrier|delivery model|shipping mode|network|spoke|route", ql):
        n = eng.network
        dm = ", ".join(f"{r['key'].lower()} {r['rate']}%" for r in n["delivery_model"])
        sens = n["sensitivity"]
        paras = ["These logistics fields are SIMULATED from real fields (states, distance, promise, freight, weight), because "
                 "Olist does not publish how orders actually moved.",
                 f"Late rate by simulated delivery model: {dm}. Orders that would need more hub transfers run later, "
                 "mostly because they are the long routes.",
                 f"Adding these fields to the model changed ROC-AUC from {sens['without simulated fields']} to "
                 f"{sens['with simulated fields']} (no real lift), as expected for data derived from what the model already knows. "
                 "Real carrier and route data would be needed to improve the model this way."]
        return dict(topic="network", paragraphs=paras, sources=["Network (simulated)"], facts={"delivery_model": n["delivery_model"],
                    "transfers": n["transfers"], "sensitivity": sens}, follow_ups=["What is pattern 1?"])

    # 7. recommendations
    if re.search(r"recommend|suggest|what should|what can we|fix|improve|reduce|action|plan", ql):
        paras = ["The biggest wins, by pattern size:"]
        for p in pats[:3]:
            paras.append(f"Pattern {p['rank']} ({p['name']}, {p['share_of_late']}% of late orders): {p['recommendations'][0]}")
        paras.append("For essential products (health & beauty, baby), treat HIGH and CRITICAL alerts as same-day priority.")
        return dict(topic="recommendations", paragraphs=paras, sources=["Patterns", "Focus categories"],
                    facts={"top_patterns": [{"name": p["name"], "share": p["share_of_late"], "recs": p["recommendations"]} for p in pats[:3]]},
                    follow_ups=["What is pattern 1?", "What should we do about baby products?"])

    # 8. model quality — report the model configured for live scoring, not legacy checkout metrics.
    if re.search(r"\bauc\b|accura|precision|recall|threshold|model|xgboost|reliab|trust|good is|how well", ql):
        settings = load_settings()
        key = settings.get("scoring_model", "enriched") or "enriched"
        info = next((m for m in eng.models_info() if m["key"] == key), None)
        if key == "synthetic_operational" and info and info.get("available"):
            auc = info.get("benchmark_roc_auc")
            pr_auc = info.get("benchmark_pr_auc")
            paras = []
            try:
                q = eng.queue
                em = (getattr(eng, "enriched_meta", None) or {}).get("metrics_on_test", {})
                yq = q["is_delayed"].astype(int)
                fl = q["flagged"].astype(bool)
                tp = int((fl & (yq == 1)).sum())
                if em.get("roc_auc") is not None and fl.sum():
                    paras.append(f"For real orders, the risk queue's ROC-AUC is {float(em['roc_auc']):.3f} on the newest 20% of orders it never saw "
                                 f"(0.5 is a coin flip, 1.0 is perfect). Its {int(fl.sum()):,} alerts ({fl.mean() * 100:.0f}% of orders) were really late "
                                 f"{tp / fl.sum() * 100:.0f}% of the time and caught {tp / max(int(yq.sum()), 1) * 100:.0f}% of all late orders; about {yq.mean() * 100:.0f}% of orders are late overall. "
                                 "The 0.90 goal was not reached on real data.")
            except Exception:  # noqa: BLE001 - the answer still works without the queue figures
                pass
            paras += [
                "Live single-order and batch scoring are configured to use the dispatch operational XGBoost model.",
                f"Its held-out SIMULATED benchmark reports ROC-AUC {float(auc):.3f} and PR-AUC {float(pr_auc):.3f}. That shows the pipeline works on simulated data; it is not a measure of real-world accuracy." if auc is not None and pr_auc is not None else "The dispatch benchmark is loaded, but its saved metric report is unavailable.",
                "Operational signals and target labels in this benchmark are simulated. Treat scores as a prioritisation signal, not a calibrated real-world probability; supply actual dispatch inputs and validate against later labelled outcomes."
            ]
            return dict(topic="model", paragraphs=paras, sources=["Live scoring configuration", "Synthetic dispatch benchmark"],
                        facts={"live_scoring_model": key, "label": info.get("label"), "available": True,
                               "benchmark_roc_auc": auc, "benchmark_pr_auc": pr_auc,
                               "benchmark_type": info.get("benchmark_type")},
                        follow_ups=["Which dispatch features matter most?", "How do I supply actual operational inputs?"])
        label = (info or {}).get("label", key)
        available = bool((info or {}).get("available", False))
        paras = [f"The model configured for live scoring is {label}.",
                 "The primary real-data checkout evaluation reached ROC-AUC 0.717 on the newest 19,294 real orders from Olist in a chronological holdout; the 0.90 goal was not reached.",
                 "The 0.915 in-transit result uses SIMULATED tracking/weather/strike signals, and the 0.9433 dispatch result uses GENERATED rows and labels. Neither is verified real-world accuracy.",
                 "Single-order and batch scoring use the same configured model unless a request explicitly selects another supported model. The historical order queue remains based on stored real-Olist checkout-time features."]
        if not available:
            paras.append("This model is not currently available. Check the model files, dependency versions and application logs.")
        return dict(topic="model", paragraphs=paras, sources=["Live scoring configuration"],
                    facts={"live_scoring_model": key, "label": label, "available": available},
                    follow_ups=["Which model is selected in Settings?", "How do I validate the live model?"])

    # 9. counts / board
    if re.search(r"how many|number of|count|board|critical|alerts?|flagged", ql):
        s = eng.summary()
        paras = [f"The queue holds {s['n_queue']:,} orders: {s['bands']['CRITICAL']:,} critical, {s['bands']['HIGH']:,} high, "
                 f"{s['bands']['MEDIUM']:,} medium and {s['bands']['LOW']:,} low. {s['n_flagged']:,} have an alert."]
        if board_counts:
            paras.append(f"On the action board: {board_counts['to_review']} to review, {board_counts['investigating']} investigating, "
                         f"{board_counts['customer_informed']} customer informed, {board_counts['resolved']} resolved.")
        return dict(topic="counts", paragraphs=paras, sources=["Overview", "Action board"], facts={"bands": s["bands"],
                    "alerts": s["n_flagged"], "board": board_counts}, follow_ups=["Which states have the most late deliveries?"])

    # 10. seasonality
    if re.search(r"month|season|peak|november|black friday|march|february|trend|when", ql):
        mo = sorted(eng.insights["monthly"], key=lambda m: -m["rate"])[:3]
        paras = ["Late deliveries are strongly seasonal. Worst months: " + ", ".join(f"{m['month']} ({m['rate']}%)" for m in mo) + ".",
                 "The November 2017 Black Friday surge and the February-March 2018 spike are capacity problems: add buffer days to "
                 "promises in those months and book extra carrier capacity in advance."]
        return dict(topic="season", paragraphs=paras, sources=["Monthly trend"], facts={"worst_months": mo}, follow_ups=["What is pattern 1?"])

    # 11. method
    if re.search(r"how does|how do you|work|leak|explain|method|data|feature", ql):
        paras = ["OnTime scores each order when it is placed, using only what is known then: route distance, promised window, "
                 "timing, basket, payment, and the past late rates of the seller, category and region.",
                 "Past late rates count only orders delivered before this order was placed, so no future information leaks in. "
                 "A tuned XGBoost model gives a probability; TreeSHAP explains which factors pushed it up; the top factor picks "
                 "the suggested action.",
                 "Late orders are also grouped into patterns with k-means, so the team can fix causes rather than single orders."]
        return dict(topic="method", paragraphs=paras, sources=["How it works"], facts={}, follow_ups=["How reliable is the model?"])

    s = eng.summary()
    return dict(topic="help", paragraphs=[
        f"I can answer questions about the {s['n_queue']:,} orders in the queue, the {len(pats)} late-delivery patterns, the six "
        "focus categories, states, sellers, the simulated logistics data and the model.",
        "Try one of the suggestions below, or paste an order ID."], sources=[], facts={}, follow_ups=EXAMPLES[:4])


_REFER = re.compile(r"\b(it|this|that|them|these|those|this one|the same|that one|there)\b")
_HAS_ENTITY = re.compile(r"\b[0-9a-f]{6,32}\b|(pattern|cluster|group)\s*#?\s*\d")


def resolve_followup(question, context):
    """'What should we do about it?' after an order/pattern/state answer -> the same question with the thing named.
    Returns (question_for_lookup, note_or_None)."""
    if not context or _HAS_ENTITY.search(question.lower()):
        return question, None
    ql = question.lower()
    short = len(ql.split()) <= 7
    if not (_REFER.search(ql) or (short and re.search(r"^(and|what about|why|how about|so)\b", ql))):
        return question, None
    if context.get("order_id"):
        return f"{question} (order {context['order_id']})", f"order {context['order_id'][:8]}…"
    if context.get("pattern"):
        return f"{question} (pattern {context['pattern']})", f"pattern {context['pattern']}"
    if context.get("state"):
        return f"{question} ({context['state']})", f"state {context['state']}"
    return question, None


def _context_of(base):
    ctx = {}
    if base.get("order_id"):
        ctx["order_id"] = base["order_id"]
    f = base.get("facts") or {}
    if base["topic"] == "pattern" and isinstance(f.get("pattern"), dict):
        ctx["pattern"] = f["pattern"].get("rank")
    if base["topic"] == "state" and f.get("state"):
        ctx["state"] = f["state"]
    return ctx


def _auto_engine(s):
    st = engine_status(s)
    if st["gemini"]["available"]:
        return "gemini"
    if st["laptop"]["available"]:
        return "laptop"
    return "builtin"


def _chain(first, s, requested_auto=False):
    """Engines to try in order. An AI engine that fails hands over to the other one before the built-in answer."""
    order = [first]
    if requested_auto:
        order += [e for e in ("gemini", "laptop") if e != first and (e == "laptop" or s.get("gemini_api_key"))]
    return order


def ask(question, eng, board_counts=None, engine=None, settings=None, context=None, history=None):
    s = settings or load_settings()
    engine = engine or s["engine"]
    t0 = time.time()
    lookup_q, resolved = resolve_followup(question, context)
    base = builtin_answer(lookup_q, eng, board_counts)
    reply = {"question": question, "requested_engine": engine, "engine": "builtin", "paragraphs": base["paragraphs"],
             "sources": base["sources"], "follow_ups": base.get("follow_ups", []), "topic": base["topic"],
             "order_id": base.get("order_id"), "fallback_reason": None, "resolved_as": resolved,
             "context": _context_of(base) or (context or {})}
    if engine == "auto":
        engine = _auto_engine(s)
        reply["requested_engine"] = "auto"
    if engine in ("laptop", "gemini") and base["topic"] != "help":
        convo = ""
        if history:
            lines = [f"{'User' if h.get('role') == 'user' else 'OnTime'}: {str(h.get('text', ''))[:300]}" for h in history[-4:]]
            convo = "RECENT CONVERSATION:\n" + "\n".join(lines) + "\n\n"
        user = (convo + "FACTS (JSON):\n" + json.dumps(base["facts"], default=str)[:12000] +
                "\n\nDRAFT ANSWER FROM THE FACTS:\n" + "\n".join(base["paragraphs"]) +
                f"\n\nQUESTION: {question}\n\nWrite the final answer.")
        errors = []
        for eng_try in _chain(engine, s, requested_auto=reply["requested_engine"] == "auto"):
            try:
                text = run_engine(eng_try, s, SYSTEM_PROMPT, user)
                reply.update(engine=eng_try, paragraphs=[p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()])
                if errors:
                    reply["fallback_reason"] = f"{errors[0]} Answered by {eng_try} instead."
                break
            except EngineError as e:
                errors.append(str(e))
        else:
            reply["fallback_reason"] = " ".join(errors[:2])
    elif engine in ("laptop", "gemini", "auto"):
        reply["fallback_reason"] = "Answered by the built-in engine because the question needs a topic first."
    reply["elapsed_ms"] = int((time.time() - t0) * 1000)
    return reply


REWRITE_STYLES = {
    "executive": "Rewrite as a crisp executive summary in 3-4 sentences for a head of logistics.",
    "customer": "Rewrite as a short, warm, honest message to the customer about a possible delay. Do not mention internal "
                "scores, models, sellers' records or patterns. Offer a realistic new expectation and an apology. Max 90 words.",
    "simple": "Rewrite in very simple English a new team member would understand. Max 120 words.",
}


def rewrite(text, style, settings=None, engine=None):
    s = settings or load_settings()
    engine = engine or s["engine"]
    t0 = time.time()
    if engine not in ("laptop", "gemini"):
        return {"engine": "builtin", "text": text, "elapsed_ms": 0,
                "fallback_reason": "The built-in engine does not rewrite text. Choose the laptop or cloud AI engine in Settings."}
    instr = REWRITE_STYLES.get(style, REWRITE_STYLES["executive"])
    try:
        out = run_engine(engine, s, SYSTEM_PROMPT + " " + instr,
                         "Rewrite the following without adding any facts that are not in it:\n\n" + text)
        return {"engine": engine, "text": out, "elapsed_ms": int((time.time() - t0) * 1000), "fallback_reason": None}
    except EngineError as e:
        return {"engine": "builtin", "text": text, "elapsed_ms": int((time.time() - t0) * 1000), "fallback_reason": str(e)}


def test_engine(engine, settings=None, timeout_s=None):
    s = settings or load_settings()
    t0 = time.time()
    try:
        out = run_engine(engine, s, "Reply with exactly: OK", "Say OK.", timeout_s=timeout_s or min(s["timeout_s"], 30))
        return {"ok": True, "engine": engine, "reply": out[:60], "elapsed_ms": int((time.time() - t0) * 1000)}
    except EngineError as e:
        return {"ok": False, "engine": engine, "error": str(e), "elapsed_ms": int((time.time() - t0) * 1000)}
