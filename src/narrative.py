"""
narrative.py
-------------
Narrative reporting: turns numbers into a short story a manager can read
(mentor's guidance: "narrate the story of what is happening for that
particular cluster or for that particular record").

Everything here is template-based and uses only figures computed from the
data, so a narrative never states something the data does not support and
works with no internet connection. The assistant can optionally rewrite a
narrative in a different tone with an AI engine, but the facts come from
here.
"""
from datetime import datetime


def _pct(x, d=0):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def _date(s):
    try:
        return datetime.fromisoformat(str(s)[:19]).strftime("%d %b %Y")
    except ValueError:
        return str(s)


def _nice(s):
    return str(s or "").replace("_", " ")


def _join(items):
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


BAND_WORDS = {"LOW": "low", "MEDIUM": "moderate", "HIGH": "high", "CRITICAL": "critical"}


def order_story(o, pattern=None, category=None):
    """o: the order dict returned by the API (features, reasons, outcome...)."""
    f = o["features"]
    route = (f"from a seller in {o.get('seller_city') or o['seller_state']}, {o['seller_state']} "
             f"to {o.get('customer_city') or o['customer_state']}, {o['customer_state']}")
    para1 = (f"This {_nice(o['product_category'])} order (R$ {f['total_order_value']:,.2f}) was placed on "
             f"{_date(o['purchased_at'])} and travels {route}, about {f['seller_customer_distance_km']:,.0f} km. "
             f"The customer was promised delivery by {_date(o['promised_by'])}, a window of "
             f"{f['promised_delivery_days']:.0f} days.")
    reasons = o.get("reasons") or []
    why = _join([r["label"].lower() for r in reasons[:3]])
    para2 = (f"OnTime puts the chance of a late delivery at {_pct(o['probability'])}, which is {BAND_WORDS.get(o['risk_level'], '')} "
             f"risk ({o['risk_level']})" + (f". The main reasons are {why}." if why else "."))
    extras = []
    if f.get("seller_historical_delay_rate") is not None:
        r = f["seller_historical_delay_rate"]
        extras.append(f"the seller had been late on {_pct(r)} of earlier orders" if r >= 0.12
                      else f"the seller's record is fine ({_pct(r)} late before)")
    if f.get("customer_state_historical_delay_rate", 0) >= 0.12:
        extras.append(f"deliveries to {o['customer_state']} are late {_pct(f['customer_state_historical_delay_rate'])} of the time")
    if f.get("purchase_month") in (11, 12, 2, 3):
        extras.append("it was placed in a peak month")
    if extras:
        para2 += " For context, " + _join(extras) + "."
    parts = [para1, para2]
    if pattern:
        parts.append(f"It looks like the \"{pattern['name']}\" pattern, which accounts for {pattern['share_of_late']}% of past late "
                     f"deliveries. In that pattern: {pattern['recommendations'][0]}")
    if category:
        parts.append(f"{category['label']} is a focus category ({category['tier'].lower()}): {category['impact']}")
    act = o["suggested_action"]
    parts.append(f"Suggested action (urgent): {act[8:]}" if act.startswith("URGENT: ") else f"Suggested action: {act}")
    out = o.get("outcome")
    if out:
        if out["was_late"]:
            parts.append(f"What actually happened: it arrived {abs(out['days_vs_promise']):.0f} days after the promised date.")
        else:
            parts.append("What actually happened: it arrived on time"
                         + (f", {abs(out['days_vs_promise']):.0f} days early." if out.get("days_vs_promise") is not None else "."))
    return parts


def pattern_story(p, total_late):
    reg = p["regions"][0] if p["regions"] else None
    st = [f"{s['key']} ({s['share']:.0f}%)" for s in p["states"][:3]]
    cats = [f"{_nice(c['key'])}" for c in p["categories"][:3]]
    logi = p["logistics"]
    model = max(logi["delivery_model"].items(), key=lambda kv: kv[1]) if logi["delivery_model"] else ("", 0)
    distinct = _join([f"{d['label']} ({'higher' if d['z'] > 0 else 'lower'} than usual)" for d in p["distinguishing"][:3]])
    return [
        f"\"{p['name']}\" groups {p['n']:,} late deliveries, {p['share_of_late']}% of all {total_late:,} late orders.",
        f"What sets it apart from late orders in general: {distinct or 'no single factor'}. "
        f"These orders travel a median {p['median_distance_km']:,} km, were promised {p['median_promised_days']:.0f} days and "
        f"actually took {p['median_actual_days']:.0f} days, ending up {p['median_days_late']:.0f} days late (median).",
        (f"They go disproportionately to the {reg['key']} ({reg['share']:.0f}% of the pattern, {reg['lift']:.1f}x the usual share). "
         if reg and reg["lift"] >= 1.2 else "")
        + (f"Most common destinations: {_join(st)}. " if st else "")
        + (f"Over-represented products: {_join(cats)}." if cats else ""),
        f"Simulated logistics view: {model[1]:.0f}% would move by {model[0].lower()}, with a median of "
        f"{logi['median_transfers']:.0f} hub transfer(s), last mile through the {logi['main_last_mile_hub']} hub.",
        "What to do: " + " ".join(p["recommendations"][:2]),
    ]


def category_story(key, c, stats):
    trend = ""
    if stats.get("late_rate_peak") is not None and stats.get("late_rate_offpeak") is not None:
        trend = (f" In peak months it rises to {stats['late_rate_peak']:.1f}%, against "
                 f"{stats['late_rate_offpeak']:.1f}% the rest of the year.")
    return [
        f"{c['label']} ({c['tier'].lower()}): {c['why']}",
        f"Across {stats['n']:,} delivered orders, {stats['late_rate']:.1f}% arrived late (platform average "
        f"{stats['platform_rate']:.1f}%), on average {stats['avg_days_late']:.0f} days late.{trend}",
        f"Why it matters: {c['impact']}",
        "Playbook: " + " ".join(c["playbook"]),
    ]


def briefing_story(b):
    """b: the dict built by the API for one period."""
    lines = [
        f"OnTime briefing for {b['period_label']}.",
        f"{b['n_orders']:,} orders were scored in this period (held-out sample). {b['n_flagged']:,} raised an alert "
        f"({b['flagged_share']:.1f}%), of which {b['n_critical']:,} were critical.",
    ]
    if b.get("prev"):
        d = b["flagged_share"] - b["prev"]["flagged_share"]
        lines.append(f"The alert rate is {'up' if d > 0 else 'down'} {abs(d):.1f} points on {b['prev']['period_label']}.")
    if b.get("top_patterns"):
        tp = b["top_patterns"][0]
        lines.append(f"Most alerts match the \"{tp['name']}\" pattern ({tp['n']} orders). {tp['recommendation']}")
    if b.get("top_drivers"):
        lines.append("Leading reasons behind alerts: " + _join([f"{d['driver'].lower()} ({d['n']})" for d in b["top_drivers"][:3]]) + ".")
    if b.get("focus"):
        worst = max(b["focus"], key=lambda r: r["flagged_share"]) if b["focus"] else None
        if worst and worst["n"] >= 10:
            lines.append(f"Among focus categories, {worst['label']} has the highest alert rate ({worst['flagged_share']:.0f}% of "
                         f"{worst['n']} orders).")
    if b.get("outcome"):
        o = b["outcome"]
        lines.append(f"Looking back: {o['flagged_late_rate']:.0f}% of this period's alerts really arrived late, and the alerts "
                     f"caught {o['caught']:.0f}% of the period's late deliveries.")
    lines.append("Recommended focus for the next period: " + " ".join(b.get("actions", [])[:3]))
    return lines


def customer_message(o):
    """A plain, honest note to the customer when a delay is likely.
    Deliberately leaves out internal scores, sellers' records and patterns."""
    f = o["features"]
    extra_days = 3 if o["risk_level"] in ("HIGH", "CRITICAL") else 2
    return (f"Hello,\n\nThank you for your order of {_nice(o['product_category'])} items. We are keeping a close eye on its "
            f"journey to {o.get('customer_city') or o['customer_state']}, and it may arrive a little after "
            f"{_date(o['promised_by'])}. We are working with the seller and the carrier to get it to you as quickly as possible, "
            f"and we will message you as soon as it ships. Please allow up to {extra_days} extra days.\n\n"
            "We are sorry for the wait, and thank you for your patience.\n\nThe customer care team")
