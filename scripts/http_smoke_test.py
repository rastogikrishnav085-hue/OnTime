#!/usr/bin/env python3
"""Run a deployment-level HTTP smoke test against an already-started OnTime instance."""
from __future__ import annotations
import json
import sys
import time
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")

def request(path, method="GET", payload=None, headers=None, timeout=15):
    data = None if payload is None else (payload if isinstance(payload, bytes) else json.dumps(payload).encode())
    hdr = dict(headers or {})
    if payload is not None and not isinstance(payload, bytes):
        hdr.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(BASE + path, data=data, headers=hdr, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        content_type = resp.headers.get("Content-Type", "")
        return resp.status, body, content_type


def main():
    checks = []
    def check(name, fn):
        result = fn()
        checks.append(name)
        print(f"PASS {name}")
        return result

    check("frontend HTML", lambda: (lambda r: (_ for _ in ()).throw(AssertionError("missing OnTime HTML")) if b"OnTime" not in r[1] else r)(request("/")))
    check("frontend JavaScript", lambda: request("/assets/app.js"))
    health = json.loads(check("health endpoint", lambda: request("/api/health"))[1])
    assert health["status"] == "ok" and health["orders_in_queue"] > 0
    check("readiness diagnostics", lambda: request("/api/health/diagnostics"))
    meta = json.loads(check("model metadata", lambda: request("/api/meta"))[1])
    assert meta["live_scoring_available"] is True
    check("model listing", lambda: request("/api/models"))
    summary = json.loads(check("summary", lambda: request("/api/summary"))[1])
    assert summary["n_queue"] > 0
    check("risk queue", lambda: request("/api/orders?limit=1"))

    score = json.loads(check("single-order model scoring", lambda: request("/api/score", "POST", {
        "model": "synthetic_operational", "fields": {"order_id": "smoke-order-1", "route_km": 400,
        "weather_severity_forecast": 3, "congestion_forecast": 3, "carrier_reliability_prior": 0.7,
        "capacity_utilization_at_dispatch": 0.9, "remaining_sla_slack_hours": 8,
        "facility_backlog_at_dispatch": 20, "route_disruption_alert_at_dispatch": 0,
        "last_scan_age_hours": 3, "estimated_truck_delay_hours_at_dispatch": 2, "dispatch_lag_hours": 8}
    }))[1])
    assert score["status"] == "SCORED" and 0 <= score["probability"] <= 1 and score["scored_at"]

    _, template, _ = check("batch template", lambda: request("/api/template.csv"))
    boundary = "ontime-smoke-boundary"
    multipart = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"smoke.csv\"\r\nContent-Type: text/csv\r\n\r\n".encode()
                 + template + f"\r\n--{boundary}--\r\n".encode())
    batch = json.loads(check("batch scoring", lambda: request("/api/score/batch?model=synthetic_operational", "POST", multipart,
                                      {"Content-Type": f"multipart/form-data; boundary={boundary}"}))[1])
    assert batch["summary"]["total"] == 10
    assert batch["summary"]["scored"] + batch["summary"]["rejected"] == 10
    assert all("scored_at" in row for row in batch["rows"])
    print(f"\nHTTP smoke test passed: {len(checks)} checks against {BASE}")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
