"""
store.py - Action board with dual-engine storage:
1. Local SQLite (zero-config, offline-first)
2. Supabase Cloud (PostgreSQL via Supabase PostgREST REST API)

Automatically detects Supabase credentials from environment or settings.json.
If Supabase is unreachable or unconfigured, it falls back to local SQLite with zero errors.
"""
import os
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import httpx

STAGES = ["to_review", "investigating", "customer_informed", "resolved"]
STAGE_LABELS = {
    "to_review": "To review",
    "investigating": "Investigating",
    "customer_informed": "Customer informed",
    "resolved": "Resolved"
}

# What each stage means, what the team does there, and the button that moves an order on.
STAGE_INFO = {
    "to_review": {
        "who": "Nobody yet", "customer_told": "No", "typical_time": "Same day",
        "meaning": "Flagged by the model; nobody has looked yet.",
        "do": "Read the risk reasons and assign an owner.",
        "steps": ["Open the order and read why it was flagged", "Assign an owner", "Decide: real risk, or a false alarm to resolve"],
        "done_when": "An owner has taken it.",
        "next_label": "Start investigating",
    },
    "investigating": {
        "who": "The owner", "customer_told": "No", "typical_time": "1-2 days",
        "meaning": "An owner is checking with the seller and the carrier.",
        "do": "Confirm the dispatch date and a new ETA; record it in the note.",
        "steps": ["Contact the seller: has it shipped?", "Ask the carrier for the last scan", "Write the confirmed ETA in the note"],
        "done_when": "You know whether it will be late and by how much.",
        "next_label": "Tell the customer",
    },
    "customer_informed": {
        "who": "The owner (waiting)", "customer_told": "Yes", "typical_time": "Until delivery",
        "meaning": "The customer has been given the updated ETA.",
        "do": "Watch for delivery, then close the order.",
        "steps": ["Send the updated ETA (use Copy message)", "Offer a remedy if the delay is long", "Close once the parcel arrives"],
        "done_when": "The parcel has been delivered or the case is settled.",
        "next_label": "Mark resolved",
    },
    "resolved": {
        "who": "Nobody", "customer_told": "Yes / not needed", "typical_time": "Closed",
        "meaning": "Closed. The real outcome is shown on the card.",
        "do": "Nothing; reopen if the problem returns.",
        "steps": ["Compare the outcome with the model's call", "Reopen if it comes back"],
        "done_when": "Closed.",
        "next_label": "",
    },
}

SUPABASE_SCHEMA_SQL = """-- Run this in the Supabase SQL Editor to initialize OnTime tables:
create table if not exists board (
    order_id   text primary key,
    stage      text not null default 'to_review',
    owner      text not null default '',
    note       text not null default '',
    created_at text not null,
    updated_at text not null
);

create table if not exists activity (
    id         bigint generated always as identity primary key,
    order_id   text not null,
    action     text not null,
    detail     text not null default '',
    at         text not null
);

-- Enable Row Level Security (optional) or allow anon read/write:
alter table board enable row level security;
alter table activity enable row level security;
create policy "Allow all board access" on board for all using (true) with check (true);
create policy "Allow all activity access" on activity for all using (true) with check (true);
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS board (
    order_id   TEXT PRIMARY KEY,
    stage      TEXT NOT NULL DEFAULT 'to_review',
    owner      TEXT NOT NULL DEFAULT '',
    note       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS activity (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    action   TEXT NOT NULL,
    detail   TEXT NOT NULL DEFAULT '',
    at       TEXT NOT NULL
);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SqliteBackend:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()

    @contextmanager
    def _conn(self):
        """Open the database, commit on success, roll back on error, and always close it.

        sqlite3's own `with` block only commits; it never closes the connection, which
        leaves the file locked on Windows (and OneDrive-synced folders).
        """
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(SCHEMA)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _log(self, c, order_id, action, detail=""):
        c.execute("INSERT INTO activity(order_id, action, detail, at) VALUES (?,?,?,?)", (order_id, action, detail, _now()))

    def items(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM board ORDER BY updated_at DESC")]

    def get(self, order_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM board WHERE order_id=?", (order_id,)).fetchone()
            return dict(r) if r else None

    def ids(self) -> set:
        with self._conn() as c:
            return {r[0] for r in c.execute("SELECT order_id FROM board")}

    def add(self, order_id: str, owner: str = "", note: str = "", auto: bool = False) -> bool:
        with self._lock, self._conn() as c:
            if c.execute("SELECT 1 FROM board WHERE order_id=?", (order_id,)).fetchone():
                return False
            now = _now()
            c.execute("INSERT INTO board(order_id, stage, owner, note, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                      (order_id, "to_review", owner.strip(), note.strip(), now, now))
            self._log(c, order_id, "added", "Added automatically (highest risk)" if auto else "Added to the board")
            return True

    def update(self, order_id: str, stage: Optional[str] = None, owner: Optional[str] = None, note: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as c:
            cur = c.execute("SELECT * FROM board WHERE order_id=?", (order_id,)).fetchone()
            if not cur:
                return None
            changes = []
            if stage is not None and stage != cur["stage"]:
                if stage not in STAGES:
                    raise ValueError(f"Unknown stage '{stage}'. Use one of: {', '.join(STAGES)}.")
                changes.append(("stage", stage, f"Moved from {STAGE_LABELS.get(cur['stage'], cur['stage'])} to {STAGE_LABELS[stage]}"))
            if owner is not None and owner.strip() != cur["owner"]:
                changes.append(("owner", owner.strip(), f"Owner set to {owner.strip() or 'nobody'}"))
            if note is not None and note.strip() != cur["note"]:
                changes.append(("note", note.strip(), "Note updated"))
            for col, val, msg in changes:
                c.execute(f"UPDATE board SET {col}=?, updated_at=? WHERE order_id=?", (val, _now(), order_id))
                self._log(c, order_id, col, msg)
            r = c.execute("SELECT * FROM board WHERE order_id=?", (order_id,)).fetchone()
            return dict(r)

    def remove(self, order_id: str) -> bool:
        with self._lock, self._conn() as c:
            n = c.execute("DELETE FROM board WHERE order_id=?", (order_id,)).rowcount
            if n:
                self._log(c, order_id, "removed", "Removed from the board")
            return bool(n)

    def activity(self, limit: int = 40) -> List[Dict[str, Any]]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM activity ORDER BY id DESC LIMIT ?", (limit,))]

    def counts(self) -> Dict[str, int]:
        with self._conn() as c:
            got = {r[0]: r[1] for r in c.execute("SELECT stage, COUNT(*) FROM board GROUP BY stage")}
            return {s: int(got.get(s, 0)) for s in STAGES}


class SupabaseBackend:
    """Supabase Cloud backend via PostgREST REST API. Zero native binary dependencies."""
    def __init__(self, url: str, key: str):
        self.url = url.rstrip("/")
        self.key = key
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }

    def _client(self):
        return httpx.Client(base_url=f"{self.url}/rest/v1", headers=self.headers, timeout=6.0)

    def test_connection(self) -> Dict[str, Any]:
        t0 = time.time()
        try:
            with self._client() as client:
                r = client.get("/board?select=order_id&limit=1")
                if r.status_code == 200:
                    return {"ok": True, "mode": "supabase", "latency_ms": int((time.time() - t0) * 1000), "detail": "Supabase connected successfully."}
                elif r.status_code == 404:
                    return {"ok": False, "mode": "supabase", "detail": "Supabase project reachable, but 'board' table is missing. Run the SQL setup script."}
                else:
                    return {"ok": False, "mode": "supabase", "detail": f"Supabase returned HTTP {r.status_code}: {r.text[:150]}"}
        except Exception as e:
            return {"ok": False, "mode": "supabase", "detail": f"Could not reach Supabase: {str(e)}"}

    def items(self) -> List[Dict[str, Any]]:
        with self._client() as client:
            r = client.get("/board?order=updated_at.desc")
            r.raise_for_status()
            return r.json()

    def get(self, order_id: str) -> Optional[Dict[str, Any]]:
        with self._client() as client:
            r = client.get(f"/board?order_id=eq.{order_id}&limit=1")
            r.raise_for_status()
            rows = r.json()
            return rows[0] if rows else None

    def ids(self) -> set:
        with self._client() as client:
            r = client.get("/board?select=order_id")
            r.raise_for_status()
            return {row["order_id"] for row in r.json()}

    def _log(self, client, order_id: str, action: str, detail: str = ""):
        client.post("/activity", json={"order_id": order_id, "action": action, "detail": detail, "at": _now()})

    def add(self, order_id: str, owner: str = "", note: str = "", auto: bool = False) -> bool:
        with self._client() as client:
            # Check if exists
            chk = client.get(f"/board?order_id=eq.{order_id}&limit=1")
            if chk.status_code == 200 and chk.json():
                return False
            now = _now()
            payload = {
                "order_id": order_id,
                "stage": "to_review",
                "owner": owner.strip(),
                "note": note.strip(),
                "created_at": now,
                "updated_at": now
            }
            r = client.post("/board", json=payload)
            r.raise_for_status()
            self._log(client, order_id, "added", "Added automatically (highest risk)" if auto else "Added to the board")
            return True

    def update(self, order_id: str, stage: Optional[str] = None, owner: Optional[str] = None, note: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._client() as client:
            chk = client.get(f"/board?order_id=eq.{order_id}&limit=1")
            if chk.status_code != 200 or not chk.json():
                return None
            cur = chk.json()[0]
            patch: Dict[str, Any] = {"updated_at": _now()}
            logs = []
            if stage is not None and stage != cur["stage"]:
                if stage not in STAGES:
                    raise ValueError(f"Unknown stage '{stage}'. Use one of: {', '.join(STAGES)}.")
                patch["stage"] = stage
                logs.append(("stage", f"Moved from {STAGE_LABELS.get(cur['stage'], cur['stage'])} to {STAGE_LABELS[stage]}"))
            if owner is not None and owner.strip() != cur["owner"]:
                patch["owner"] = owner.strip()
                logs.append(("owner", f"Owner set to {owner.strip() or 'nobody'}"))
            if note is not None and note.strip() != cur["note"]:
                patch["note"] = note.strip()
                logs.append(("note", "Note updated"))
            r = client.patch(f"/board?order_id=eq.{order_id}", json=patch)
            r.raise_for_status()
            for col, msg in logs:
                self._log(client, order_id, col, msg)
            res = r.json()
            return res[0] if res else None

    def remove(self, order_id: str) -> bool:
        with self._client() as client:
            r = client.delete(f"/board?order_id=eq.{order_id}")
            if r.status_code == 200:
                self._log(client, order_id, "removed", "Removed from the board")
                return True
            return False

    def activity(self, limit: int = 40) -> List[Dict[str, Any]]:
        with self._client() as client:
            r = client.get(f"/activity?order=id.desc&limit={limit}")
            r.raise_for_status()
            return r.json()

    def counts(self) -> Dict[str, int]:
        with self._client() as client:
            r = client.get("/board?select=stage")
            r.raise_for_status()
            tally: Dict[str, int] = {s: 0 for s in STAGES}
            for row in r.json():
                stg = row.get("stage")
                if stg in tally:
                    tally[stg] += 1
            return tally


class BoardStore:
    """Unified Board Store with SQLite and Supabase mode, plus automatic fallback."""
    def __init__(self, sqlite_path: str):
        self.path = sqlite_path
        self.sqlite = SqliteBackend(sqlite_path)
        self.supabase: Optional[SupabaseBackend] = None
        self._init_supabase()

    def _init_supabase(self):
        url = os.environ.get("SUPABASE_URL", "").strip()
        key = os.environ.get("SUPABASE_KEY", "").strip() or os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
        mode = os.environ.get("DATABASE_MODE", "auto").strip().lower()

        # Check settings.json if not in env
        settings_path = os.environ.get("ONTIME_SETTINGS", os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json"))
        if os.path.exists(settings_path) and (not url or not key):
            try:
                with open(settings_path, encoding="utf-8") as f:
                    data = json.load(f)
                    if not url:
                        url = data.get("supabase_url", "").strip()
                    if not key:
                        key = data.get("supabase_key", "").strip()
                    if mode == "auto":
                        mode = data.get("database_mode", "auto").strip().lower()
            except Exception:
                pass

        if mode != "sqlite" and url and key:
            self.supabase = SupabaseBackend(url, key)
            self.mode = "supabase"
        else:
            self.supabase = None
            self.mode = "sqlite"

    @property
    def active_mode(self) -> str:
        return "supabase" if self.supabase is not None else "sqlite"

    def status(self) -> Dict[str, Any]:
        if self.supabase:
            t = self.supabase.test_connection()
            return {
                "mode": "supabase",
                "configured": True,
                "connected": t["ok"],
                "detail": t["detail"],
                "url": self.supabase.url,
                "sqlite_path": self.path
            }
        return {
            "mode": "sqlite",
            "configured": True,
            "connected": True,
            "detail": f"Local SQLite database at {self.path}",
            "sqlite_path": self.path
        }

    # Proxy methods with automatic SQLite fallback
    def items(self):
        if self.supabase:
            try:
                return self.supabase.items()
            except Exception:
                pass
        return self.sqlite.items()

    def get(self, order_id: str):
        if self.supabase:
            try:
                return self.supabase.get(order_id)
            except Exception:
                pass
        return self.sqlite.get(order_id)

    def ids(self):
        if self.supabase:
            try:
                return self.supabase.ids()
            except Exception:
                pass
        return self.sqlite.ids()

    def add(self, order_id: str, owner: str = "", note: str = "", auto: bool = False):
        if self.supabase:
            try:
                return self.supabase.add(order_id, owner, note, auto)
            except Exception:
                pass
        return self.sqlite.add(order_id, owner, note, auto)

    def update(self, order_id: str, stage: Optional[str] = None, owner: Optional[str] = None, note: Optional[str] = None):
        if self.supabase:
            try:
                return self.supabase.update(order_id, stage, owner, note)
            except Exception:
                pass
        return self.sqlite.update(order_id, stage, owner, note)

    def remove(self, order_id: str):
        if self.supabase:
            try:
                return self.supabase.remove(order_id)
            except Exception:
                pass
        return self.sqlite.remove(order_id)

    def activity(self, limit: int = 40):
        if self.supabase:
            try:
                return self.supabase.activity(limit)
            except Exception:
                pass
        return self.sqlite.activity(limit)

    def counts(self):
        if self.supabase:
            try:
                return self.supabase.counts()
            except Exception:
                pass
        return self.sqlite.counts()
