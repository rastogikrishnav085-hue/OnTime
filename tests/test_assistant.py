"""The assistant: settings, built-in answers, and the laptop (Ollama) and Gemini
engines against fake local servers, including timeouts and fallbacks."""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "backend", "app_data", "patterns.json")),
                                reason="app data not built")


@pytest.fixture(scope="module")
def eng():
    import sys
    sys.path.insert(0, ROOT)
    from backend.app.engine import get_engine
    return get_engine()


@pytest.fixture
def ai(tmp_path, monkeypatch):
    from backend.app import assistant
    monkeypatch.setattr(assistant, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return assistant


class _Fake(BaseHTTPRequestHandler):
    behaviour = {}

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            return self._send(200, {"models": [{"name": "llama3.2:latest"}]})
        self._send(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        b = _Fake.behaviour
        _Fake.last = {"path": self.path, "body": req, "headers": dict(self.headers)}
        time.sleep(b.get("sleep", 0))
        if b.get("code", 200) != 200:
            return self._send(b["code"], {"error": {"message": "API key not valid"}})
        if self.path == "/api/chat":
            return self._send(200, {"message": {"role": "assistant", "content": b.get("text", "Laptop answer.")}})
        if ":generateContent" in self.path:
            return self._send(200, {"candidates": [{"content": {"parts": [{"text": b.get("text", "Gemini answer.")}]}}]})
        self._send(404, {})


@pytest.fixture
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    _Fake.behaviour = {}
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


# ---------------- settings ----------------
def test_timeout_is_clamped_to_5_seconds_and_2_minutes(ai):
    assert ai.clamp_timeout(500) == 120 and ai.clamp_timeout(1) == 5 and ai.clamp_timeout("x") == 30
    s = ai.save_settings({"timeout_s": 999, "engine": "laptop"})
    assert s["timeout_s"] == 120 and ai.load_settings()["engine"] == "laptop"


def test_api_key_is_never_returned_in_full(ai):
    ai.save_settings({"gemini_api_key": "AIzaSyTESTKEY1234567890"})
    pub = ai.public_settings(ai.load_settings())
    assert "gemini_api_key" not in pub and pub["gemini_key_set"] and "TESTKEY" not in pub["gemini_key_hint"]
    ai.save_settings({"gemini_api_key": "__keep__", "engine": "gemini"})
    assert ai.load_settings()["gemini_api_key"] == "AIzaSyTESTKEY1234567890"


def test_unknown_engine_is_refused(ai):
    with pytest.raises(ValueError):
        ai.save_settings({"engine": "chatgpt"})


# ---------------- built-in engine ----------------
@pytest.mark.parametrize("question,topic", [
    ("What is pattern 2?", "pattern"), ("show me the clusters", "patterns"), ("what about baby products", "category"),
    ("tell me about Bahia", "state"), ("deliveries to RJ?", "state"), ("which states are worst", "states"),
    ("which sellers are worst", "sellers"), ("how reliable is the model", "model"), ("what does the simulated hub data show", "network"),
    ("what should we do", "recommendations"), ("how many critical orders are there", "counts"), ("when are delays worst", "season"),
    ("hello there", "help")])
def test_builtin_routes_questions(eng, question, topic):
    from backend.app.assistant import builtin_answer
    r = builtin_answer(question, eng)
    assert r["topic"] == topic and r["paragraphs"]


def test_builtin_explains_an_order_by_id_prefix(eng):
    from backend.app.assistant import builtin_answer
    oid = eng.queue.order_id.iloc[0]
    r = builtin_answer(f"why is {oid[:8]} risky", eng)
    assert r["topic"] == "order" and r["order_id"] == oid and "%" in r["paragraphs"][1]


# ---------------- AI engines with fake servers ----------------
def test_laptop_engine_answers_with_grounded_facts(ai, eng, fake_server):
    ai.save_settings({"engine": "laptop", "ollama_url": fake_server, "ollama_model": "llama3.2"})
    _Fake.behaviour = {"text": "Pattern 2 is local late deliveries.\n\nFix the last mile."}
    r = ai.ask("What is pattern 2?", eng)
    assert r["engine"] == "laptop" and r["fallback_reason"] is None and len(r["paragraphs"]) == 2
    sent = _Fake.last["body"]["messages"]
    assert sent[0]["role"] == "system" and "FACTS" in sent[1]["content"] and "ONLY" in sent[0]["content"]
    assert ai.engine_status(ai.load_settings())["laptop"]["available"] is True


def test_gemini_engine_sends_key_in_header(ai, eng, fake_server, monkeypatch):
    monkeypatch.setattr(ai, "GEMINI_BASE", fake_server + "/v1beta")
    ai.save_settings({"engine": "gemini", "gemini_api_key": "test-key-123456", "gemini_model": "gemini-2.5-flash"})
    r = ai.ask("how reliable is the model", eng)
    assert r["engine"] == "gemini" and r["paragraphs"] == ["Gemini answer."]
    assert "gemini-2.5-flash:generateContent" in _Fake.last["path"]
    assert _Fake.last["headers"].get("x-goog-api-key") == "test-key-123456"
    assert "key=" not in _Fake.last["path"], "the key must not be put in the URL"


def test_bad_gemini_key_falls_back_with_a_clear_reason(ai, eng, fake_server, monkeypatch):
    monkeypatch.setattr(ai, "GEMINI_BASE", fake_server + "/v1beta")
    ai.save_settings({"engine": "gemini", "gemini_api_key": "wrong-key-123456"})
    _Fake.behaviour = {"code": 400}
    r = ai.ask("What is pattern 1?", eng)
    assert r["engine"] == "builtin" and "rejected" in r["fallback_reason"] and r["paragraphs"]


def test_slow_engine_hits_the_time_limit_then_falls_back(ai, eng, fake_server):
    ai.save_settings({"engine": "laptop", "ollama_url": fake_server, "timeout_s": 5})
    _Fake.behaviour = {"sleep": 8}
    t0 = time.time()
    r = ai.ask("What is pattern 1?", eng)
    took = time.time() - t0
    assert r["engine"] == "builtin" and "5-second limit" in r["fallback_reason"]
    assert took < 7.5, f"took {took:.1f}s; the limit must be enforced"


def test_unreachable_laptop_engine_says_how_to_fix_it(ai, eng):
    ai.save_settings({"engine": "laptop", "ollama_url": "http://127.0.0.1:9"})
    r = ai.ask("What is pattern 1?", eng)
    assert r["engine"] == "builtin" and "ollama" in r["fallback_reason"].lower()


def test_rewrite_keeps_text_when_builtin(ai):
    r = ai.rewrite("Some story.", "executive", engine="builtin")
    assert r["text"] == "Some story." and r["fallback_reason"]
