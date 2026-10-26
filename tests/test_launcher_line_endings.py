"""Windows cmd.exe can misparse .bat files with bare LF line endings, so every launcher must use CRLF."""
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_bat_file_uses_crlf():
    bad = []
    for path in glob.glob(os.path.join(ROOT, "*.bat")):
        data = open(path, "rb").read()
        if data.count(b"\n") != data.count(b"\r\n"):
            bad.append(os.path.basename(path))
    assert not bad, f"bare-LF line endings in: {bad}"


def test_run_all_delegates_to_hardened_launcher():
    text = open(os.path.join(ROOT, "run_all.bat"), encoding="utf-8").read()
    assert "start_ontime.bat" in text


def test_launcher_repairs_environment_and_protects_db_on_onedrive():
    text = open(os.path.join(ROOT, "start_ontime.bat"), encoding="utf-8").read()
    assert "import fastapi" in text and "setup.bat" in text
    assert "OneDrive" in text and "ONTIME_DB" in text
