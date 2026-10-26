#!/usr/bin/env python3
"""
Start the OnTime web app (backend + frontend):

    python run.py        ->  http://localhost:8000
                             API docs: http://localhost:8000/docs

Settings come from environment variables, or from a .env file next to
this script (copy .env.example). Real environment variables win.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_dotenv(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


if __name__ == "__main__":
    os.chdir(HERE)  # relative paths in .env resolve from the project folder
    load_dotenv(os.path.join(HERE, ".env"))
    import uvicorn

    host = os.environ.get("ONTIME_HOST", "127.0.0.1")
    port = int(os.environ.get("ONTIME_PORT", "8000"))
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    print(f"\n  OnTime is running at http://{shown}:{port}   (API docs: /docs, stop with Ctrl+C)\n")
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=False)
