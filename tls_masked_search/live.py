"""Mission Control, live: run the benchmark (step 4) and watch it in the TESS visualizer as it happens.

Serves the visualizer page (reports/tess-survey/index.html) at http://127.0.0.1:8765/ and opens the
browser. Click "Run pipeline" there to start 04_benchmark_100_stars.py, or pass its arguments here to
start a run right away. The page follows the run's event log at /events (Server-Sent Events: every event
so far, then each new one as it is written). Runs write the log and their CSVs to output/live/ (so the
committed output/benchmark_100*.csv stay untouched); a new run replaces the one in progress. Keeps
serving after a run ends so you can look around or start another. Ctrl-C to stop.

Endpoints: GET / (the page), GET /events, GET /targets (tic, hostname, group of each benchmark star),
GET /run (the run's state: idle, running, finished, failed or stopped, and its exit code),
POST /run with JSON {"tics": [...]} or {"limit": N} (stops the run in progress, starts a new one),
POST /stop.

Usage (from tls_masked_search/); any extra arguments go to 04_benchmark_100_stars.py:
    python live.py                     # serve and wait for the page's Run pipeline button
    python live.py --limit 5
    python live.py --tics 26826078 4070275
    python live.py --tics 26826078 --port 8000 --no-browser
"""
import argparse
import csv
import json
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).parent
PAGE = HERE.parent / "reports" / "tess-survey" / "index.html"
LIVE = HERE / "output" / "live"   # this run's event log and CSVs (git-ignored)
EVENTS = LIVE / "events.jsonl"
TARGETS = [{"tic": int(r["tic"]), "hostname": r["hostname"], "group": r["catalog_group"]}
           for r in csv.DictReader((HERE.parent / "data" / "benchmark" / "targets.csv").read_text().splitlines())]

lock = threading.RLock()
run = None   # the latest benchmark process
gen = 0      # run counter: an /events stream that sees it change starts over on the new log


def start(bench_args):
    """Stop the run in progress, clear the event log and start a new run."""
    global run, gen
    with lock:
        stop()
        EVENTS.write_bytes(b"")
        gen += 1
        run = subprocess.Popen([sys.executable, "04_benchmark_100_stars.py", "--events", str(EVENTS),
                                "--out", str(LIVE), *bench_args], cwd=HERE)
        run.stopped = False


def stop():
    with lock:
        if run and run.poll() is None:
            run.stopped = True
            run.terminate()
            run.wait()


def status():
    if run is None:
        return {"state": "idle", "code": None}
    code = run.poll()
    state = "running" if code is None else "stopped" if run.stopped else "finished" if code == 0 else "failed"
    return {"state": state, "code": code}


def run_args(body):
    """04_benchmark_100_stars.py arguments for a POST /run body; ValueError if it isn't valid."""
    key, value = next(iter(body.items())) if isinstance(body, dict) and len(body) == 1 else (None, None)
    values = value if key == "tics" and isinstance(value, list) else [value]
    if key not in ("tics", "limit") or not values or any(type(v) is not int for v in values):
        raise ValueError('send {"tics": [TIC IDs]} or {"limit": N}, whole numbers only')
    if key == "limit":
        if not 1 <= value <= len(TARGETS):
            raise ValueError(f"limit must be 1 to {len(TARGETS)}")
        return ["--limit", str(value)]
    unknown = sorted(set(values) - {t["tic"] for t in TARGETS})
    if unknown:
        raise ValueError(f"not in data/benchmark/targets.csv: {unknown}")
    return ["--tics", *map(str, dict.fromkeys(values))]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            self.send_body(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/targets":
            self.send_json(TARGETS)
        elif path == "/run":
            self.send_json(status())
        elif path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                self.stream()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the page was closed
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in ("/run", "/stop"):
            return self.send_error(404)
        # JSON only: another site's page can't send it here without a CORS preflight, which is never allowed.
        if self.headers.get_content_type() != "application/json":
            return self.send_json({"error": "send application/json"}, 415)
        if path == "/stop":
            stop()
            return self.send_json(status())
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= 65536:
                raise ValueError("empty or too large request")
            bench_args = run_args(json.loads(self.rfile.read(n)))
        except ValueError as error:  # also bad JSON (JSONDecodeError) and bad Content-Length
            return self.send_json({"error": str(error)}, 400)
        start(bench_args)
        self.send_json(status())

    def send_body(self, body, content_type, code=200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, code=200):
        self.send_body(json.dumps(obj).encode(), "application/json", code)

    def stream(self):
        """Send the event log from the start, then follow it like `tail -f`; start over for a new run."""
        pos, partial, seen, quiet_since = 0, b"", gen, time.time()
        while True:
            if seen != gen:                         # a new run cleared the log
                pos, partial, seen = 0, b"", gen
            try:
                with open(EVENTS, "rb") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
            except FileNotFoundError:               # no run yet
                chunk = b""
            *lines, partial = (partial + chunk).split(b"\n")
            for line in lines:
                if line:
                    self.wfile.write(b"data: " + line + b"\n\n")
            if chunk:
                quiet_since = time.time()
            elif time.time() - quiet_since > 15:    # keep-alive; also notices closed pages
                self.wfile.write(b": ping\n\n")
                quiet_since = time.time()
            else:
                time.sleep(0.2)

    def log_message(self, *args):
        pass  # keep the terminal for the pipeline's own output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args, bench_args = parser.parse_known_args()

    LIVE.mkdir(parents=True, exist_ok=True)
    EVENTS.unlink(missing_ok=True)  # so the page never shows the previous session's run
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Mission Control at {url}" + ("" if bench_args else ": click Run pipeline"), flush=True)
    if bench_args:
        start(bench_args)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop()


if __name__ == "__main__":
    main()
