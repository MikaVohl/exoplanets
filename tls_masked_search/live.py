"""Mission Control, live: run the benchmark (step 4) and watch it in the TESS visualizer as it happens.

Serves the visualizer page (reports/tess-survey/index.html) at http://127.0.0.1:8765/ and the
pipeline's event log at /events (Server-Sent Events: the page gets every event so far, then each new one
as it is written). Starts 04_benchmark_100_stars.py with --events pointing at that log and its CSVs in
output/live/ (so the committed output/benchmark_100*.csv stay untouched), opens the browser, and keeps
serving after the run ends so you can look around. Ctrl-C to stop.

Usage (from tls_masked_search/); any extra arguments go to 04_benchmark_100_stars.py:
    python live.py --limit 5
    python live.py --tics 26826078 4070275
    python live.py --tics 26826078 --port 8000 --no-browser
"""
import argparse
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


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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

    def stream(self):
        """Send the event log from the start, then follow it like `tail -f`."""
        pos, partial, quiet_since = 0, b"", time.time()
        while True:
            try:
                with open(EVENTS, "rb") as f:
                    if f.seek(0, 2) < pos:          # the file was restarted: begin again
                        pos, partial = 0, b""
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
            except FileNotFoundError:               # the run hasn't written anything yet
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
    EVENTS.unlink(missing_ok=True)  # so the page never shows the previous run
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Mission Control at {url}", flush=True)
    run = subprocess.Popen([sys.executable, "04_benchmark_100_stars.py", "--events", str(EVENTS), "--out", str(LIVE),
                            *bench_args],
                           cwd=HERE)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        run.wait()
        print(f"\nRun finished (exit code {run.returncode}). Still serving {url}; Ctrl-C to stop.", flush=True)
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        run.terminate()


if __name__ == "__main__":
    main()
