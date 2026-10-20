"""Serve a preview PLY on localhost so the SuperSplat web editor can load it."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from urllib.parse import quote, urlencode

ORIGINS = {
    "https://superspl.at",
    "https://playcanvas.com",
    "https://code.playcanvas.com",
}


def make_server(path):
    path = Path(path).resolve(strict=True)
    route = "/" + secrets.token_urlsafe(24) + "/" + quote(path.name)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def permitted(self):
            return (
                self.path == route
                and self.headers.get("Origin", "https://superspl.at") in ORIGINS
            )

        def send_headers(self, code, length=0):
            self.send_response(code)
            self.send_header(
                "Access-Control-Allow-Origin",
                self.headers.get("Origin", "https://superspl.at"),
            )
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Range")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(length))
            self.end_headers()

        def do_OPTIONS(self):
            if not self.permitted():
                self.send_error(404)
                return
            self.send_headers(204)

        def do_HEAD(self):
            if not self.permitted():
                self.send_error(404)
                return
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                self.send_error(
                    410, "Preview expired. Open SuperSplat again from the project."
                )
                return
            self.send_headers(200, size)

        def do_GET(self):
            if not self.permitted():
                self.send_error(404)
                return
            try:
                source = path.open("rb")
            except FileNotFoundError:
                self.send_error(
                    410, "Preview expired. Open SuperSplat again from the project."
                )
                return
            with source:
                self.send_headers(200, os.fstat(source.fileno()).st_size)
                try:
                    while chunk := source.read(1024 * 1024):
                        self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_port}{route}"
    return (
        server,
        dict(
            data_url=url, url="https://superspl.at/editor?" + urlencode({"load": url})
        ),
    )


def launch(path):
    path = Path(path)
    info_path = path.with_suffix(".bridge.json")
    if info_path.exists():
        from urllib.request import urlopen, Request

        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            with urlopen(
                Request(info["data_url"], method="HEAD"), timeout=1
            ) as response:
                if response.status == 200:
                    return info
        except (OSError, ValueError, KeyError):
            pass
    ready = path.with_name(path.stem + "." + secrets.token_hex(6) + ".ready.json")
    process = subprocess.Popen(
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "orchard_scene_decomposition.utils.preview_server",
            "--file",
            str(path),
            "--ready",
            str(ready),
        ],
        cwd=Path(__file__).resolve().parents[2],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    for _ in range(100):
        if ready.exists():
            info = json.loads(ready.read_text(encoding="utf-8"))
            info["pid"] = process.pid
            info_path.write_text(json.dumps(info), encoding="utf-8")
            ready.unlink()
            return info
        if process.poll() is not None:
            break
        time.sleep(0.1)
    process.terminate()
    raise RuntimeError(
        "Could not start the local preview bridge. The PLY snapshot is still available."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--ready", required=True)
    parser.add_argument("--lifetime", type=int, default=3600)
    args = parser.parse_args()
    server, info = make_server(args.file)
    ready = Path(args.ready)
    temporary_ready_path = ready.with_suffix(".tmp")
    temporary_ready_path.write_text(json.dumps(info), encoding="utf-8")
    temporary_ready_path.replace(ready)
    server.timeout = 1
    deadline = time.monotonic() + args.lifetime
    try:
        while time.monotonic() < deadline and Path(args.file).is_file():
            server.handle_request()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
