"""Launcher for Barca Vision Labeler.

Initializes the index, finds an available port (default 5000), starts the local
Flask server, and opens the default web browser automatically.
"""

import os
import sys
import time
import socket
import webbrowser
import threading
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from app import app, initialize  # noqa: E402


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def is_labeler_already_running(port: int = 5000) -> bool:
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/info", headers={"User-Agent": "BarcaCheck"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def open_browser(port: int):
    time.sleep(1.2)
    url = f"http://127.0.0.1:{port}"
    print(f"[Labeler] Opening browser at {url} ...")
    webbrowser.open(url)


def main():
    print("=" * 60)
    print("  BARCA VISION LABELER -- TACTICAL ANNOTATION STUDIO")
    print("=" * 60)

    # Check if already running on port 5000
    if is_labeler_already_running(5000):
        print("[Labeler] Barca Vision Labeler is ALREADY running on http://127.0.0.1:5000")
        print("[Labeler] Re-opening web browser tab...")
        webbrowser.open("http://127.0.0.1:5000")
        print("Ready! You can switch to your open browser window.")
        return

    # Find free port starting at 5000
    port = 5000
    while is_port_in_use(port) and port < 5010:
        print(f"[Labeler] Port {port} is occupied, trying port {port + 1}...")
        port += 1

    print("Building high-performance frame index...")
    initialize()
    print(f"Ready! Starting web server on http://127.0.0.1:{port}")
    print("Press Ctrl+C in terminal to stop.")
    print("=" * 60)

    # Launch browser thread
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    # Run server
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()

