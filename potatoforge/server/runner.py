"""Run the local browser UI and its FastAPI service."""

from __future__ import annotations

import socket
import threading
import time
import webbrowser
from pathlib import Path

from .app import create_app


def run_ui(static_dir: Path) -> None:
    import uvicorn

    host = "127.0.0.1"
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(static_dir=static_dir), host=host, port=port)
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        daemon=True,
    )
    thread.start()

    try:
        deadline = time.monotonic() + 10
        while thread.is_alive() and not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("UI server failed to start.")

        url = f"http://{host}:{port}"
        print(url, flush=True)
        webbrowser.open(url)
        thread.join()
    except KeyboardInterrupt:
        server.should_exit = True
        thread.join()
    finally:
        server.should_exit = True
        if thread.is_alive():
            thread.join(timeout=5)
        listener.close()
