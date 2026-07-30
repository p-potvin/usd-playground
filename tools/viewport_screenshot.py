"""Render a scene in the viewport and save a screenshot — debugging aid.

Usage:
    .venv\\Scripts\\python.exe tools\\viewport_screenshot.py <scene-rel-path> <out.png> [settle_seconds] [extra_query]

extra_query example (explicit camera pose): "px=1&py=2&pz=3&lx=0&ly=0&lz=0"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--enable-unsafe-swiftshader")

from gui.viewport import ViewportWindow, register_viewer_scheme  # noqa: E402

register_viewer_scheme()
from PySide6.QtCore import QTimer, QUrl  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def _latest_job_dir() -> Path | None:
    """Newest job holding a real splat.

    Placeholder-safe reconstruction writes a 4-vertex cloud.ply when a run
    degrades, so "has a reconstruction folder" is not enough — pick the newest
    job with a substantive cloud.splat or cloud.ply, else nothing renders.
    """
    jobs_root = ROOT / "data" / "jobs"
    if not jobs_root.is_dir():
        return None
    min_bytes = 1024 * 1024  # 1 MB — filters the placeholder PLY
    candidates = []
    for job in jobs_root.iterdir():
        recon = job / "reconstruction"
        if not recon.is_dir():
            continue
        if any(
            (recon / name).exists() and (recon / name).stat().st_size > min_bytes
            for name in ("cloud.splat", "cloud_preview.ply", "cloud.ply")
        ):
            candidates.append(job)
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def main() -> int:
    scene_rel = sys.argv[1]
    out_path = Path(sys.argv[2]).resolve()
    settle_ms = int(float(sys.argv[3]) * 1000) if len(sys.argv) > 3 else 12_000
    job_dir = _latest_job_dir()
    if job_dir is None:
        print("No job directory found under data/jobs.", file=sys.stderr)
        return 1
    print(f"job dir: {job_dir}", flush=True)

    app = QApplication(["screenshot"])
    messages: list[str] = []
    window = ViewportWindow(translate=lambda key: key)
    tab = window.panel
    tab.log.connect(lambda m: (messages.append(m), print(m, flush=True)))
    window.resize(1280, 800)
    window.show()
    tab._server.job_root = job_dir
    tab._job_dir = job_dir

    url = QUrl(tab._server.url())
    query = [f"scene={scene_rel}"]
    frame = tab._scene_framing()
    if frame:
        query.append(frame)
    if len(sys.argv) > 4 and sys.argv[4]:
        query.append(sys.argv[4])
    url.setQuery("&".join(query))
    print("loading:", url.toString(), flush=True)
    tab.web_view.load(url)

    def grab_and_quit() -> None:
        image = tab.web_view.grab()
        image.save(str(out_path))
        print(f"screenshot saved: {out_path}", flush=True)
        app.quit()

    def check() -> None:
        if any("Scene loaded" in m or "Failed" in m for m in messages):
            timer.stop()
            QTimer.singleShot(settle_ms, grab_and_quit)

    timer = QTimer()
    timer.timeout.connect(check)
    timer.start(1500)
    QTimer.singleShot(170_000, app.quit)
    app.exec()
    return 0


if __name__ == "__main__":
    sys.exit(main())
