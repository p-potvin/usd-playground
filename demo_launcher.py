from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BUNDLE_ROOT = _bundle_root()
APP_ROOT = _app_root()
SOURCE_VIDEO = BUNDLE_ROOT / "test_input.mp4"
OUTPUT_ROOT = APP_ROOT / "data" if getattr(sys, "frozen", False) else APP_ROOT / "data" / "demo_outputs"
EXTRACTED_FRAMES_DIR = OUTPUT_ROOT / "extracted_frames"
RECONSTRUCTION_DIR = OUTPUT_ROOT / "reconstruction"
RECONSTRUCTION_STAGE_PATH = RECONSTRUCTION_DIR / "cloud.usda"
FINAL_STAGE_PATH = OUTPUT_ROOT / "digital_twin_scene.usda"


@dataclass(frozen=True)
class DemoOutputs:
    extracted_frames_dir: Path = EXTRACTED_FRAMES_DIR
    reconstruction_dir: Path = RECONSTRUCTION_DIR
    reconstruction_stage_path: Path = RECONSTRUCTION_STAGE_PATH
    final_stage_path: Path = FINAL_STAGE_PATH


class DemoPipeline:
    def __init__(self, log):
        self.log = log
        self.outputs = DemoOutputs()

    def prepare_outputs(self) -> None:
        for path in (self.outputs.extracted_frames_dir, self.outputs.reconstruction_dir):
            if path.exists():
                shutil.rmtree(path)
        self.outputs.extracted_frames_dir.mkdir(parents=True, exist_ok=True)
        self.outputs.reconstruction_dir.mkdir(parents=True, exist_ok=True)
        if self.outputs.final_stage_path.exists():
            self.outputs.final_stage_path.unlink()
        if self.outputs.reconstruction_stage_path.exists():
            self.outputs.reconstruction_stage_path.unlink()

    def run(self) -> Path:
        if not SOURCE_VIDEO.exists():
            raise FileNotFoundError(f"Missing bundled input video: {SOURCE_VIDEO}")

        self.prepare_outputs()
        self.extract_frames()
        self.run_reconstruction()
        self.compose_stage()

        if not self.outputs.final_stage_path.exists():
            raise FileNotFoundError(f"Expected output not found: {self.outputs.final_stage_path}")

        self.log(f"Demo finished. Final USD: {self.outputs.final_stage_path}")
        return self.outputs.final_stage_path

    def extract_frames(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg was not found on PATH.")

        output_pattern = self.outputs.extracted_frames_dir / "frame_%04d.png"
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(SOURCE_VIDEO),
            "-vf",
            "fps=2",
            "-q:v",
            "2",
            str(output_pattern),
        ]
        self.log("Step 1/3: extracting frames with ffmpeg.")
        self._run_command(cmd, "Frame extraction failed.")

        frame_count = len(list(self.outputs.extracted_frames_dir.glob("*.png")))
        self.log(f"Extracted {frame_count} frames to {self.outputs.extracted_frames_dir}")

    def run_reconstruction(self) -> None:
        ns_process_data = shutil.which("ns-process-data")
        self.log("Step 2/3: reconstruction.")
        if ns_process_data is None:
            self.log("Nerfstudio/COLMAP not found. Writing placeholder reconstruction instead.")
            self._write_placeholder_reconstruction()
            return

        cmd = [
            ns_process_data,
            "images",
            "--data",
            str(self.outputs.extracted_frames_dir),
            "--output-dir",
            str(self.outputs.reconstruction_dir),
        ]

        try:
            self._run_command(cmd, "Nerfstudio reconstruction failed.")
        except RuntimeError as exc:
            self.log(f"{exc} Falling back to placeholder reconstruction.")
            self._write_placeholder_reconstruction()
            return

        if not self.outputs.reconstruction_stage_path.exists():
            self.log("Nerfstudio completed but no cloud.usda was found. Writing placeholder reconstruction.")
            self._write_placeholder_reconstruction()
        else:
            self.log(f"Reconstruction ready at {self.outputs.reconstruction_stage_path}")

    def compose_stage(self) -> None:
        self.log("Step 3/3: composing final USD stage.")
        output_path = self.outputs.final_stage_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        stage = Usd.Stage.CreateNew(str(output_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        ground = UsdGeom.Cube.Define(stage, "/World/Environment/Ground")
        ground.CreateSizeAttr(1.0)
        ground.AddScaleOp().Set(Gf.Vec3f(20.0, 0.1, 20.0))
        ground.AddTranslateOp().Set(Gf.Vec3f(0.0, -0.05, 0.0))

        light = UsdLux.DistantLight.Define(stage, "/World/Environment/Sun")
        light.CreateIntensityAttr(1000.0)
        light.CreateAngleAttr(0.53)

        twin = UsdGeom.Xform.Define(stage, "/World/DigitalTwin")
        twin.GetPrim().CreateAttribute(
            "sourceVideo", Sdf.ValueTypeNames.String, custom=True
        ).Set(str(SOURCE_VIDEO))

        if self.outputs.reconstruction_stage_path.exists():
            twin.GetPrim().GetReferences().AddReference(str(self.outputs.reconstruction_stage_path))
        else:
            points = UsdGeom.Points.Define(stage, "/World/DigitalTwin/Reconstruction")
            points.GetPointsAttr().Set(
                [
                    Gf.Vec3f(-1.0, 0.0, 0.0),
                    Gf.Vec3f(0.0, 0.8, 0.2),
                    Gf.Vec3f(1.0, 0.0, 0.0),
                    Gf.Vec3f(0.0, 0.2, 1.0),
                ]
            )
            points.GetWidthsAttr().Set([0.08, 0.08, 0.08, 0.08])

        stage.GetRootLayer().Save()
        self.log(f"Wrote final stage to {output_path}")

    def _write_placeholder_reconstruction(self) -> None:
        stage_path = self.outputs.reconstruction_stage_path
        if stage_path.exists():
            stage_path.unlink()

        stage = Usd.Stage.CreateNew(str(stage_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        points = UsdGeom.Points.Define(stage, "/World/Reconstruction")
        points.GetPointsAttr().Set(
            [
                Gf.Vec3f(-0.5, 0.0, 0.0),
                Gf.Vec3f(0.0, 0.5, 0.0),
                Gf.Vec3f(0.5, 0.0, 0.0),
            ]
        )
        points.GetWidthsAttr().Set([0.05, 0.05, 0.05])
        stage.GetRootLayer().Save()
        self.log(f"Wrote placeholder reconstruction to {stage_path}")

    def _run_command(self, cmd: list[str], error_message: str) -> None:
        self.log(f"Running: {' '.join(cmd)}")
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.stdout:
            self.log(completed.stdout.strip())
        if completed.stderr:
            self.log(completed.stderr.strip())
        if completed.returncode != 0:
            raise RuntimeError(error_message)


class DemoLauncherUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("USD Playground Demo")
        self.root.geometry("920x620")

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.pipeline = DemoPipeline(self.enqueue_log)
        self.run_thread: threading.Thread | None = None
        self.status_text = tk.StringVar(value="Ready")

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.flush_logs)

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X)

        ttk.Button(controls, text="Start Demo", command=self.start_demo).pack(side=tk.LEFT)
        ttk.Button(controls, text="Open Output Folder", command=self.open_output_folder).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(frame, textvariable=self.status_text).pack(anchor=tk.W, pady=(10, 6))

        self.log_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=30)
        self.log_widget.pack(fill=tk.BOTH, expand=True)
        self.log_widget.configure(state=tk.DISABLED)

        self.enqueue_log(f"Bundle root: {BUNDLE_ROOT}")
        self.enqueue_log(f"Output root: {OUTPUT_ROOT}")
        self.enqueue_log("The launcher runs the demo locally in one process.")

    def enqueue_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}")

    def flush_logs(self) -> None:
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_widget.configure(state=tk.NORMAL)
            self.log_widget.insert(tk.END, message + "\n")
            self.log_widget.see(tk.END)
            self.log_widget.configure(state=tk.DISABLED)
        self.root.after(100, self.flush_logs)

    def start_demo(self) -> None:
        if self.run_thread and self.run_thread.is_alive():
            return

        def _run() -> None:
            self.status_text.set("Running demo...")
            try:
                output = self.pipeline.run()
                self.status_text.set("Demo finished")
                self.enqueue_log(f"Success. Final USD: {output}")
            except Exception as exc:
                self.status_text.set("Demo failed")
                self.enqueue_log(f"[ERROR] {exc}")
                self.root.after(0, lambda: messagebox.showerror("Demo failed", str(exc)))

        self.run_thread = threading.Thread(target=_run, daemon=True)
        self.run_thread.start()

    def open_output_folder(self) -> None:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(OUTPUT_ROOT)])

    def on_close(self) -> None:
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_headless() -> int:
    def _log(message: str) -> None:
        print(message, flush=True)

    pipeline = DemoPipeline(_log)
    output = pipeline.run()
    print(f"Final USD: {output}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-window launcher for the usd-playground demo.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the demo without the Tk UI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.headless:
        return run_headless()
    app = DemoLauncherUI()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
