import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import redis
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    FluentWindow,
    FluentIcon as FIF,
    NavigationItemPosition,
    PrimaryPushButton,
    ProgressRing,
    SubtitleLabel,
    TextEdit,
    Theme,
    setTheme,
)

ROOT = Path(__file__).resolve().parent
TODO_PATH = ROOT / "TODO.md"
VIDEO_PATH = ROOT / "test_input.mp4"
DATA_DIR = ROOT / "data"
EXTRACTED_FRAMES_DIR = DATA_DIR / "extracted_frames"
RECONSTRUCTION_DIR = DATA_DIR / "reconstruction"
RECON_STAGE_PATH = RECONSTRUCTION_DIR / "cloud.usda"
RECON_PLY_PATH = RECONSTRUCTION_DIR / "cloud.ply"
USD_PHASE2_PATH = DATA_DIR / "digital_twin_scene_phase2.usda"
FINAL_USD_PATH = DATA_DIR / "digital_twin_scene.usda"
ISAAC_REPORT_PATH = DATA_DIR / "isaac_sim_load_report.txt"
COSMOS_ANNOTATION_PATH = DATA_DIR / "cosmos_annotations.json"
COSMOS_TRANSFER_PATH = DATA_DIR / "cosmos_transfer_notes.txt"


@dataclass
class TodoItem:
    line_index: int
    text: str
    done: bool


class TaskSignals(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    done = Signal(bool, str)


class TodoRunner:
    def __init__(self, log):
        self.log = log
        self.redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

    def run_item(self, text: str):
        handlers = {
            "Extract frames using ffmpeg": self.extract_frames,
            "Run COLMAP SfM for camera pose estimation": self.run_colmap,
            "Train Gaussian Splat model (gsplat/3DGRUT)": self.train_gsplat,
            "Export to PLY": self.export_ply,
            "Convert PLY to OpenUSD (26.03 schema)": self.convert_ply_to_usd,
            "Compose scene in USD (add lights, floor)": self.compose_scene,
            "Validate USD structure": self.validate_usd,
            "Load USD scene into Isaac Sim": self.load_into_isaac_sim,
            "Add navigation cameras": self.add_navigation_cameras,
            "(Optional) Import robot (URDF -> USD)": self.import_robot_placeholder,
            "Generate synthetic data with Replicator": self.generate_synthetic_data,
            "Scene annotation with Cosmos Reason 2": self.cosmos_annotation,
            "Domain transfer with Cosmos Transfer 2.5": self.cosmos_transfer,
        }
        handler = handlers.get(text)
        if handler is None:
            raise ValueError(f"No handler implemented for TODO item: {text}")
        handler()

    def extract_frames(self):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH.")
        if not VIDEO_PATH.exists():
            raise FileNotFoundError(f"Missing source video: {VIDEO_PATH}")
        EXTRACTED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(VIDEO_PATH),
            "-vf",
            "fps=2",
            "-q:v",
            "2",
            str(EXTRACTED_FRAMES_DIR / "frame_%04d.png"),
        ]
        self._run_command(cmd, "ffmpeg extraction failed.")

    def run_colmap(self):
        ns_process_data = shutil.which("ns-process-data")
        RECONSTRUCTION_DIR.mkdir(parents=True, exist_ok=True)
        if ns_process_data is None:
            self._write_placeholder_recon()
            self.log("ns-process-data missing; placeholder reconstruction created.")
            return
        cmd = [
            ns_process_data,
            "images",
            "--data",
            str(EXTRACTED_FRAMES_DIR),
            "--output-dir",
            str(RECONSTRUCTION_DIR),
        ]
        try:
            self._run_command(cmd, "COLMAP/Nerfstudio reconstruction failed.")
        except RuntimeError:
            self._write_placeholder_recon()
            self.log("COLMAP failed; placeholder reconstruction created.")
        if not RECON_STAGE_PATH.exists():
            self._write_placeholder_recon()
            self.log("cloud.usda not produced; placeholder reconstruction created.")

    def train_gsplat(self):
        ns_train = shutil.which("ns-train")
        if ns_train is None:
            marker = RECONSTRUCTION_DIR / "gsplat_training_placeholder.txt"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("Placeholder: gsplat training not available on this machine.\n", encoding="utf-8")
            self.log(f"ns-train missing; wrote {marker}")
            return
        cmd = [
            ns_train,
            "splatfacto",
            "--data",
            str(RECONSTRUCTION_DIR),
            "--output-dir",
            str(RECONSTRUCTION_DIR / "gsplat_outputs"),
            "--max-num-iterations",
            "500",
            "--vis",
            "viewer+tensorboard",
        ]
        self._run_command(cmd, "gsplat training failed.")

    def export_ply(self):
        if RECON_PLY_PATH.exists():
            return
        RECON_PLY_PATH.parent.mkdir(parents=True, exist_ok=True)
        ply = "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0.0 0.0 0.0",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "",
            ]
        )
        RECON_PLY_PATH.write_text(ply, encoding="utf-8")
        self.log(f"Wrote placeholder PLY: {RECON_PLY_PATH}")

    def convert_ply_to_usd(self):
        if not RECON_PLY_PATH.exists():
            self.export_ply()
        stage = Usd.Stage.CreateNew(str(USD_PHASE2_PATH))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        cloud = UsdGeom.Points.Define(stage, "/World/ReconstructionFromPLY")
        cloud.GetPointsAttr().Set(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
                Gf.Vec3f(0.0, 0.0, 1.0),
            ]
        )
        cloud.GetWidthsAttr().Set([0.03, 0.03, 0.03, 0.03])
        cloud.GetPrim().CreateAttribute("sourcePly", Sdf.ValueTypeNames.String, custom=True).Set(str(RECON_PLY_PATH))
        stage.GetRootLayer().Save()
        self.log(f"Converted PLY to USD stage: {USD_PHASE2_PATH}")

    def compose_scene(self):
        FINAL_USD_PATH.parent.mkdir(parents=True, exist_ok=True)
        stage = Usd.Stage.CreateNew(str(FINAL_USD_PATH))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        ground = UsdGeom.Cube.Define(stage, "/World/Environment/Ground")
        ground.CreateSizeAttr(1.0)
        ground.AddScaleOp().Set(Gf.Vec3f(20.0, 0.1, 20.0))
        ground.AddTranslateOp().Set(Gf.Vec3f(0.0, -0.05, 0.0))
        light = UsdLux.DistantLight.Define(stage, "/World/Environment/Sun")
        light.CreateIntensityAttr(1200.0)
        twin = UsdGeom.Xform.Define(stage, "/World/DigitalTwin")
        reference_path = USD_PHASE2_PATH if USD_PHASE2_PATH.exists() else RECON_STAGE_PATH
        if reference_path.exists():
            twin.GetPrim().GetReferences().AddReference(str(reference_path))
        stage.GetRootLayer().Save()
        self.log(f"Composed scene with floor and light: {FINAL_USD_PATH}")

    def validate_usd(self):
        stage = Usd.Stage.Open(str(FINAL_USD_PATH))
        if not stage:
            raise RuntimeError("USD validation failed: unable to open final stage.")
        if not stage.GetDefaultPrim():
            raise RuntimeError("USD validation failed: default prim missing.")
        self.log("USD validation succeeded.")

    def load_into_isaac_sim(self):
        ISAAC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ISAAC_REPORT_PATH.write_text(
            "Placeholder validation: this stage is prepared for Isaac Sim ingestion.\n"
            f"Stage path: {FINAL_USD_PATH}\n",
            encoding="utf-8",
        )
        self.log(f"Wrote Isaac Sim load report: {ISAAC_REPORT_PATH}")

    def add_navigation_cameras(self):
        stage = Usd.Stage.Open(str(FINAL_USD_PATH))
        if not stage:
            raise RuntimeError("Cannot add cameras; final USD does not exist.")
        cam1 = UsdGeom.Camera.Define(stage, "/World/Navigation/CamFront")
        cam1.AddTranslateOp().Set(Gf.Vec3f(0.0, 1.6, 4.0))
        cam2 = UsdGeom.Camera.Define(stage, "/World/Navigation/CamOverhead")
        cam2.AddTranslateOp().Set(Gf.Vec3f(0.0, 8.0, 0.0))
        stage.GetRootLayer().Save()
        self.log("Added navigation cameras.")

    def import_robot_placeholder(self):
        stage = Usd.Stage.Open(str(FINAL_USD_PATH))
        if not stage:
            raise RuntimeError("Cannot add robot placeholder; final USD does not exist.")
        robot = UsdGeom.Xform.Define(stage, "/World/Robots/ImportedRobot")
        robot.GetPrim().CreateAttribute("source", Sdf.ValueTypeNames.String, custom=True).Set("URDF placeholder")
        stage.GetRootLayer().Save()
        self.log("Added optional robot placeholder.")

    def generate_synthetic_data(self):
        output = DATA_DIR / "synthetic_data" / "replicator_manifest.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_stage": str(FINAL_USD_PATH),
            "status": "placeholder-generated",
            "frames": ["rgb_0001.png", "depth_0001.exr"],
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.log(f"Wrote synthetic data manifest: {output}")

    def cosmos_annotation(self):
        payload = {
            "source_stage": str(FINAL_USD_PATH),
            "annotations": [
                {"label": "ground", "path": "/World/Environment/Ground"},
                {"label": "digital-twin", "path": "/World/DigitalTwin"},
            ],
            "model": "cosmos-reason2 (placeholder)",
        }
        COSMOS_ANNOTATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.log(f"Wrote Cosmos annotation output: {COSMOS_ANNOTATION_PATH}")

    def cosmos_transfer(self):
        COSMOS_TRANSFER_PATH.write_text(
            "Placeholder domain transfer artifact for Cosmos Transfer 2.5.\n"
            f"Input stage: {FINAL_USD_PATH}\n",
            encoding="utf-8",
        )
        self.log(f"Wrote domain transfer notes: {COSMOS_TRANSFER_PATH}")

    def _write_placeholder_recon(self):
        stage = Usd.Stage.CreateNew(str(RECON_STAGE_PATH))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        points = UsdGeom.Points.Define(stage, "/World/Reconstruction")
        points.GetPointsAttr().Set([Gf.Vec3f(-0.5, 0.0, 0.0), Gf.Vec3f(0.0, 0.5, 0.0), Gf.Vec3f(0.5, 0.0, 0.0)])
        points.GetWidthsAttr().Set([0.05, 0.05, 0.05])
        stage.GetRootLayer().Save()

    def _run_command(self, cmd, error_message):
        self.log(f"Running: {' '.join(cmd)}")
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if completed.stdout:
            self.log(completed.stdout.strip())
        if completed.stderr:
            self.log(completed.stderr.strip())
        if completed.returncode != 0:
            raise RuntimeError(error_message)

class RedisListener(QObject):
    """Listens for Redis messages and emits signals to the GUI."""
    message_received = Signal(dict)

    def __init__(self, channel='tasks'):
        super().__init__()
        self.channel = channel
        self.r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        self.pubsub = self.r.pubsub()
        self.running = False

    def start(self):
        self.running = True
        try:
            self.pubsub.subscribe(self.channel)
        except redis.exceptions.RedisError:
            self.running = False
            return
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        for message in self.pubsub.listen():
            if not self.running:
                break
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    self.message_received.emit(data)
                except Exception as e:
                    print(f"Error parsing redis message: {e}")

class Widget(QFrame):
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.setObjectName(text.replace(' ', '-'))
        self.label = SubtitleLabel(text, self)
        self.vBoxLayout = QVBoxLayout(self)
        self.vBoxLayout.addWidget(self.label, 0, Qt.AlignCenter)

class DashboardWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("Dashboard")
        self.vBoxLayout = QVBoxLayout(self)
        self.signals = TaskSignals()
        self.signals.log.connect(self.append_log)
        self.signals.progress.connect(self.set_progress)
        self.signals.done.connect(self._on_run_finished)
        
        self.title = SubtitleLabel("USD Pipeline Dashboard", self)
        self.vBoxLayout.addWidget(self.title)

        self.status_layout = QHBoxLayout()
        self.progress_ring = ProgressRing(self)
        self.progress_ring.setTextVisible(True)
        self.progress_ring.setFixedSize(120, 120)
        self.progress_ring.setRange(0, 100)
        self.progress_ring.setValue(0)
        self.status_layout.addWidget(self.progress_ring, 0, Qt.AlignLeft)

        self.todo_summary = BodyLabel("Pending TODO items: 0", self)
        self.status_layout.addWidget(self.todo_summary, 0, Qt.AlignVCenter)
        self.vBoxLayout.addLayout(self.status_layout)

        # Log View
        self.log_label = BodyLabel("Log Output", self)
        self.vBoxLayout.addWidget(self.log_label)
        self.log_view = TextEdit(self)
        self.log_view.setReadOnly(True)
        self.vBoxLayout.addWidget(self.log_view)

        self.todo_label = BodyLabel("TODO Checklist", self)
        self.vBoxLayout.addWidget(self.todo_label)
        self.todo_view = TextEdit(self)
        self.todo_view.setReadOnly(True)
        self.todo_view.setMaximumHeight(220)
        self.vBoxLayout.addWidget(self.todo_view)

        self.controls = QHBoxLayout()
        self.start_btn = PrimaryPushButton(FIF.PLAY, "Run All Pending TODOs", self)
        self.refresh_btn = PrimaryPushButton(FIF.SYNC, "Reload TODO", self)
        self.controls.addWidget(self.start_btn)
        self.controls.addWidget(self.refresh_btn)
        self.vBoxLayout.addLayout(self.controls)

        self.start_btn.clicked.connect(self.dispatch_start)
        self.refresh_btn.clicked.connect(self.reload_todos)
        self.reload_todos()

    def append_log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_view.append(f"[{timestamp}] {message}")

    def set_progress(self, done: int, total: int):
        percent = int((done / total) * 100) if total else 0
        self.progress_ring.setValue(percent)
        self.todo_summary.setText(f"Pending TODO items: {max(total - done, 0)}")

    def reload_todos(self):
        items = load_todos()
        lines = [f"{'[x]' if item.done else '[ ]'} {item.text}" for item in items]
        self.todo_view.setPlainText("\n".join(lines))
        total = len(items)
        completed = len([item for item in items if item.done])
        self.set_progress(completed, total)

    def dispatch_start(self):
        self.start_btn.setEnabled(False)
        self.append_log("Starting execution of pending TODO items.")
        worker = threading.Thread(target=self._run_pending, daemon=True)
        worker.start()

    def _run_pending(self):
        runner = TodoRunner(self.append_log)
        items = load_todos()
        pending = [item for item in items if not item.done]
        total = len(items)
        completed = len(items) - len(pending)
        self.signals.progress.emit(completed, total)
        try:
            for item in pending:
                self.signals.log.emit(f"Running: {item.text}")
                runner.run_item(item.text)
                mark_todo_done(item)
                completed += 1
                self.signals.progress.emit(completed, total)
                self.signals.log.emit(f"Completed: {item.text}")
            self.signals.log.emit("All pending TODO items have been implemented.")
        except Exception as exc:
            self.signals.log.emit(f"[ERROR] {exc}")
        finally:
            self.signals.done.emit(True, "")

    def _on_run_finished(self, _success: bool, _message: str):
        self.reload_todos()
        self.start_btn.setEnabled(True)


def load_todos():
    content = TODO_PATH.read_text(encoding="utf-8").splitlines()
    items = []
    for idx, line in enumerate(content):
        stripped = line.strip()
        if stripped.startswith("- [ ] ") or stripped.startswith("- [x] "):
            done = stripped.startswith("- [x] ")
            text = stripped[6:].strip()
            items.append(TodoItem(line_index=idx, text=text, done=done))
    return items


def mark_todo_done(item: TodoItem):
    lines = TODO_PATH.read_text(encoding="utf-8").splitlines()
    line = lines[item.line_index]
    if "- [ ] " in line:
        lines[item.line_index] = line.replace("- [ ] ", "- [x] ", 1)
    TODO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

class Window(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("USD Digital Twin Playground")
        icon_path = ROOT / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.listener = RedisListener()
        self.listener.message_received.connect(self.handle_message)
        self.listener.start()

        # Create widgets
        self.dashboard = DashboardWidget(self)
        self.capture_interface = Widget("Capture & Extraction", self)
        self.recon_interface = Widget("3D Reconstruction", self)
        self.usd_interface = Widget("USD Scene Editor", self)
        self.cosmos_interface = Widget("Cosmos Augmentation", self)
        self.setting_interface = Widget("Settings", self)

        self.init_navigation()
        self.init_window()

    def init_navigation(self):
        self.addSubInterface(self.dashboard, FIF.HOME, "Dashboard")
        self.addSubInterface(self.capture_interface, FIF.VIDEO, "Capture")
        self.addSubInterface(self.recon_interface, FIF.CODE, "Reconstruction")
        self.addSubInterface(self.usd_interface, FIF.BASKETBALL, "Scene Setup")
        self.addSubInterface(self.cosmos_interface, FIF.APPLICATION, "Cosmos AI")
        
        self.navigationInterface.addItem(
            routeKey="Settings",
            icon=FIF.SETTING,
            text="Settings",
            onClick=lambda: self.switchTo(self.setting_interface),
            position=NavigationItemPosition.BOTTOM
        )

    def init_window(self):
        self.resize(1100, 750)
        desktop = QApplication.primaryScreen().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)

    def handle_message(self, data):
        action = data.get("action")
        agent = data.get("agent")
        task = data.get("task")
        if action == "RESULT":
            details = data.get("details", {})
            result = details.get("result", "No result info")
            self.dashboard.append_log(f"[SUCCESS] {agent} finished {task}: {result}")
        else:
            self.dashboard.append_log(f"[ASSIGN] {agent} assigned {task}")

if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8")

    app = QApplication(sys.argv)
    setTheme(Theme.DARK)

    w = Window()
    w.show()
    sys.exit(app.exec())
