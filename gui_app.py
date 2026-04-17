import sys
import os
import json
import redis
import threading
from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtGui import QIcon, QFont, QColor
from PySide6.QtWidgets import QApplication, QFrame, QVBoxLayout, QHBoxLayout

from qfluentwidgets import (NavigationInterface, NavigationItemPosition, NavigationWidget, 
                            MessageBox, isDarkTheme, setFont, setTheme, Theme, FluentWindow,
                            SubtitleLabel, BodyLabel, PrimaryPushButton, ProgressRing, 
                            TextEdit, FluentIcon as FIF)
from qfluentwidgets import FluentIcon

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
        self.pubsub.subscribe(self.channel)
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
        
        self.title = SubtitleLabel("USD Pipeline Dashboard", self)
        self.vBoxLayout.addWidget(self.title)

        # Status Cards
        self.status_layout = QHBoxLayout()
        self.vBoxLayout.addLayout(self.status_layout)

        # Log View
        self.log_label = BodyLabel("Log Output", self)
        self.vBoxLayout.addWidget(self.log_label)
        self.log_view = TextEdit(self)
        self.log_view.setReadOnly(True)
        self.vBoxLayout.addWidget(self.log_view)

        # Controls
        self.start_btn = PrimaryPushButton(FIF.PLAY, "Start Pipeline", self)
        self.vBoxLayout.addWidget(self.start_btn)
        self.start_btn.clicked.connect(self.dispatch_start)

    def dispatch_start(self):
        self.append_log("[USER] Manual Pipeline Trigger")
        r = redis.Redis(host='localhost', port=6379, db=0)
        task = {
            "agent": "user-gui",
            "action": "ASSIGN",
            "task": "sample_frames",
            "target": "video-specialist",
            "details": {
                "source": "test_input.mp4",
                "output_dir": "data/extracted_frames",
                "fps": 2
            }
        }
        r.publish("tasks", json.dumps(task))
        self.append_log("[ORCHESTRATOR] Dispatched sample_frames")

class Window(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("USD Digital Twin Playground")
        self.setWindowIcon(QIcon("icon.png")) # Placeholder
        
        # Initialize Redis
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
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
            onClick=None,
            position=NavigationItemPosition.BOTTOM
        )

    def init_window(self):
        self.resize(1100, 750)
        desktop = QApplication.primaryScreen().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)

    def handle_message(self, data):
        action = data.get('action')
        agent = data.get('agent')
        task = data.get('task')
        
        if action == "RESULT":
            details = data.get('details', {})
            result = details.get('result', 'No result info')
            self.dashboard.append_log(f"[SUCCESS] {agent} finished {task}: {result}")
        else:
            self.dashboard.append_log(f"[ASSIGN] {agent} assigned {task}")

if __name__ == '__main__':
    # Set default encoding to UTF-8
    sys.stdout.reconfigure(encoding='utf-8')
    
    app = QApplication(sys.argv)
    setTheme(Theme.DARK)
    
    # Custom colors from STYLE.md (#4A5459 background, #cc9b21 accent)
    # The Fluent library handles themes well, but we can customize if needed.
    
    w = Window()
    w.show()
    sys.exit(app.exec())
