"""Settings dialog for VaultWares Studio.

Houses remote compute config (HF token, flavor, artifact repo),
strict mode toggle, and dependency health check.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.strings import t as _t


class SettingsDialog(QDialog):
    def __init__(self, parent=None, strict_mode: bool = False, on_strict_toggled=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(560, 480)
        self._strict_mode = strict_mode
        self._on_strict_toggled = on_strict_toggled

        layout = QVBoxLayout(self)
        tabs = QTabWidget(self)
        layout.addWidget(tabs)

        tabs.addTab(self._build_remote_tab(), "Remote Compute")
        tabs.addTab(self._build_health_tab(), "Dependency Health")
        tabs.addTab(self._build_general_tab(), "General")

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close, parent=self
        )
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    # -- Remote Compute tab ----------------------------------------------------

    def _build_remote_tab(self) -> QWidget:
        from vaultwares_studio.runners import HfJobsConfig, get_hf_token, set_hf_token

        tab = QWidget(self)
        form = QFormLayout(tab)
        form.setContentsMargins(20, 20, 20, 20)
        form.setSpacing(12)

        config = HfJobsConfig.load()

        self._hf_token_edit = QLineEdit(self)
        current_token = get_hf_token() or ""
        masked = "•" * 20 if current_token else ""
        self._hf_token_edit.setPlaceholderText("Paste HF token here…")
        self._hf_token_edit.setText(masked)
        self._hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow(_t("hf_token_label"), self._hf_token_edit)

        self._hf_repo_edit = QLineEdit(self)
        self._hf_repo_edit.setPlaceholderText("user/vw-studio-artifacts (blank = auto)")
        self._hf_repo_edit.setText(config.artifact_repo)
        form.addRow(_t("hf_repo_label"), self._hf_repo_edit)

        self._flavor_combo = QComboBox(self)
        self._flavor_combo.addItem("CPU Upgrade (cpu-upgrade) — $0.04/h", "cpu-upgrade")
        self._flavor_combo.addItem("L4 (l4x1) — $0.80/h", "l4x1")
        self._flavor_combo.addItem("A10G Small (a10g-small) — $1.00/h", "a10g-small")
        self._flavor_combo.addItem("A100 Large (a100-large) — $4.00/h", "a100-large")
        self._flavor_combo.addItem("ZeroGPU — $6.00/h (20 min MAX, approval required)", "zerogpu")
        idx = self._flavor_combo.findData(config.default_flavor)
        if idx >= 0:
            self._flavor_combo.setCurrentIndex(idx)
        self._flavor_combo.currentIndexChanged.connect(self._on_flavor_changed)
        form.addRow(_t("hf_flavor_label"), self._flavor_combo)

        self._zerogpu_warning = QLabel("", self)
        self._zerogpu_warning.setWordWrap(True)
        form.addRow("", self._zerogpu_warning)

        self._remote_enabled = QCheckBox("Enable remote compute", self)
        self._remote_enabled.setChecked(config.enabled)
        form.addRow("", self._remote_enabled)

        btn_row = QHBoxLayout()
        save_btn = QPushButton(_t("save_remote"), self)
        save_btn.setProperty("class", "primary")
        save_btn.clicked.connect(self._save_remote)
        btn_row.addWidget(save_btn)

        test_btn = QPushButton(_t("test_remote"), self)
        test_btn.clicked.connect(self._test_remote)
        btn_row.addWidget(test_btn)

        btn_row.addStretch(1)
        form.addRow("", btn_row)

        self._remote_status = QLabel("", self)
        self._remote_status.setWordWrap(True)
        self._remote_status.setStyleSheet("color: #6BE675;")
        form.addRow("", self._remote_status)

        self._hf_config = config
        return tab

    def _on_flavor_changed(self, index: int):
        """Show warning when ZeroGPU is selected."""
        flavor = self._flavor_combo.currentData()
        if flavor == "zerogpu":
            self._zerogpu_warning.setText(
                "⚠ ZeroGPU bills at 6x the rate of a Blackwell GPU when over limit.\n"
                "20-minute MAX. Manual approval required for every job."
            )
            self._zerogpu_warning.setStyleSheet("color: #FF6B7A; font-weight: bold;")
        else:
            self._zerogpu_warning.setText("")

    def _save_remote(self):
        from vaultwares_studio.runners import HfJobsConfig, set_hf_token

        token = self._hf_token_edit.text().strip()
        if token and not token.startswith("•"):
            set_hf_token(token)

        self._hf_config.artifact_repo = self._hf_repo_edit.text().strip()
        self._hf_config.default_flavor = self._flavor_combo.currentData()
        self._hf_config.enabled = self._remote_enabled.isChecked()
        self._hf_config.save()
        self._remote_status.setText(f"✓ {_t('remote_saved')}")
        self._remote_status.setStyleSheet("color: #6BE675;")

    def _test_remote(self):
        from vaultwares_studio.runners import run_echo_smoke_test, get_hf_token, HfJobsConfig

        if not get_hf_token():
            self._remote_status.setText(f"✗ {_t('remote_no_token')}")
            self._remote_status.setStyleSheet("color: #FF6B7A;")
            return
        self._remote_status.setText("Launching echo test job…")
        self._remote_status.setStyleSheet("color: rgba(237, 230, 255, 0.72);")
        try:
            config = HfJobsConfig.load()
            result = run_echo_smoke_test(
                config=config,
                confirm_cost=lambda _: True,
                log=lambda msg: None,
            )
            self._remote_status.setText(f"✓ Echo test: {result}")
            self._remote_status.setStyleSheet("color: #6BE675;")
        except Exception as exc:
            self._remote_status.setText(f"✗ Echo test failed: {exc}")
            self._remote_status.setStyleSheet("color: #FF6B7A;")

    # -- Dependency Health tab -------------------------------------------------

    def _build_health_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)

        refresh_btn = QPushButton(_t("refresh_health"), self)
        refresh_btn.clicked.connect(self._refresh_health)
        layout.addWidget(refresh_btn)

        self._health_table = QTableWidget(0, 4, self)
        self._health_table.setHorizontalHeaderLabels(["Name", "Kind", "Status", "Detail"])
        self._health_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._health_table.verticalHeader().setVisible(False)
        layout.addWidget(self._health_table, 1)

        self._refresh_health()
        return tab

    def _refresh_health(self):
        from vaultwares_studio.pipeline import build_dependency_health

        table = self._health_table
        rows = build_dependency_health()
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            name_item = QTableWidgetItem(row["name"])
            table.setItem(i, 0, name_item)
            kind_item = QTableWidgetItem(row["kind"])
            table.setItem(i, 1, kind_item)
            status_item = QTableWidgetItem(row["status"])
            if row["status"] == "ok":
                status_item.setForeground(QColor("#6BE675"))
            else:
                status_item.setForeground(QColor("#FF6B7A"))
            table.setItem(i, 2, status_item)
            detail_item = QTableWidgetItem(row["detail"])
            table.setItem(i, 3, detail_item)

    # -- General tab -----------------------------------------------------------

    def _build_general_tab(self) -> QWidget:
        tab = QWidget(self)
        form = QFormLayout(tab)
        form.setContentsMargins(20, 20, 20, 20)
        form.setSpacing(12)

        self._strict_checkbox = QCheckBox("Enable strict tool mode", self)
        self._strict_checkbox.setChecked(self._strict_mode)
        self._strict_checkbox.toggled.connect(self._on_strict_toggled_slot)
        form.addRow(_t("enable_strict"), self._strict_checkbox)

        note = QLabel(
            "Strict mode: missing heavy tools (COLMAP, ns-train) fail the stage\n"
            "instead of falling back to placeholder outputs."
        )
        note.setStyleSheet("color: rgba(237, 230, 255, 0.72);")
        form.addRow("", note)

        return tab

    def _on_strict_toggled_slot(self, checked: bool):
        self._strict_mode = checked
        if self._on_strict_toggled:
            self._on_strict_toggled(checked)
