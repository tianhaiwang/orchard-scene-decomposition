"""Qt workbench that runs backend requests in a separate worker process."""

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path
import sys
import tempfile
from PyQt5 import QtCore, QtGui, QtWidgets
from ..config import defaults, validate, SCHEMA, STAGES, TITLES

ROOT = Path(__file__).resolve().parents[2]
GROUPS = [
    ("1 Load", ["load"]),
    ("2 Ground", ["ground"]),
    ("3 Poles", ["poles"]),
    ("4 Wires", ["wires"]),
    (
        "5 Trees & review",
        [
            "roi",
            "denoise",
            "stumps",
            "trunks",
            "match_trunks",
            "branches",
            "assign_branches",
            "refill",
        ],
    ),
    ("6 Preview & export", []),
]


class ValueDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        if index.column() != 1:
            return None
        return super().createEditor(parent, option, index)


class Window(QtWidgets.QMainWindow):
    def __init__(self, project=None, backend=None, wsl_distro=None):
        super().__init__()
        self.setWindowTitle("Orchard Scene Decomposition")
        self.resize(1200, 900)
        self.project = project
        self.config = defaults()
        self.revision = None
        self.proc = None
        self.viewer = False
        self.replacing = False
        self.wsl_distro = wsl_distro
        self.temp = tempfile.TemporaryDirectory(
            prefix="orchard-scene-decomposition-ui-"
        )
        self.counter = 0
        self.status = {}
        body = QtWidgets.QWidget()
        self.setCentralWidget(body)
        layout = QtWidgets.QVBoxLayout(body)
        headline = QtWidgets.QLabel("Orchard Scene Decomposition Annotation Workbench")
        headline.setStyleSheet("font-size:24px; font-weight:600; color:#173c48")
        layout.addWidget(headline)
        self.path_label = QtWidgets.QLabel(
            "Create a project and load enhanced 3DGS PLY. Source data stays read-only."
        )
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)
        project_controls = QtWidgets.QHBoxLayout()
        layout.addLayout(project_controls)
        self.backend = QtWidgets.QLineEdit(backend or sys.executable)
        project_controls.addWidget(
            QtWidgets.QLabel(
                "Backend Python" + (" (WSL: " + wsl_distro + ")" if wsl_distro else "")
            )
        )
        project_controls.addWidget(self.backend)
        self.button(project_controls, "Choose Python", self.choose_python)
        self.button(project_controls, "New project", self.create_project)
        self.button(project_controls, "Open project", self.open_project)
        project_menu = self.menuBar().addMenu("Project")
        project_menu.addAction(
            "Recover lock",
            lambda: self.request(
                {"action": "recover-lock"},
                lambda _: self.log.appendPlainText("Lock checked and recovered."),
            ),
        )
        panel_splitter = QtWidgets.QSplitter()
        layout.addWidget(panel_splitter, 1)
        self.steps = QtWidgets.QListWidget()
        self.steps.setMaximumWidth(235)
        for title, _ in GROUPS:
            self.steps.addItem(title)
        panel_splitter.addWidget(self.steps)
        stage_panel = QtWidgets.QWidget()
        panel_splitter.addWidget(stage_panel)
        right_layout = QtWidgets.QVBoxLayout(stage_panel)
        stage_controls = QtWidgets.QHBoxLayout()
        right_layout.addLayout(stage_controls)
        self.stage = QtWidgets.QComboBox()
        stage_controls.addWidget(self.stage, 1)
        self.run_button = self.button(stage_controls, "Run stage", self.run_stage)
        self.cancel_button = self.button(stage_controls, "Cancel task", self.cancel)
        self.state_label = QtWidgets.QLabel("")
        self.state_label.setWordWrap(True)
        right_layout.addWidget(self.state_label)
        self.facility_round_controls = QtWidgets.QWidget()
        round_controls_layout = QtWidgets.QHBoxLayout(self.facility_round_controls)
        round_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.append_round_button = self.button(
            round_controls_layout,
            "Append round",
            lambda: self.begin_facility_round("append"),
        )
        self.replace_round_button = self.button(
            round_controls_layout,
            "Replace result",
            lambda: self.begin_facility_round("replace"),
        )
        self.cancel_round_button = self.button(
            round_controls_layout, "Cancel round", self.cancel_facility_round
        )
        self.append_round_button.setToolTip(
            "Keep accepted results and pick remaining points. Settings apply to this round"
        )
        self.replace_round_button.setToolTip(
            "Replace facilities after success. Older revisions remain available"
        )
        right_layout.addWidget(self.facility_round_controls)
        self.ground_strategy = QtWidgets.QWidget()
        ground_controls_layout = QtWidgets.QHBoxLayout(self.ground_strategy)
        ground_controls_layout.setContentsMargins(0, 0, 0, 0)
        ground_controls_layout.addWidget(QtWidgets.QLabel("Ground method"))
        self.ground_method = QtWidgets.QComboBox()
        ground_controls_layout.addWidget(self.ground_method, 1)
        for method in SCHEMA["ground.method"]["choices"]:
            self.ground_method.addItem(
                SCHEMA["ground.method"]["choice_labels"][method], method
            )
        right_layout.addWidget(self.ground_strategy)
        self.ground_method.currentIndexChanged.connect(self.change_ground_method)
        self.parameters = QtWidgets.QTreeWidget()
        self.parameters.setHeaderLabels(["Parameter / unit", "Value (double-click)"])
        self.parameters.setColumnWidth(0, 390)
        self.parameters.setColumnWidth(1, 160)
        right_layout.addWidget(self.parameters, 1)
        self.parameters.setItemDelegate(ValueDelegate(self.parameters))
        self.parameters.setMinimumHeight(230)
        parameter_controls = QtWidgets.QHBoxLayout()
        right_layout.addLayout(parameter_controls)
        self.button(parameter_controls, "Save parameters", self.save_params)
        self.button(parameter_controls, "Reset stage defaults", self.reset_params)
        self.button(parameter_controls, "Import JSON", self.import_config)
        self.button(parameter_controls, "Export JSON", self.export_config)
        self.button(parameter_controls, "Edit saved picks", self.edit_picks)
        self.actions = QtWidgets.QGroupBox("Interaction & review")
        right_layout.addWidget(self.actions)
        interaction_layout = QtWidgets.QGridLayout(self.actions)
        buttons = [
            ("Pick facilities / ROI", self.pick),
            ("Preview result", lambda: self.start_viewer("preview")),
            ("Review fragments", lambda: self.start_viewer("transfer")),
            ("Undo", lambda: self.edit("undo")),
            ("Redo", lambda: self.edit("redo")),
            ("Transfer mode", lambda: self.viewer_command({"action": "transfer_mode"})),
            (
                "Rectangle split mode",
                lambda: self.viewer_command({"action": "split_mode"}),
            ),
            ("Apply split", lambda: self.viewer_command({"action": "split"})),
            ("Clear selection", lambda: self.viewer_command({"action": "clear"})),
        ]
        for button_index, (label, callback) in enumerate(buttons):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(callback)
            interaction_layout.addWidget(button, button_index // 3, button_index % 3)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("Isolate labels, e.g. 4,5 (blank = all)")
        interaction_layout.addWidget(self.filter, 3, 0, 1, 2)
        self.target = QtWidgets.QSpinBox()
        self.target.setRange(4, 2147483647)
        interaction_layout.addWidget(self.target, 3, 2)
        button = QtWidgets.QPushButton("Transfer fragment to selected tree label")
        button.clicked.connect(
            lambda: self.viewer_command(
                {"action": "target", "label": self.target.value()}
            )
        )
        interaction_layout.addWidget(button, 4, 0, 1, 3)
        hint = QtWidgets.QLabel(
            "Shift + click to pick. For review, pick a source fragment, then a target tree.\nFor splitting, Shift + drag a rectangle, then Apply split. Press Q to close the 3D view."
        )
        hint.setWordWrap(True)
        interaction_layout.addWidget(hint, 5, 0, 1, 3)
        color_controls = QtWidgets.QHBoxLayout()
        right_layout.addLayout(color_controls)
        color_controls.addWidget(QtWidgets.QLabel("Point colors"))
        self.color_mode = QtWidgets.QComboBox()
        self.color_mode.addItem("Source RGB (approx.)", "rgb")
        self.color_mode.addItem("Segmentation labels", "labels")
        color_controls.addWidget(self.color_mode)
        self.color_mode.currentIndexChanged.connect(self.change_color_mode)
        export_controls = QtWidgets.QHBoxLayout()
        right_layout.addLayout(export_controls)
        self.output_name = QtWidgets.QLineEdit("annotated")
        export_controls.addWidget(self.output_name)
        self.button(
            export_controls, "Export full annotation", lambda: self.export(True)
        )
        self.button(export_controls, "Export selected 3DGS", lambda: self.export(False))
        self.button(export_controls, "Open SuperSplat", self.open_supersplat)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(155)
        layout.addWidget(self.log)
        self.steps.currentRowChanged.connect(self.change_group)
        self.stage.currentIndexChanged.connect(self.show_params)
        self.steps.setCurrentRow(0)
        self.setStyleSheet(
            "QPushButton{padding:5px 8px} QGroupBox{font-weight:600} QListWidget::item{padding:15px 8px} QTreeWidget{font-size:14px}"
        )
        if project:
            QtCore.QTimer.singleShot(
                0, lambda: self.request({"action": "status"}, self.update_status)
            )

    def button(self, layout, text, callback):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def error(self, message):
        self.log.appendPlainText("Error: " + str(message))
        QtWidgets.QMessageBox.warning(self, "Operation failed", str(message))

    def choose_python(self):
        if self.wsl_distro:
            value, ok = QtWidgets.QInputDialog.getText(
                self,
                "WSL backend Python",
                "Existing Linux interpreter path:",
                text=self.backend.text(),
            )
            if ok:
                self.backend.setText(value)
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select backend Python",
            self.backend.text(),
            "Python (python*);;All files (*)",
        )
        if path:
            self.backend.setText(path)

    def create_project(self):
        source, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select enhanced 3DGS PLY", "", "PLY (*.ply)"
        )
        if not source:
            return
        parent = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose a parent folder for the project", ""
        )
        if not parent:
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New project", "New project folder name:", text=Path(source).stem
        )
        if not ok:
            return
        if not name or Path(name).name != name:
            self.error("Enter a single folder name")
            return
        target = Path(parent) / name
        self.request(
            {"action": "create", "source": source, "project": str(target)},
            self.update_status,
        )

    def open_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open orchard project", "", "Project (project.json)"
        )
        if path:
            self.request(
                {"action": "status", "project": str(Path(path).parent)},
                self.update_status,
            )

    def change_group(self, index):
        self.stage.blockSignals(True)
        self.stage.clear()
        for stage in GROUPS[index][1]:
            self.stage.addItem(TITLES[stage], stage)
        self.stage.blockSignals(False)
        self.show_params()

    def show_params(self):
        if not self.replacing and self.parameters.topLevelItemCount():
            try:
                self.config = self.collect()
            except ValueError as error:
                self.error(
                    str(error)
                    + "; edits have been kept. Correct the value before switching."
                )
                return
        self.parameters.clear()
        stage = self.stage.currentData()
        self.ground_strategy.setVisible(stage == "ground")
        self.facility_round_controls.setVisible(stage in ("poles", "wires"))
        blocker = QtCore.QSignalBlocker(self.ground_method)
        self.ground_method.setCurrentIndex(
            self.ground_method.findData(self.config["ground"]["method"])
        )
        del blocker
        self.actions.setVisible(self.steps.currentRow() >= 2)
        if not stage:
            return
        for key, specification in SCHEMA.items():
            if specification["stage"] != stage:
                continue
            if key == "ground.method":
                continue
            if specification.get("ground_method") not in (
                None,
                self.config["ground"]["method"],
            ):
                continue
            parameter_name = key.split(".")[1]
            item = QtWidgets.QTreeWidgetItem(
                [
                    specification["label"],
                    json.dumps(self.config[stage][parameter_name], ensure_ascii=False),
                ]
            )
            item.setData(0, QtCore.Qt.UserRole, key)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
            item.setToolTip(
                1,
                "Choices: "
                + str(specification["choices"])
                + " Range: "
                + str(specification["minimum"])
                + " … "
                + str(specification["maximum"])
                + "\nQuote strings; use true/false for booleans and JSON for arrays.",
            )
            self.parameters.addTopLevelItem(item)
        self.render_status()

    def change_ground_method(self):
        method = self.ground_method.currentData()
        try:
            self.config = self.collect()
        except ValueError as error:
            blocker = QtCore.QSignalBlocker(self.ground_method)
            self.ground_method.setCurrentIndex(
                self.ground_method.findData(self.config["ground"]["method"])
            )
            del blocker
            self.error(error)
            return
        self.config["ground"]["method"] = method
        self.replacing = True
        self.show_params()
        self.replacing = False

    def collect(self):
        config = copy.deepcopy(self.config)
        for row_index in range(self.parameters.topLevelItemCount()):
            item = self.parameters.topLevelItem(row_index)
            group_name, parameter_name = item.data(0, QtCore.Qt.UserRole).split(".")
            try:
                config[group_name][parameter_name] = json.loads(item.text(1))
            except ValueError:
                raise ValueError(item.text(0) + ": Enter a valid JSON value")
        return validate(config)

    def save_params(self, after=None):
        try:
            config = self.collect()
        except ValueError as error:
            self.error(error)
            return

        def saved(status):
            self.update_status(status)
            if callable(after):
                after()

        save = lambda: self.request({"action": "configure", "config": config}, saved)
        stage = self.stage.currentData()
        if stage in ("poles", "wires") and config[stage] != self.status.get(
            "config", {}
        ).get(stage):
            self.ensure_facility_round(save)
        else:
            save()

    def begin_facility_round(self, mode, after=None):
        stage = self.stage.currentData()
        if stage not in ("poles", "wires"):
            self.error("Select Poles or Wires first")
            return
        try:
            edited = self.collect()
        except ValueError as error:
            self.error(error)
            return

        def begun(status):
            self.update_status(status)
            self.config = edited
            self.replacing = True
            self.show_params()
            self.replacing = False
            if callable(after):
                after()

        self.request(
            {"action": "begin-facility-round", "stage": stage, "mode": mode}, begun
        )

    def ensure_facility_round(self, after):
        stage = self.stage.currentData()
        if stage not in ("poles", "wires"):
            after()
            return
        current = (
            self.status.get("facility_rounds", {}).get(stage, {}).get("current") or {}
        )
        if current.get("status") == "draft":
            after()
            return
        completed = stage in self.status.get("completed", [])
        if not current and not completed and self.status.get("picks", {}).get(stage):
            after()
            return
        try:
            enabled = self.collect()[stage]["enabled"]
        except ValueError as error:
            self.error(error)
            return
        self.begin_facility_round(
            "append" if completed and enabled else "replace", after
        )

    def cancel_facility_round(self):
        stage = self.stage.currentData()
        if stage in ("poles", "wires"):
            self.request(
                {"action": "cancel-facility-round", "stage": stage}, self.update_status
            )

    def reset_params(self):
        stage = self.stage.currentData()
        if stage:
            self.config[stage] = defaults()[stage]
            self.replacing = True
            self.show_params()
            self.replacing = False

    def import_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import parameters", "", "JSON (*.json)"
        )
        if path:
            try:
                self.config = validate(
                    json.loads(Path(path).read_text(encoding="utf-8-sig"))
                )
                self.replacing = True
                self.show_params()
                self.replacing = False
                self.log.appendPlainText("Preset loaded. Save parameters to apply it.")
            except Exception as error:
                self.error(error)

    def export_config(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export parameters", "preset.json", "JSON (*.json)"
        )
        if path:
            try:
                Path(path).write_text(
                    json.dumps(self.collect(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as error:
                self.error(error)

    def update_status(self, status):
        self.status = status
        self.project = status["project"]
        self.config = status["config"]
        self.revision = status["revision"]
        self.path_label.setText(self.project + "\nSource: " + status["source"]["path"])
        self.replacing = True
        self.show_params()
        self.replacing = False
        for stage in ("poles", "wires"):
            for obj in status.get("diagnostics", {}).get(stage, {}).get("objects", []):
                if obj.get("status") in ("warning", "error"):
                    self.log.appendPlainText(
                        f"[{obj['status'].upper()}] {TITLES[stage]} pair {obj['pair_number']}: {obj['reason']}"
                    )

    def render_status(self):
        stage = self.stage.currentData()
        done = self.status.get("completed", [])
        diagnostic = self.status.get("diagnostics", {}).get(stage, {})
        details = json.dumps(diagnostic, ensure_ascii=False)
        if "failed_objects" in diagnostic:
            failed = [
                str(object_diagnostic["pair_number"])
                for object_diagnostic in diagnostic["objects"]
                if object_diagnostic["status"] != "ok"
            ]
            details = f"Succeeded: {diagnostic['succeeded_objects']} / {diagnostic['total_objects']} pairs; {diagnostic['failed_objects']} failed"
            if failed:
                details += (
                    "\nCheck pairs: " + ", ".join(failed) + "; see the log for details."
                )
        rounds = self.status.get("facility_rounds", {}).get(stage, {})
        current = rounds.get("current") or {}
        draft = current.get("status") == "draft"
        self.append_round_button.setEnabled(
            bool(rounds.get("can_append")) and not draft
        )
        self.cancel_round_button.setEnabled(draft)
        title = (
            "Status: complete (partial success)"
            if diagnostic.get("status") == "partial" and stage in done
            else (
                "Status: complete"
                if stage in done
                else "Status: pending / parameters changed"
            )
        )
        if current:
            mode = "Append" if current["mode"] == "append" else "Replace"
            title = f"Round {current['number']} / {mode} · " + (
                "pending"
                if draft
                else (
                    "complete"
                    if current["status"] == "complete"
                    else "Upstream changed; replace result"
                )
            )
            if draft:
                details = (
                    "Saved this round: "
                    + str(len(self.status.get("picks", {}).get(stage, [])) // 2)
                    + " pairs; settings apply to this round.\nPrevious round: "
                    + details
                )
            details = (
                "Total: "
                + str(rounds.get("cumulative_points", 0))
                + " points\n"
                + details
            )
            if not draft:
                details += (
                    "\nAdded: " + str(diagnostic.get("points_added", 0)) + " points"
                )
        self.state_label.setText(
            title
            + "   |   Finished "
            + str(len(done))
            + "/"
            + str(len(STAGES))
            + " stages\n"
            + details
        )

    def change_color_mode(self):
        if self.viewer and getattr(self, "viewer_mode", None) in (
            "preview",
            "transfer",
            "split",
        ):
            self.viewer_command(
                {"action": "color_mode", "mode": self.color_mode.currentData()}
            )

    def run_stage(self):
        stage = self.stage.currentData()
        if stage:
            self.ensure_facility_round(
                lambda: self.save_params(
                    lambda: self.request(
                        {"action": "run", "stage": stage}, self.update_status
                    )
                )
            )

    def request(self, request, callback):
        if self.proc is not None:
            self.error("Finish or close the current task first")
            return
        request.setdefault("project", self.project)
        if self.wsl_distro:
            request["path_format"] = "windows"
        if request["action"] not in ("create", "schema") and not request["project"]:
            self.error("Create or open a project first")
            return
        if request["action"] not in ("create", "status", "schema"):
            request["expected_revision"] = self.revision
        self.counter += 1
        request_stem = Path(self.temp.name) / str(self.counter)
        request_path = request_stem.with_suffix(".request.json")
        response_path = request_stem.with_suffix(".response.json")
        request_path.write_text(
            json.dumps(request, ensure_ascii=False), encoding="utf-8"
        )
        self.start_process(
            [
                "-X",
                "utf8",
                "-m",
                "orchard_scene_decomposition.worker",
                "--request",
                str(request_path),
                "--response",
                str(response_path),
            ],
            response_path,
            lambda data: callback(data["result"]),
        )

    def start_process(self, args, result, callback, viewer=False):
        program = self.backend.text()
        if self.wsl_distro:
            program = shutil.which("wsl.exe")
            if not program:
                self.error("WSL launcher not found")
                return
            if not self.backend.text().startswith("/"):
                self.error(
                    "Enter an absolute Linux interpreter path for the WSL backend"
                )
                return
            mapped = subprocess.run(
                [
                    program,
                    "-d",
                    self.wsl_distro,
                    "--",
                    "wslpath",
                    "-u",
                    ROOT.as_posix(),
                ],
                capture_output=True,
                text=True,
            )
            if mapped.returncode or not mapped.stdout.strip():
                self.error(
                    "Cannot map the source directory to WSL: " + mapped.stderr.strip()
                )
                return
            if not viewer:
                self.worker_pid_file = result.with_suffix(".pid")
                self.worker_request = args[args.index("--request") + 1]
                args = args + ["--pid-file", str(self.worker_pid_file)]
            args = [
                (
                    argument.replace("\\", "/")
                    if len(argument) > 2
                    and argument[1] == ":"
                    and (argument[2] in "\\/")
                    else argument
                )
                for argument in args
            ]
            self.wsl_source_root = mapped.stdout.strip()
            args = [
                "-d",
                self.wsl_distro,
                "--cd",
                mapped.stdout.strip(),
                "--",
                "env",
                "-u",
                "WAYLAND_DISPLAY",
                "XDG_SESSION_TYPE=x11",
                "PYTHONDONTWRITEBYTECODE=1",
                self.backend.text(),
                "-B",
            ] + args
        elif not Path(program).is_file():
            self.error("Backend Python not found. Select an existing interpreter")
            return
        self.viewer = viewer
        process = QtCore.QProcess(self)
        self.proc = process
        process.setWorkingDirectory(str(ROOT))
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(
            lambda: self.log.appendPlainText(
                bytes(process.readAllStandardOutput())
                .decode("utf-8", errors="replace")
                .rstrip()
            )
        )

        def failed(err):
            self.log.appendPlainText("Process error: " + process.errorString())
            if err == QtCore.QProcess.FailedToStart:
                self.proc = None
                self.viewer = False
                process.deleteLater()

        process.errorOccurred.connect(failed)

        def finished(code, _):
            self.proc = None
            self.viewer = False
            if result.exists():
                try:
                    response_payload = json.loads(result.read_text(encoding="utf-8"))
                    if response_payload.get("ok"):
                        callback(response_payload)
                    else:
                        self.error(response_payload.get("error", "Unknown error"))
                except Exception as error:
                    self.error(error)
            elif code:
                self.log.appendPlainText(
                    "Task interrupted. Committed revisions remain available."
                )
            process.deleteLater()

        process.finished.connect(finished)
        process.start(program, args)

    def pick(self):
        stage = self.stage.currentData()
        if stage not in ("poles", "wires", "roi"):
            self.error("Select Poles, Wires or ROI to pick points")
            return
        self.ensure_facility_round(
            lambda: self.save_params(lambda: self.start_viewer("picks", stage))
        )

    def edit_picks(self):
        self.ensure_facility_round(self.edit_draft_picks)

    def edit_draft_picks(self):
        stage = self.stage.currentData()
        if stage not in ("poles", "wires", "roi"):
            self.error("Select Poles, Wires or ROI to edit picks")
            return
        old = self.status.get("picks", {}).get(stage, [])
        text, ok = QtWidgets.QInputDialog.getMultiLineText(
            self,
            "Saved original point IDs",
            "Remove or reorder pairs; [] clears all picks.",
            json.dumps(old),
        )
        if ok:
            try:
                ids = json.loads(text)
            except ValueError:
                self.error("Enter a JSON array of point IDs")
                return
            if not isinstance(ids, list):
                self.error("Enter a JSON array of point IDs")
                return
            self.request(
                {"action": "picks", "stage": stage, "ids": ids}, self.update_status
            )

    def start_viewer(self, mode, stage=None):
        if self.proc or not self.project:
            self.error("Open a project and finish the current task first")
            return
        if (
            mode != "picks"
            and self.stage.currentData()
            and (self.stage.currentData() not in self.status.get("completed", []))
        ):
            self.log.appendPlainText(
                "This stage is pending or stale. Preview shows the previous valid result."
            )
        self.counter += 1
        result = Path(self.temp.name) / (str(self.counter) + ".viewer.json")
        self.viewer_mode = mode
        args = [
            "-X",
            "utf8",
            "-m",
            "orchard_scene_decomposition.ui.viewer",
            "--project",
            self.project,
            "--mode",
            mode,
            "--result",
            str(result),
            "--stage",
            stage or "poles",
            "--color-mode",
            self.color_mode.currentData(),
        ]
        if self.filter.text().strip():
            args.extend(["--labels", self.filter.text().strip()])

        def done(data):
            if mode == "picks":
                ids = (
                    self.status.get("picks", {}).get(stage, []) + data["ids"]
                    if stage != "roi"
                    else data["ids"]
                )
                self.request(
                    {"action": "picks", "stage": stage, "ids": ids}, self.update_status
                )
            else:
                self.request({"action": "status"}, self.update_status)

        self.start_process(args, result, done, viewer=True)

    def viewer_command(self, command):
        if not self.proc or not self.viewer:
            self.error("Open the fragment review window first")
            return
        self.proc.write((json.dumps(command) + "\n").encode("utf-8"))

    def edit(self, action):
        if self.viewer:
            self.viewer_command({"action": action})
        else:
            self.request({"action": action}, self.update_status)

    def export(self, annotated):
        try:
            labels = (
                None
                if annotated
                else [int(x) for x in self.filter.text().split(",") if x.strip()]
            )
        except ValueError:
            self.error("Labels must be comma-separated integers")
            return
        if labels == []:
            self.error("Enter labels to export, e.g. 4,5")
            return
        self.request(
            {
                "action": "export",
                "name": self.output_name.text(),
                "labels": labels,
                "annotated": annotated,
            },
            lambda result: self.log.appendPlainText("Exported: " + result["path"]),
        )

    def cancel(self):
        if not self.proc:
            return
        if self.viewer:
            self.viewer_command({"action": "close"})
            self.log.appendPlainText(
                "Press Q to close the picking window. Closing the review window."
            )
        else:
            if self.wsl_distro:
                if not self.worker_pid_file.exists():
                    self.log.appendPlainText(
                        "Worker is still starting; try Cancel task again in a moment."
                    )
                    return
                stopped = subprocess.run(
                    [
                        shutil.which("wsl.exe"),
                        "-d",
                        self.wsl_distro,
                        "--cd",
                        self.wsl_source_root,
                        "--",
                        "env",
                        "PYTHONDONTWRITEBYTECODE=1",
                        self.backend.text(),
                        "-B",
                        "-m",
                        "orchard_scene_decomposition.utils.cancel",
                        "--pid-file",
                        self.worker_pid_file.as_posix(),
                        "--request",
                        self.worker_request.replace("\\", "/"),
                    ],
                    capture_output=True,
                    text=True,
                )
                if stopped.returncode:
                    self.log.appendPlainText(
                        "Could not cancel the WSL worker: " + stopped.stderr.strip()
                    )
                    return
            else:
                self.proc.kill()
            self.log.appendPlainText(
                "Worker stop requested. Use Project > Recover lock if a lock remains."
            )

    def open_supersplat(self):
        if not self.wsl_distro:
            self.request(
                {"action": "supersplat-preview"},
                lambda result: QtGui.QDesktopServices.openUrl(
                    QtCore.QUrl(result["url"])
                ),
            )
            return

        def open_snapshot(result):
            try:
                from ..utils.preview_server import launch

                bridge = launch(result["path"])
                QtGui.QDesktopServices.openUrl(QtCore.QUrl(bridge["url"]))
            except Exception as exc:
                self.error(str(exc))

        self.request({"action": "preview-snapshot"}, open_snapshot)

    def closeEvent(self, event):
        if self.proc:
            self.error("Cancel computation or close the 3D view before exiting.")
            event.ignore()
        else:
            event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project")
    parser.add_argument("--backend")
    parser.add_argument("--wsl-distro")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv[:1])
    window = Window(args.project, args.backend, args.wsl_distro)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
