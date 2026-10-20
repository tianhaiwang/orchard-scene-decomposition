"""Project state on disk: source reference, parameters, picks and revisions."""

from pathlib import Path
import copy
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
import numpy as np
from plyfile import PlyData, PlyElement
from .config import STAGES, defaults, validate
from .utils.diagnostics import copy_diagnostics
from .utils.paths import source_path


def digest(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def json_write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def atomic_json(path, value):
    temporary_path = Path(str(path) + "." + uuid.uuid4().hex + ".tmp")
    json_write(temporary_path, value)
    os.replace(temporary_path, path)


class Project:
    """A project directory that references one input PLY.

    ``project.json`` holds the working head. Every change is saved as a new
    folder in ``revisions/`` containing ``state.json`` and ``arrays.npz``.
    """

    def __init__(self, root):
        self.root = source_path(root).resolve()
        self.head = json.loads((self.root / "project.json").read_text(encoding="utf-8"))
        if self.head.get("schema_version") != 1:
            raise ValueError("Unsupported project version")
        validate(self.head["config"])
        self.source = source_path(self.head["source"]["path"])
        if digest(self.source) != self.head["source"]["sha256"]:
            raise ValueError("Source file has changed. Create a new project")
        self.ply = PlyData.read(str(self.source), mmap="r")
        self.vertices = self.ply["vertex"].data
        self.point_count = len(self.vertices)
        self.arrays = {}
        self._read_arrays(self.head.get("revision"))

    @classmethod
    def create(cls, root, source):
        saved_source = str(source)
        root = source_path(root).resolve()
        source = source_path(source).resolve(strict=True)
        if root == source or source.is_relative_to(root):
            raise ValueError("Source file must be outside the project directory")
        if root.exists() and any(root.iterdir()):
            raise ValueError("Choose an empty project directory")
        ply = PlyData.read(str(source), mmap="r")
        vertices = ply["vertex"].data
        required = ["x", "y", "z", "opacity", "intensity", "f_dc_0", "f_dc_1", "f_dc_2"]
        missing = [
            field_name
            for field_name in required
            if field_name not in vertices.dtype.names
        ]
        if missing:
            raise ValueError(
                "Enhanced 3DGS PLY is missing fields: " + ", ".join(missing)
            )
        if len(vertices) < 3:
            raise ValueError("Input requires at least three points")
        for field_name in required:
            if not np.isfinite(vertices[field_name]).all():
                raise ValueError("Non-finite input field: " + field_name)
        root.mkdir(parents=True, exist_ok=True)
        (root / "revisions").mkdir()
        (root / "exports").mkdir()
        from pathlib import PureWindowsPath

        source_reference = (
            saved_source if PureWindowsPath(saved_source).is_absolute() else str(source)
        )
        head = dict(
            schema_version=1,
            source=dict(
                path=source_reference, sha256=digest(source), point_count=len(vertices)
            ),
            config=defaults(),
            picks={"poles": [], "wires": [], "roi": []},
            completed={},
            revision=None,
            history=[],
            cursor=0,
            diagnostics={},
            events=[],
            created=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(root / "project.json", head)
        return cls(root)

    def _read_arrays(self, revision):
        if revision:
            with np.load(
                self.root / "revisions" / revision / "arrays.npz", allow_pickle=False
            ) as archive:
                self.arrays = {
                    array_name: archive[array_name] for array_name in archive.files
                }
        else:
            self.arrays = {}

    def revision_state(self, revision):
        if not isinstance(revision, str) or not re.fullmatch(
            r"\d{8}T\d{6}-[0-9a-f]{10}", revision
        ):
            raise ValueError("Invalid revision identifier")
        state = json.loads(
            (self.root / "revisions" / revision / "state.json").read_text(
                encoding="utf-8"
            )
        )
        if state.get("schema_version") != 1 or state.get("revision") != revision:
            raise ValueError("Invalid saved revision")
        if (
            state["source"]["sha256"] != self.head["source"]["sha256"]
            or state["source"]["point_count"] != self.point_count
        ):
            raise ValueError("Saved revision belongs to a different source")
        validate(state["config"])
        state["source"] = copy.deepcopy(self.head["source"])
        return state

    def restore_revision(self, revision):
        if revision in self.head.get("deleted_results", []):
            raise ValueError("This result was deleted. Select a retained version")
        state = self.revision_state(revision)
        previous = self.head
        self._read_arrays(revision)
        state["revision"] = previous["revision"]
        state["events"] = copy.deepcopy(previous["events"])
        state["restored_from"] = revision
        state["result_revision"] = revision
        state["deleted_results"] = copy.deepcopy(previous.get("deleted_results", []))
        state.pop("working_selection_required", None)
        self.head = state
        return self.save("restored revision " + revision)

    def delete_results(self, revisions):
        from .utils.history import result_stage

        if (
            not isinstance(revisions, list)
            or not revisions
            or any(not isinstance(saved_revision, str) for saved_revision in revisions)
        ):
            raise ValueError("Select one or more saved results to delete")
        targets = {}
        deleted = set(self.head.get("deleted_results", []))
        for revision in dict.fromkeys(revisions):
            if revision in deleted:
                raise ValueError("This result was already deleted")
            stage, kind = result_stage(self.revision_state(revision), revision)
            if stage is None:
                raise ValueError("Only saved stage results can be deleted")
            targets[revision] = stage
        completed = self.head["completed"]
        last = next((completed[s] for s in reversed(STAGES) if s in completed), None)
        working = self.head.get("result_revision") or last
        for revision, stage in targets.items():
            if completed.get(stage) == revision:
                completed.pop(stage)
                current = (
                    self.head.get("facility_rounds", {}).get(stage, {}).get("current")
                )
                if current and current.get("revision") == revision:
                    current["status"] = "deleted"
        if working in targets:
            self.head.pop("result_revision", None)
            self.head["working_selection_required"] = True
        self.head["deleted_results"] = sorted(deleted | targets.keys())
        return self.save("deleted results " + ", ".join(targets))

    def save(self, event):
        previous = self.head["revision"]
        current = json.loads((self.root / "project.json").read_text(encoding="utf-8"))
        if current["revision"] != previous:
            raise RuntimeError("Another process modified this project. Reopen it")
        revision = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:10]
        )
        folder = self.root / "revisions" / revision
        folder.mkdir()
        self.head["revision"] = revision
        if event.startswith(
            ("completed ", "transfer fragment ", "split fragment ", "undo:", "redo:")
        ):
            self.head["result_revision"] = revision
        self.head["events"].append(
            dict(
                time=datetime.now(timezone.utc).isoformat(),
                event=event,
                parent=previous,
            )
        )
        for stage_name, value in list(self.head["completed"].items()):
            if value == "pending":
                self.head["completed"][stage_name] = revision
        for state in self.head.get("facility_rounds", {}).values():
            for record in state.get("rounds", []) + (
                [state["current"]] if state.get("current") else []
            ):
                if record.get("revision") == "pending":
                    record["revision"] = revision
        np.savez_compressed(folder / "arrays.npz", **self.arrays)
        json_write(folder / "state.json", self.head)
        atomic_json(self.root / "project.json", self.head)
        return revision

    def invalidate(self, stage, preserve_facility=None):
        for stage_name in STAGES[STAGES.index(stage) :]:
            if stage_name != preserve_facility:
                self.head["completed"].pop(stage_name, None)
            current = (
                self.head.get("facility_rounds", {}).get(stage_name, {}).get("current")
            )
            if current and stage_name != preserve_facility:
                current["status"] = "invalidated"

    def _invalidate_after(self, stage):
        index = STAGES.index(stage) + 1
        if index < len(STAGES):
            self.invalidate(STAGES[index])

    def _facility_current(self, stage):
        return self.head.get("facility_rounds", {}).get(stage, {}).get("current")

    def _check_preceding(self, stage):
        preceding = STAGES[: STAGES.index(stage)]
        missing = [
            stage_name
            for stage_name in preceding
            if stage_name not in self.head["completed"]
        ]
        if missing:
            raise ValueError("Complete these stages first: " + ", ".join(missing))
        drafts = [
            stage_name
            for stage_name in preceding
            if (self._facility_current(stage_name) or {}).get("status") == "draft"
        ]
        if drafts:
            raise ValueError(
                "Finish or cancel preceding facility rounds: " + ", ".join(drafts)
            )

    def _check_facility_base(self, stage, current):
        self._check_preceding(stage)
        predecessor = STAGES[STAGES.index(stage) - 1]
        if self.head["completed"].get(predecessor) != current["upstream_revision"]:
            raise ValueError("Upstream results changed. Start a replacement round")
        if (
            current["mode"] == "append"
            and self.head["completed"].get(stage) != current["base_revision"]
        ):
            raise ValueError(
                "Accumulated facility checkpoint is stale. Start a replacement round"
            )

    def begin_facility_round(self, stage, mode):
        if stage not in ("poles", "wires") or mode not in ("append", "replace"):
            raise ValueError("Facility rounds require poles/wires and append/replace")
        self._check_preceding(stage)
        predecessor = STAGES[STAGES.index(stage) - 1]
        upstream_revision = self.head["completed"][predecessor]
        accepted_revision = self.head["completed"].get(stage)
        if mode == "append" and not accepted_revision:
            raise ValueError("No valid accumulated result. Start a replacement round")
        state = self.head.setdefault("facility_rounds", {}).setdefault(
            stage, dict(rounds=[], current=None)
        )
        if accepted_revision and (
            not any(
                round_record.get("status") == "complete"
                and round_record.get("revision") == accepted_revision
                for round_record in state["rounds"]
            )
        ):
            saved = json.loads(
                (self.root / "revisions" / accepted_revision / "state.json").read_text(
                    encoding="utf-8"
                )
            )
            self._read_arrays(accepted_revision)
            imported_number = 1 + max(
                [round_record["number"] for round_record in state["rounds"]] + [0]
            )
            state["rounds"].append(
                dict(
                    number=imported_number,
                    mode="replace",
                    status="complete",
                    revision=accepted_revision,
                    upstream_revision=saved["completed"].get(predecessor),
                    base_revision=saved["completed"].get(predecessor),
                    config=copy.deepcopy(saved["config"][stage]),
                    picks=list(saved["picks"][stage]),
                    diagnostics=copy.deepcopy(saved["diagnostics"].get(stage, {})),
                    points_added=len(self.arrays.get(stage, [])),
                    cumulative_points=len(self.arrays.get(stage, [])),
                    imported_from_revision=True,
                )
            )
        if (state.get("current") or {}).get("status") == "draft":
            abandoned = copy.deepcopy(state["current"])
            abandoned.update(
                status="abandoned",
                config=copy.deepcopy(self.head["config"][stage]),
                picks=list(self.head["picks"][stage]),
            )
            state["rounds"].append(abandoned)
        number = 1 + max(
            [round_record["number"] for round_record in state["rounds"]] + [0]
        )
        base_revision = accepted_revision if mode == "append" else upstream_revision
        self._read_arrays(accepted_revision or base_revision)
        state["current"] = dict(
            number=number,
            mode=mode,
            status="draft",
            base_revision=base_revision,
            upstream_revision=upstream_revision,
            revision=None,
            config=copy.deepcopy(self.head["config"][stage]),
            picks=[],
        )
        self.head["picks"][stage] = []
        self._invalidate_after(stage)
        self.head["history"] = []
        self.head["cursor"] = 0
        self.save("begin " + stage + " round " + str(number) + " " + mode)

    def cancel_facility_round(self, stage):
        if stage not in ("poles", "wires"):
            raise ValueError("Only pole or wire rounds can be cancelled")
        current = self._facility_current(stage)
        if not current or current["status"] != "draft":
            raise ValueError("No pending facility round to cancel")
        self._check_facility_base(stage, current)
        accepted = self.head["completed"].get(stage)
        revision = accepted or current["upstream_revision"]
        saved = json.loads(
            (self.root / "revisions" / revision / "state.json").read_text(
                encoding="utf-8"
            )
        )
        state = self.head["facility_rounds"][stage]
        cancelled = copy.deepcopy(current)
        cancelled.update(
            status="cancelled",
            revision="pending",
            config=copy.deepcopy(self.head["config"][stage]),
            picks=list(self.head["picks"][stage]),
        )
        state["rounds"].append(cancelled)
        self._read_arrays(revision)
        self.head["config"][stage] = copy.deepcopy(saved["config"][stage])
        self.head["picks"][stage] = list(saved["picks"][stage]) if accepted else []
        if accepted:
            self.head["diagnostics"][stage] = copy.deepcopy(
                saved["diagnostics"].get(stage, {})
            )
            state["current"] = copy.deepcopy(
                next(
                    (
                        record
                        for record in reversed(state["rounds"])
                        if record.get("status") == "complete"
                        and record.get("revision") == accepted
                    ),
                    None,
                )
            )
        else:
            self.head["diagnostics"].pop(stage, None)
            state["current"] = None
        self._invalidate_after(stage)
        self.head["history"] = []
        self.head["cursor"] = 0
        self.save("cancel " + stage + " round " + str(current["number"]))

    def prepare_picking(self, stage):
        if stage not in ("poles", "wires", "roi"):
            raise ValueError("This stage does not use picking")
        self._check_preceding(stage)
        current = self._facility_current(stage)
        if current and current["status"] == "draft":
            self._check_facility_base(stage, current)
            revision = current["base_revision"]
        else:
            revision = self.head["completed"][STAGES[STAGES.index(stage) - 1]]
        self._read_arrays(revision)
        return dict(
            base_revision=revision,
            active_points=len(self.arrays["active"]),
            round_number=(
                current["number"] if current and current["status"] == "draft" else None
            ),
        )

    def finish_facility_round(self, stage, report):
        current = self._facility_current(stage)
        if not current or current["status"] != "draft":
            return
        record = copy.deepcopy(current)
        record.update(
            status="complete",
            revision="pending",
            config=copy.deepcopy(self.head["config"][stage]),
            picks=list(self.head["picks"][stage]),
            diagnostics=copy.deepcopy(report),
            points_added=report["points_added"],
            cumulative_points=report["cumulative_points"],
            imported_from_revision=False,
        )
        state = self.head["facility_rounds"][stage]
        state["rounds"].append(record)
        state["current"] = copy.deepcopy(record)

    def set_config(self, config):
        validate(config)
        changed = [
            stage_name
            for stage_name in STAGES
            if config[stage_name] != self.head["config"][stage_name]
        ]
        if changed:
            stage = changed[0]
            current = self._facility_current(stage)
            if current and current["status"] == "complete":
                raise ValueError(
                    "Start an append or replacement round before changing these parameters"
                )
            if current and current["status"] == "draft":
                self._check_facility_base(stage, current)
                self._invalidate_after(stage)
                current["config"] = copy.deepcopy(config[stage])
                self._read_arrays(
                    self.head["completed"].get(stage) or current["base_revision"]
                )
            else:
                self.invalidate(stage)
            self.head["config"] = copy.deepcopy(config)
            self.save("parameters changed: " + ", ".join(changed))

    def set_picks(self, stage, ids):
        if stage not in self.head["picks"]:
            raise ValueError("This stage does not use picking")
        if any(type(i) is not int or i < 0 or i >= self.point_count for i in ids):
            raise ValueError("Picks must use original row indices")
        if stage in ("poles", "wires") and len(ids) % 2:
            raise ValueError("Facility endpoints must be paired")
        if stage == "roi" and len(ids) > 1:
            raise ValueError("ROI side selection requires exactly one point")
        if ids != self.head["picks"][stage]:
            current = self._facility_current(stage)
            if current and current["status"] == "complete":
                raise ValueError(
                    "Start an append or replacement round before picking endpoints"
                )
            if current and current["status"] == "draft":
                self._check_facility_base(stage, current)
                self._invalidate_after(stage)
                current["picks"] = list(ids)
                self._read_arrays(
                    self.head["completed"].get(stage) or current["base_revision"]
                )
            else:
                self.invalidate(stage)
            self.head["picks"][stage] = list(ids)
            self.save("picks changed: " + stage)

    def prepare(self, stage):
        i = STAGES.index(stage)
        self._check_preceding(stage)
        self._facility_run = None
        current = self._facility_current(stage)
        if current:
            if current["status"] == "complete":
                raise ValueError("Round complete. Start an append or replacement round")
            if current["status"] != "draft":
                raise ValueError("Round is stale. Start a replacement round")
            if current["mode"] == "append" and (
                not self.head["config"][stage]["enabled"]
            ):
                raise ValueError(
                    "Cannot disable extraction in an append round. Start a replacement round to clear facilities"
                )
            self._check_facility_base(stage, current)
            self._facility_run = dict(current, stage=stage)
            self._read_arrays(current["base_revision"])
            self.invalidate(stage, preserve_facility=stage)
            self.head["history"] = []
            self.head["cursor"] = 0
            return
        if stage == "refill":
            self.require_annotations()
            self.strip_refill()
        elif i:
            self._read_arrays(self.head["completed"][STAGES[i - 1]])
        else:
            self.arrays = {}
        self.invalidate(stage)
        if stage != "refill":
            self.head["history"] = []
            self.head["cursor"] = 0

    def require_annotations(self):
        if "assign_branches" not in self.head["completed"]:
            raise ValueError(
                "Run Assign branches first; rerun it after upstream changes"
            )

    def strip_refill(self):
        if "provenance" not in self.arrays:
            return
        arrays = self.arrays
        mask = arrays["provenance"] == 2
        arrays["labels"][mask] = 0
        arrays["provenance"][mask] = 0
        self.head["completed"].pop("refill", None)

    def status(self):
        from .utils.history import catalogue

        rounds = {}
        for stage in ("poles", "wires"):
            state = copy.deepcopy(
                self.head.get("facility_rounds", {}).get(
                    stage, dict(current=None, rounds=[])
                )
            )
            state["can_append"] = stage in self.head["completed"] and (
                not any(
                    (self._facility_current(stage_name) or {}).get("status") == "draft"
                    for stage_name in STAGES[: STAGES.index(stage)]
                )
            )
            successful = [
                round_record
                for round_record in state["rounds"]
                if round_record["status"] == "complete"
            ]
            state["cumulative_points"] = (
                successful[-1]["cumulative_points"]
                if successful and stage in self.head["completed"]
                else (
                    self.head["diagnostics"].get(stage, {}).get("points", 0)
                    if stage in self.head["completed"]
                    else 0
                )
            )
            rounds[stage] = state
        return dict(
            project=str(self.root),
            source=self.head["source"],
            revision=self.head["revision"],
            completed=list(self.head["completed"]),
            config=self.head["config"],
            picks=self.head["picks"],
            diagnostics=copy_diagnostics(self.head["diagnostics"]),
            history_cursor=self.head["cursor"],
            history_length=len(self.head["history"]),
            facility_rounds=rounds,
            stage_versions=catalogue(self),
            deleted_results=self.head.get("deleted_results", []),
            working_selection_required=self.head.get(
                "working_selection_required", False
            ),
        )

    def transfer(self, source_point, target_point=None, target_label=None):
        self.require_annotations()
        self.strip_refill()
        arrays = self.arrays
        if not 0 <= source_point < self.point_count:
            raise ValueError("Source point index out of range")
        fragment = int(arrays["fragment_id"][source_point])
        if fragment < 0:
            raise ValueError("Source point is not in an editable branch fragment")
        if target_point is not None:
            if not 0 <= target_point < self.point_count:
                raise ValueError("Target point index out of range")
            target_label = int(arrays["labels"][target_point])
        if (
            target_label is None
            or target_label < 4
            or target_label not in arrays["labels"]
        ):
            raise ValueError("Target must be an existing tree instance (label >= 4)")
        ids = np.flatnonzero(arrays["fragment_id"] == fragment)
        self._edit(
            ids,
            dict(labels=target_label, manual_override=1, provenance=1),
            "transfer fragment " + str(fragment) + " to tree " + str(target_label),
        )

    def split(self, ids):
        self.require_annotations()
        self.strip_refill()
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        if not len(ids) or ids.min() < 0 or ids.max() >= self.point_count:
            raise ValueError("Split selection is empty or out of range")
        fragments = np.unique(self.arrays["fragment_id"][ids])
        if len(fragments) != 1 or fragments[0] < 0:
            raise ValueError("Split selection must belong to one branch fragment")
        if len(ids) == np.count_nonzero(self.arrays["fragment_id"] == fragments[0]):
            raise ValueError(
                "Transfer the fragment directly when all its points are selected"
            )
        self._edit(
            ids,
            dict(fragment_id=int(self.arrays["fragment_id"].max()) + 1),
            "split fragment " + str(fragments[0]),
        )

    def _edit(self, ids, changes, description):
        """Apply a manual edit and keep before/after values for undo and redo."""
        history = self.head["history"][: self.head["cursor"]]
        key = "edit_" + uuid.uuid4().hex
        self.arrays[key + "_ids"] = ids
        for field_name, values in changes.items():
            self.arrays[key + "_before_" + field_name] = self.arrays[field_name][
                ids
            ].copy()
            self.arrays[field_name][ids] = values
            self.arrays[key + "_after_" + field_name] = self.arrays[field_name][
                ids
            ].copy()
        history.append(dict(key=key, fields=list(changes), description=description))
        self.head["history"] = history
        self.head["cursor"] = len(history)
        self.save(description)

    def undo(self, redo=False):
        self.require_annotations()
        self.strip_refill()
        cursor = self.head["cursor"]
        history = self.head["history"]
        if redo and cursor == len(history) or (not redo and cursor == 0):
            raise ValueError("Nothing to undo or redo")
        record = history[cursor if redo else cursor - 1]
        key = record["key"]
        ids = self.arrays[key + "_ids"]
        for field_name in record["fields"]:
            self.arrays[field_name][ids] = self.arrays[
                key + ("_after_" if redo else "_before_") + field_name
            ]
        self.head["cursor"] = cursor + (1 if redo else -1)
        self.save(("redo: " if redo else "undo: ") + record["description"])

    def export(self, name="annotated", labels=None, annotated=True):
        if self.head.get("working_selection_required"):
            raise ValueError(
                "Select a retained version and use it as the working state first"
            )
        self.require_annotations()
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("Export name must be a filename without a path")
        path = self.root / "exports" / (name + ".ply")
        return self._write_ply(path, labels, annotated)

    def preview_snapshot(self):
        if self.head.get("working_selection_required"):
            raise ValueError(
                "Select a retained version and use it as the working state first"
            )
        from .utils.preview_cache import prune_previews

        completed = [stage for stage in STAGES if stage in self.head["completed"]]
        stage = completed[-1] if completed else "source"
        folder = self.root / "previews"
        if folder.resolve() != folder:
            raise ValueError("Preview cache must be inside the project directory")
        folder.mkdir(exist_ok=True)
        revision = self.head["revision"] or self.head["source"]["sha256"][:16]
        path = folder / (stage + "-" + revision + ".ply")
        if path.resolve() != path or path.with_suffix(
            ".json"
        ).resolve() != path.with_suffix(".json"):
            raise ValueError("Preview snapshot must be inside the project cache")
        if path.exists() and path.with_suffix(".json").exists():
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if (
                meta.get("snapshot_stage") == stage
                and meta.get("revision") == self.head["revision"]
            ):
                prune_previews(folder, path, self.head["source"]["sha256"])
                return dict(
                    path=str(path),
                    stage=stage,
                    revision=self.head["revision"],
                    rows=self.point_count,
                )
        saved = self.arrays
        try:
            if completed:
                if "assign_branches" not in completed:
                    self._read_arrays(self.head["completed"][stage])
            else:
                self.arrays = dict(
                    labels=np.zeros(self.point_count, dtype="i4"),
                    fragment_id=np.full(self.point_count, -1, dtype="i4"),
                    manual_override=np.zeros(self.point_count, dtype="u1"),
                    provenance=np.zeros(self.point_count, dtype="u1"),
                    reason=np.zeros(self.point_count, dtype="u1"),
                )
            self._write_ply(path, metadata=dict(snapshot_stage=stage, preview=True))
        finally:
            self.arrays = saved
        prune_previews(folder, path, self.head["source"]["sha256"])
        return dict(
            path=str(path),
            stage=stage,
            revision=self.head["revision"],
            rows=self.point_count,
        )

    def _write_ply(self, path, labels=None, annotated=True, metadata=None):
        arrays = self.arrays
        ids = (
            np.arange(self.point_count)
            if labels is None
            else np.flatnonzero(np.isin(arrays["labels"], labels))
        )
        if not len(ids):
            raise ValueError("No points match the selected labels")
        required_gaussian_fields = (
            ["f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
            + ["scale_" + str(component_index) for component_index in range(3)]
            + ["rot_" + str(component_index) for component_index in range(4)]
        )
        missing = [
            annotation_name
            for annotation_name in required_gaussian_fields
            if annotation_name not in self.vertices.dtype.names
        ]
        if missing:
            raise ValueError(
                "Cannot export complete 3DGS; missing: " + ", ".join(missing)
            )
        if path.exists():
            raise FileExistsError(
                "Output already exists. Choose another name: " + str(path)
            )
        annotation_fields = {}
        if annotated:
            annotation_fields = dict(
                segmentation=arrays["labels"][ids].astype("i4"),
                fragment_id=arrays["fragment_id"][ids].astype("i4"),
                original_point_id=ids.astype("i4"),
                manual_override=arrays["manual_override"][ids].astype("u1"),
                label_source=arrays["provenance"][ids].astype("u1"),
                dontcare_reason=arrays["reason"][ids].astype("u1"),
            )
        output_field_names = {}
        used_field_names = set(self.vertices.dtype.names)
        for field_name in annotation_fields:
            output_field_name = field_name
            while output_field_name in used_field_names:
                output_field_name = "osd_" + output_field_name
            output_field_names[field_name] = output_field_name
            used_field_names.add(output_field_name)
        output_dtype = self.vertices.dtype.descr + [
            (output_field_names[annotation_name], annotation_values.dtype.str)
            for annotation_name, annotation_values in annotation_fields.items()
        ]
        output_vertices = np.empty(len(ids), dtype=output_dtype)
        for field_name in self.vertices.dtype.names:
            output_vertices[field_name] = self.vertices[field_name][ids]
        for field_name, values in annotation_fields.items():
            output_vertices[output_field_names[field_name]] = values
        elements = [PlyElement.describe(output_vertices, "vertex")]
        if labels is None:
            elements.extend(
                element for element in self.ply.elements if element.name != "vertex"
            )
        temporary_path = path.with_suffix(".tmp")
        PlyData(
            elements,
            text=False,
            byte_order=self.ply.byte_order,
            comments=self.ply.comments,
            obj_info=self.ply.obj_info,
        ).write(str(temporary_path))
        os.replace(temporary_path, path)
        json_write(
            path.with_suffix(".json"),
            dict(
                source=self.head["source"],
                revision=self.head["revision"],
                rows=len(ids),
                fields=output_field_names,
                labels=labels,
                original_attributes_unchanged=True,
                **(metadata or {}),
            ),
        )
        return str(path)
