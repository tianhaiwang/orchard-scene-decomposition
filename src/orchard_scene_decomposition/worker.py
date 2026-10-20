"""Backend request dispatcher shared by the CLI and the Qt workbench."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import struct
import sys
import numpy as np
from .config import defaults, SCHEMA, STAGES, TITLES
from .project import Project, json_write
from .utils.colors import source_colors
from .utils.paths import source_path

# Preview file: 8-byte magic, uint32 point count, VIEW_DTYPE rows,
# then one uint8 RGB row per point in version 2.
VIEW_DTYPE = np.dtype(
    [
        ("xyz", "<f4", (3,)),
        ("original", "<u4"),
        ("label", "<i4"),
        ("fragment", "<i4"),
        ("reason", "<i4"),
        ("manual", "<i4"),
    ]
)


@contextmanager
def writer_lock(root):
    path = Path(root) / ".writer.lock"
    try:
        lock_descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("Project is locked. If no task is running, use Recover lock")
    try:
        os.write(lock_descriptor, str(os.getpid()).encode("ascii"))
        os.close(lock_descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def write_view(project, path, request):
    version = request.get("view_version", 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Unsupported preview protocol version")
    head_revision = project.head["revision"]
    snapshot = request.get("snapshot")
    snapshot_stage = None
    if snapshot and request.get("before_stage"):
        raise ValueError("Historical views cannot also request picking input")
    if snapshot and request.get("latest_valid"):
        raise ValueError("Choose a saved snapshot or the current valid result")
    if snapshot and snapshot != "raw":
        if snapshot in project.head.get("deleted_results", []):
            raise ValueError("This result was deleted. Select a retained version")
        from .utils.history import result_stage

        state = project.revision_state(snapshot)
        snapshot_stage, _ = result_stage(state, snapshot)
        if snapshot_stage is None:
            raise ValueError("Select a saved stage result or manual revision")
        project.head = state
        project._read_arrays(snapshot)
    elif snapshot == "raw":
        xyz = np.column_stack(
            [project.vertices[array_name] for array_name in ("x", "y", "z")]
        ).astype(np.float64)
        project.arrays = dict(
            xyz=xyz,
            center=xyz.mean(axis=0),
            active=np.arange(project.point_count, dtype=np.int64),
            labels=np.zeros(project.point_count, dtype=np.int32),
            fragment_id=np.full(project.point_count, -1, dtype=np.int32),
            reason=np.zeros(project.point_count, dtype=np.int32),
            manual_override=np.zeros(project.point_count, dtype=np.int32),
        )
        snapshot_stage = "raw"
    if request.get("before_stage"):
        stage = request["before_stage"]
        picking_info = project.prepare_picking(stage)
    else:
        picking_info = None
    display_stage = None
    if request.get("latest_valid"):
        if project.head.get("working_selection_required"):
            raise ValueError(
                "Select a retained version and use it as the working state first"
            )
        if request.get("before_stage"):
            raise ValueError("Resume preview cannot also request picking input")
        completed = [
            stage_name
            for stage_name in STAGES
            if stage_name in project.head["completed"]
        ]
        if not completed:
            raise ValueError("No valid results. Run Load & align first")
        display_stage = completed[-1]
        if "assign_branches" not in completed:
            project._read_arrays(project.head["completed"][display_stage])
    arrays = project.arrays
    if "xyz" not in arrays:
        raise ValueError("Run Load & align first")
    facility_stage = snapshot_stage or display_stage
    parts = request.get("facility_parts") and facility_stage in (
        "ground",
        "poles",
        "wires",
    )
    if parts:
        request = dict(request, active=True)
        request.pop("array", None)
    definitions = []
    if (
        request.get("stage_parts")
        and facility_stage in STAGES
        and not request.get("array")
    ):
        from .utils.stage_parts import stage_parts

        definitions = stage_parts(project, facility_stage)
        for part in definitions:
            part["total_points"] = len(part["ids"])
            if request.get("labels") is not None:
                part["ids"] = part["ids"][
                    np.isin(arrays["labels"][part["ids"]], request["labels"])
                ]
    ids = np.arange(project.point_count, dtype=np.int64)
    if definitions:
        ids = np.concatenate([part["ids"] for part in definitions])
    elif request.get("array"):
        if request["array"] not in index_arrays(arrays):
            raise ValueError("This array is not a previewable point index set")
        values = arrays.get(request["array"])
        if values is None or values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError("Preview requires a 1D point index array")
        ids = values[(values >= 0) & (values < project.point_count)].astype(np.int64)
    elif request.get("active") or request.get("before_stage"):
        ids = arrays["active"]
    if not definitions and request.get("labels") is not None:
        ids = ids[np.isin(arrays["labels"][ids], request["labels"])]
    cap = int(request.get("max_points", 2000000))
    if cap < 1:
        raise ValueError("Display point limit must be positive")
    if len(ids) > cap:
        ids = ids[np.linspace(0, len(ids) - 1, cap, dtype=np.int64)]
    result = write_ids_view(project, path, ids, version, cap)
    result.update(arrays=index_arrays(arrays), head_revision=head_revision)
    if request.get("array"):
        result["preview_array"] = request["array"]
    if snapshot:
        result.update(
            snapshot=snapshot,
            display_stage=snapshot_stage,
            historical=snapshot == "raw" or snapshot != head_revision,
            snapshot_config=project.head["config"],
        )
    if display_stage is not None:
        result["display_stage"] = display_stage
    if picking_info is not None:
        result["picking_base"] = picking_info
    if parts:
        extracted_path = str(Path(path).with_suffix(".extracted.bin"))
        extracted = write_view(
            project,
            extracted_path,
            dict(view_version=version, array=facility_stage, max_points=cap),
        )
        result["facility_parts"] = dict(
            stage=facility_stage,
            extracted=extracted,
            extracted_total=len(arrays[facility_stage]),
            remaining_total=len(arrays["active"]),
        )
        if not snapshot:
            result["snapshot"] = project.head["completed"][facility_stage]
    if definitions:
        outputs = []
        displayed = read_view(path)["original"]
        selected = np.zeros(project.point_count, dtype=bool)
        selected[displayed] = True
        offset = 0
        for index, part in enumerate(definitions):
            part_ids = part["ids"][selected[part["ids"]]]
            view = write_ids_view(
                project,
                str(Path(path).with_suffix(f".part-{index}.bin")),
                part_ids,
                version,
                cap,
            )
            view.update(
                key=part["key"],
                name=part["name"],
                total_points=part["total_points"],
                view_offset=offset,
            )
            offset += len(part_ids)
            if part["color"] is not None:
                view["part_color"] = part["color"]
            outputs.append(view)
        result["stage_parts"] = outputs
        result["display_stage"] = facility_stage
        if not snapshot:
            result["snapshot"] = (
                project.head.get("result_revision")
                or project.head["completed"][facility_stage]
            )
    return result


def write_ids_view(project, path, ids, version, cap):
    arrays = project.arrays
    if len(ids) > cap:
        ids = ids[np.linspace(0, len(ids) - 1, cap, dtype=np.int64)]
    view_rows = np.empty(len(ids), dtype=VIEW_DTYPE)
    coordinate_shift = arrays["center"]
    view_rows["xyz"] = (arrays["xyz"][ids] - coordinate_shift).astype("f4")
    view_rows["original"] = ids
    view_rows["label"] = arrays["labels"][ids]
    view_rows["fragment"] = arrays["fragment_id"][ids]
    view_rows["reason"] = arrays["reason"][ids]
    view_rows["manual"] = arrays["manual_override"][ids]
    rgb, color_info = (
        source_colors(project.vertices, ids) if version == 2 else (None, None)
    )
    with open(path, "wb") as output_file:
        output_file.write(b"OSDVIEW2" if version == 2 else b"OSDVIEW1")
        output_file.write(struct.pack("<I", len(view_rows)))
        view_rows.tofile(output_file)
        if rgb is not None:
            rgb.tofile(output_file)
    return dict(
        path=str(path),
        points=len(ids),
        shift=coordinate_shift.tolist(),
        revision=project.head["revision"],
        view_version=version,
        color_source=color_info,
    )


def index_arrays(arrays):
    common_array_names = {
        "active",
        "ground",
        "poles",
        "wires",
        "out_of_roi",
        "tree_points",
        "branch_points",
    }
    suffixes = (
        "_candidate_ids",
        "_axis_filtered_ids",
        "_cross_section_filtered_ids",
        "_selected_ids",
        "_feature_filtered_ids",
        "_fit_inlier_ids",
    )
    return sorted(
        array_name
        for array_name, array in arrays.items()
        if array_name in common_array_names
        or (
            array_name.startswith(("pole_", "wire_"))
            and array_name.endswith(suffixes)
            and (array.ndim == 1)
            and (array.dtype.kind in "iu")
        )
    )


def read_view(path):
    with open(path, "rb") as input_file:
        magic = input_file.read(8)
        if magic not in (b"OSDVIEW1", b"OSDVIEW2"):
            raise ValueError("Invalid preview file format")
        count_bytes = input_file.read(4)
        if len(count_bytes) != 4:
            raise ValueError("Incomplete preview file header")
        point_count = struct.unpack("<I", count_bytes)[0]
        expected_size = 12 + point_count * (
            VIEW_DTYPE.itemsize + (3 if magic == b"OSDVIEW2" else 0)
        )
        if os.fstat(input_file.fileno()).st_size != expected_size:
            raise ValueError("Invalid preview file size")
        view_rows = np.fromfile(input_file, dtype=VIEW_DTYPE, count=point_count)
        if magic == b"OSDVIEW2":
            rgb = np.fromfile(
                input_file, dtype=np.uint8, count=point_count * 3
            ).reshape(point_count, 3)
            colored_rows = np.empty(
                point_count, dtype=VIEW_DTYPE.descr + [("rgb", "u1", (3,))]
            )
            for name in VIEW_DTYPE.names:
                colored_rows[name] = view_rows[name]
            colored_rows["rgb"] = rgb
            view_rows = colored_rows
        if len(view_rows) != point_count or input_file.read(1):
            raise ValueError("Invalid preview file size")
        return view_rows


def dispatch(request):
    """Execute one JSON request and return a JSON-serialisable result."""
    action = request["action"]
    if action == "schema":
        return dict(schema=SCHEMA, defaults=defaults(), stages=STAGES, titles=TITLES)
    if action == "create":
        return Project.create(request["project"], request["source"]).status()
    if action == "recover-lock":
        path = source_path(request["project"]) / ".writer.lock"
        if path.exists():
            pid = int(path.read_text(encoding="ascii"))
            if os.name == "nt":
                import ctypes

                handle = ctypes.windll.kernel32.OpenProcess(4096, False, pid)
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
                    raise RuntimeError("The writer process is still running")
            else:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError("The writer process is still running")
            path.unlink()
        return {"recovered": True}
    project = Project(request["project"])
    if action == "status":
        return project.status()
    if action == "view":
        if (
            request.get("expected_revision")
            and request["expected_revision"] != project.head["revision"]
        ):
            raise RuntimeError(
                "Project changed. Reopen it before viewing a saved result"
            )
        return write_view(project, request["output"], request)
    if action == "inspect":
        point_index = int(request["point"])
        if not 0 <= point_index < project.point_count:
            raise ValueError("Point index out of range")
        return {
            array_name: int(project.arrays[array_name][point_index])
            for array_name in [
                "labels",
                "fragment_id",
                "reason",
                "manual_override",
                "provenance",
            ]
        }
    if action in ("supersplat-preview", "preview-snapshot"):
        with writer_lock(project.root):
            project = Project(request["project"])
            if (
                request.get("expected_revision")
                and request["expected_revision"] != project.head["revision"]
            ):
                raise RuntimeError("Project changed. Refresh before opening SuperSplat")
            result = project.preview_snapshot()
            if action == "preview-snapshot":
                return result
            from .utils.preview_server import launch

            result.update(launch(result["path"]))
            return result
    if action == "export":
        with writer_lock(project.root):
            project = Project(request["project"])
            if (
                request.get("expected_revision")
                and request["expected_revision"] != project.head["revision"]
            ):
                raise RuntimeError("Export request is stale. Refresh the project")
            return {
                "path": project.export(
                    request.get("name", "annotated"),
                    request.get("labels"),
                    request.get("annotated", True),
                )
            }
    with writer_lock(project.root):
        project = Project(request["project"])
        if (
            request.get("expected_revision")
            and request["expected_revision"] != project.head["revision"]
        ):
            raise RuntimeError("View or parameters are stale. Refresh and try again")
        if project.head.get("working_selection_required") and action not in (
            "restore-revision",
            "delete-results",
        ):
            raise ValueError(
                "Select a retained version and use it as the working state first"
            )
        if action == "configure":
            project.set_config(request["config"])
        elif action == "begin-facility-round":
            project.begin_facility_round(
                request["stage"], request.get("mode", "append")
            )
        elif action == "cancel-facility-round":
            project.cancel_facility_round(request["stage"])
        elif action == "picks":
            project.set_picks(request["stage"], request["ids"])
        elif action == "restore-revision":
            project.restore_revision(request["revision"])
        elif action == "delete-results":
            project.delete_results(request["revisions"])
        elif action == "run":
            from .algorithms.pipeline import run_stage

            run_stage(project, request["stage"])
        elif action == "transfer":
            project.transfer(
                int(request["source_point"]),
                request.get("target_point"),
                request.get("target_label"),
            )
        elif action == "split":
            project.split(request["ids"])
        elif action in ("undo", "redo"):
            project.undo(redo=action == "redo")
        else:
            raise ValueError("Unknown action: " + action)
    return project.status()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--pid-file")
    args = parser.parse_args()
    response_path = source_path(args.response)
    if args.pid_file:
        source_path(args.pid_file).write_text(str(os.getpid()), encoding="ascii")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    try:
        request = json.loads(source_path(args.request).read_text(encoding="utf-8-sig"))
        if request.get("output"):
            request["output"] = str(source_path(request["output"]))
        result = dispatch(request)
        response = dict(ok=True, result=result)
        if request.get("path_format") == "windows" and os.name == "posix":
            import subprocess

            for key in ("project", "path"):
                if result.get(key) and str(result[key]).startswith("/"):
                    result[key] = subprocess.run(
                        ["wslpath", "-w", str(result[key])],
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
    except Exception as exc:
        response = dict(ok=False, error=str(exc), type=type(exc).__name__)
        if hasattr(exc, "diagnostics"):
            response["diagnostics"] = exc.diagnostics
    json_write(response_path, response)
    return 0 if response["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
