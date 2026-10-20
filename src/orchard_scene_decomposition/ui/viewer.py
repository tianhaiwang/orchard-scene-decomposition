"""Open3D windows for picking endpoints and reviewing branch assignments."""

import argparse
import json
import queue
import sys
import threading
import numpy as np
import open3d as o3d
from ..project import Project, json_write
from ..worker import writer_lock
from ..utils.colors import source_colors


def colors(labels):
    labels = np.asarray(labels)
    palette = np.array(
        [[0.5, 0.5, 0.5], [0.45, 0.32, 0.18], [0.2, 0.35, 0.9], [0.9, 0.2, 0.2]]
    )
    point_colors = np.empty((len(labels), 3))
    for palette_index, label in enumerate(np.unique(labels)):
        if 0 <= label < 4:
            color = palette[label]
        else:
            import colorsys

            color = colorsys.hsv_to_rgb(int(label) * 0.61803398875 % 1, 0.65, 0.95)
        point_colors[labels == label] = color
    return point_colors


def interact(root, mode, stage, result_path, filter_labels=None, color_mode="rgb"):
    if color_mode not in ("rgb", "labels"):
        raise ValueError("Color mode must be rgb or labels")
    project = Project(root)
    arrays = project.arrays
    if "xyz" not in arrays:
        raise ValueError("Run Load & align first")
    if mode in ("transfer", "split"):
        project.require_annotations()
    ids = np.arange(project.point_count, dtype=np.int64)
    if mode == "picks":
        project.prepare_picking(stage)
        arrays = project.arrays
        ids = arrays["active"].copy()
    if filter_labels:
        ids = ids[np.isin(arrays["labels"][ids], filter_labels)]
    if not len(ids):
        raise ValueError("No points in the current view")
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(
        arrays["xyz"][ids] - arrays["center"]
    )
    rgb, _ = source_colors(project.vertices, ids)
    rgb = rgb.astype(np.float64) / 255.0
    point_cloud.colors = o3d.utility.Vector3dVector(
        rgb if color_mode == "rgb" else colors(arrays["labels"][ids])
    )
    if mode == "picks":
        visualizer = o3d.visualization.VisualizerWithEditing()
        visualizer.create_window(
            "Orchard Scene Decomposition | SHIFT + click pairs | Q to accept",
            width=1100,
            height=800,
            visible=True,
        )
        visualizer.add_geometry(point_cloud)
        visualizer.run()
        selected = [int(ids[i]) for i in visualizer.get_picked_points()]
        visualizer.destroy_window()
        json_write(result_path, {"ok": True, "ids": selected})
        return
    visualizer = o3d.visualization.VisualizerWithVertexSelection()
    if not visualizer.create_window(
        "Orchard Scene Decomposition | SHIFT click fragment then target tree | Q close",
        width=1100,
        height=800,
        visible=True,
    ):
        raise RuntimeError(
            "Open3D could not create a window. Check the display environment"
        )
    visualizer.add_geometry(point_cloud)
    visualizer.get_render_option().point_size = 3
    commands = queue.Queue()
    state = dict(mode=mode, source=None, busy=False, color_mode=color_mode)

    def input_loop():
        for line in sys.stdin:
            try:
                commands.put(json.loads(line))
            except ValueError:
                print("Invalid view command", flush=True)

    threading.Thread(target=input_loop, daemon=True).start()

    def paint():
        point_colors = (
            rgb.copy()
            if state["color_mode"] == "rgb"
            else colors(arrays["labels"][ids])
        )
        if state["source"] is not None:
            point_colors[
                arrays["fragment_id"][ids] == arrays["fragment_id"][state["source"]]
            ] = [1.0, 1.0, 0.0]
        point_cloud.points = o3d.utility.Vector3dVector(
            arrays["xyz"][ids] - arrays["center"]
        )
        point_cloud.colors = o3d.utility.Vector3dVector(point_colors)
        visualizer.update_geometry(point_cloud)

    def apply(call):
        with writer_lock(project.root):
            call()
        paint()
        print("Changes saved.", flush=True)

    def changed():
        if state["busy"]:
            return
        selected = visualizer.get_picked_points()
        if state["mode"] != "transfer" or len(selected) != 1:
            return
        state["busy"] = True
        try:
            original = int(ids[selected[0].index])
            visualizer.clear_picked_points()
            if state["source"] is None:
                fragment = int(arrays["fragment_id"][original])
                if fragment < 0:
                    raise ValueError("Pick a branch fragment first")
                state["source"] = original
                paint()
                print(
                    "Highlighted fragment "
                    + str(fragment)
                    + "; now pick a target tree",
                    flush=True,
                )
            else:
                source = state["source"]
                project.require_annotations()
                apply(lambda: project.transfer(source, original))
                state["source"] = None
                paint()
        except Exception as exc:
            print("Operation failed: " + str(exc), flush=True)
        finally:
            state["busy"] = False

    def tick(_):
        try:
            while not commands.empty():
                command = commands.get_nowait()
                operation = command["action"]
                state["busy"] = True
                if operation in ("undo", "redo"):
                    state["source"] = None
                    apply(lambda: project.undo(operation == "redo"))
                elif operation in ("transfer_mode", "split_mode"):
                    state["source"] = None
                    state["mode"] = (
                        "transfer" if operation == "transfer_mode" else "split"
                    )
                    visualizer.clear_picked_points()
                    paint()
                elif operation == "split":
                    selected = [
                        int(ids[x.index]) for x in visualizer.get_picked_points()
                    ]
                    apply(lambda: project.split(selected))
                    visualizer.clear_picked_points()
                elif operation == "target":
                    if state["source"] is None:
                        raise ValueError("Pick a source fragment first")
                    apply(
                        lambda: project.transfer(
                            state["source"], target_label=int(command["label"])
                        )
                    )
                    state["source"] = None
                    paint()
                elif operation == "clear":
                    state["source"] = None
                    visualizer.clear_picked_points()
                    paint()
                elif operation == "color_mode":
                    if command.get("mode") not in ("rgb", "labels"):
                        raise ValueError("Color mode must be rgb or labels")
                    state["color_mode"] = command["mode"]
                    paint()
                elif operation == "close":
                    visualizer.close()
        except Exception as exc:
            print("Operation failed: " + str(exc), flush=True)
        finally:
            state["busy"] = False
        return False

    visualizer.register_selection_changed_callback(changed)
    visualizer.register_selection_moved_callback(paint)
    visualizer.register_animation_callback(tick)
    visualizer.run()
    visualizer.destroy_window()
    json_write(result_path, {"ok": True, "revision": project.head["revision"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--mode", default="preview", choices=["picks", "preview", "transfer", "split"]
    )
    parser.add_argument("--stage", default="poles")
    parser.add_argument("--result", required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--color-mode", choices=["rgb", "labels"], default="rgb")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    from ..utils.paths import source_path

    args.result = str(source_path(args.result))
    try:
        interact(
            args.project,
            args.mode,
            args.stage,
            args.result,
            [int(x) for x in args.labels.split(",") if x.strip()],
            args.color_mode,
        )
    except Exception as exc:
        json_write(args.result, dict(ok=False, error=str(exc)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
