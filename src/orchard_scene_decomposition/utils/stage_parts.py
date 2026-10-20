import colorsys
import numpy as np


def stage_parts(project, stage):
    """Split a stage result into named, coloured point groups for display."""
    arrays = project.arrays
    point_count = project.point_count
    parts = []

    def add(key, name, ids, color=None):
        ids = np.asarray(ids, dtype=np.int64)
        parts.append(dict(key=key, name=name, ids=ids, color=color))

    def instance_color(value):
        return [
            round(x * 255)
            for x in colorsys.hsv_to_rgb(value * 0.61803398875 % 1, 0.65, 0.95)
        ]

    def groups(field, prefix, scope):
        used = []
        for value in np.unique(arrays[field][scope]):
            if value < 0:
                continue
            ids = scope[arrays[field][scope] == value]
            used.append(ids)
            add(
                f"{field}_{value}",
                f"{prefix} {value}",
                ids,
                instance_color(int(value) + 4),
            )
        used = np.concatenate(used) if used else np.array([], dtype=np.int64)
        return np.setdiff1d(scope, used)

    if stage == "load":
        add("aligned", "Aligned points", arrays["active"])
    elif stage in ("ground", "poles", "wires"):
        color = {
            "ground": [115, 82, 46],
            "poles": [51, 89, 230],
            "wires": [230, 51, 51],
        }[stage]
        add("extracted", "Extracted " + stage, arrays[stage], color)
        add("remaining", "Remaining points", arrays["active"])
    elif stage == "roi":
        add("kept", "ROI kept points", arrays["active"])
        add("excluded", "ROI excluded points", arrays["out_of_roi"], [220, 90, 90])
    elif stage == "denoise":
        revision = project.head["completed"]["roi"]
        project.revision_state(revision)
        with np.load(
            project.root / "revisions" / revision / "arrays.npz", allow_pickle=False
        ) as saved:
            inputs = saved["active"]
        add("kept", "Denoise kept points", arrays["tree_points"])
        add(
            "removed",
            "Denoise removed points",
            np.setdiff1d(inputs, arrays["tree_points"]),
            [220, 90, 90],
        )
    elif stage in ("stumps", "trunks"):
        remaining_ids = groups(
            stage + "_group",
            "Stump cluster" if stage == "stumps" else "Trunk cluster",
            arrays["tree_points"],
        )
        add("remaining", "Other tree points", remaining_ids)
    elif stage == "match_trunks":
        scope = arrays["tree_points"]
        trunks = scope[arrays["trunks_group"][scope] >= 0]
        groups("matched_trunk", "Matched trunks - tree", trunks)
        for part in parts:
            value = int(part["key"].rsplit("_", 1)[1])
            part["name"] = f"Matched trunks - tree {value + 4}"
        add(
            "unmatched",
            "Unmatched trunk points",
            trunks[arrays["matched_trunk"][trunks] < 0],
            [220, 90, 90],
        )
        add("remaining", "Other tree points", np.setdiff1d(scope, trunks))
    elif stage == "branches":
        scope = arrays["tree_points"]
        valid_fragment_ids = set(arrays["valid_fragments"].tolist())
        used = []
        for fragment_id in np.unique(arrays["fragment_id"][scope]):
            if fragment_id < 0:
                continue
            ids = scope[arrays["fragment_id"][scope] == fragment_id]
            used.append(ids)
            accepted = int(fragment_id) in valid_fragment_ids
            add(
                f"fragment_{fragment_id}",
                f"Branch fragment {fragment_id}" + ("" if accepted else " - rejected"),
                ids,
                instance_color(int(fragment_id) + 4) if accepted else [220, 90, 90],
            )
        used = np.concatenate(used) if used else np.array([], dtype=np.int64)
        add("remaining", "Other tree points", np.setdiff1d(scope, used))
    elif stage in ("assign_branches", "refill"):
        names = {1: "Ground", 2: "Poles", 3: "Wires"}
        colors = {1: [115, 82, 46], 2: [51, 89, 230], 3: [230, 51, 51]}
        for label in np.unique(arrays["labels"]):
            if label <= 0:
                continue
            ids = np.flatnonzero(arrays["labels"] == label)
            name = names.get(int(label), f"Tree {label}")
            color = colors.get(int(label), instance_color(int(label)))
            added = (
                ids[arrays["provenance"][ids] == 2]
                if stage == "refill"
                else np.array([], dtype=np.int64)
            )
            add(f"label_{label}", name, np.setdiff1d(ids, added), color)
            if stage == "refill":
                add(f"refill_{label}", name + " - refill added", added, color)
        unassigned = np.flatnonzero(arrays["labels"] == 0)
        if "fragment_id" in arrays:
            remaining = groups("fragment_id", "Unassigned branch fragment", unassigned)
        else:
            remaining = unassigned
        reasons = {
            0: "Unassigned points",
            1: "ROI excluded points",
            2: "Denoise removed points",
            3: "Unclustered layer points",
            4: "Rejected fragment points",
            5: "Unassigned branch points",
            6: "Other unassigned points",
        }
        for reason in np.unique(arrays["reason"][remaining]):
            add(
                f"unassigned_{reason}",
                reasons.get(int(reason), f"Unassigned reason {reason}"),
                remaining[arrays["reason"][remaining] == reason],
                [150, 150, 150],
            )
    else:
        return []
    seen = np.zeros(point_count, dtype=bool)
    for part in parts:
        ids = part["ids"]
        if ids.ndim != 1 or (len(ids) and (ids.min() < 0 or ids.max() >= point_count)):
            raise ValueError("Invalid stage output point IDs")
        if len(np.unique(ids)) != len(ids) or seen[ids].any():
            raise ValueError("Stage output point clouds overlap")
        seen[ids] = True
    return parts
