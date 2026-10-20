import hashlib
import json
from datetime import datetime, timezone
from ..config import STAGES

GROUP_NAMES = [
    "Load-Align",
    "Ground",
    "Poles",
    "Wires",
    "ROI",
    "Denoise",
    "Stump-Clusters",
    "Trunk-Clusters",
    "Match-Trunks",
    "Branch-Fragments",
    "Assign-Branches",
    "Refill",
]


def result_stage(state, revision):
    completed = state.get("completed", {})
    for stage in STAGES:
        if completed.get(stage) == revision:
            return stage, "run"
    event = (state.get("events") or [{}])[-1].get("event", "")
    if event.startswith(("transfer fragment ", "split fragment ", "undo:", "redo:")):
        if "assign_branches" in completed:
            return "assign_branches", "manual"
    return None, None


def catalogue(project):
    """List the saved result versions of every stage, oldest first."""
    groups = {stage: [] for stage in STAGES}
    deleted = set(project.head.get("deleted_results", []))
    saved = []
    for folder in sorted((project.root / "revisions").iterdir()):
        if not folder.is_dir() or not (folder / "state.json").is_file():
            continue
        if not (folder / "arrays.npz").is_file():
            continue
        state = project.revision_state(folder.name)
        latest_event = (state.get("events") or [{}])[-1]
        event_time_text = latest_event.get("time", state.get("created"))
        event_time = datetime.fromisoformat(event_time_text.replace("Z", "+00:00"))
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        saved.append((event_time.timestamp(), folder.name, state))
    active = dict(project.head["completed"])
    marker = project.head.get("result_revision")
    if (
        not marker
        and result_stage(project.head, project.head["revision"])[1] == "manual"
    ):
        marker = project.head["revision"]
    for _, revision, state in saved:
        stage, kind = result_stage(state, revision)
        if revision == marker and stage in active:
            active[stage] = revision
    version_counts = {stage: 0 for stage in STAGES}
    for _, revision, state in sorted(saved):
        stage, kind = result_stage(state, revision)
        if stage is None:
            continue
        version_counts[stage] += 1
        if revision in deleted:
            continue
        latest_event = (state.get("events") or [{}])[-1]
        saved_config = state["config"]
        fingerprint = hashlib.sha256(
            json.dumps(saved_config, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        input_revision = (
            state["completed"].get(STAGES[STAGES.index(stage) - 1])
            if stage != "load"
            else None
        )
        groups[stage].append(
            dict(
                revision=revision,
                stage=stage,
                kind=kind,
                version=version_counts[stage],
                time=latest_event.get("time", state.get("created")),
                parameters=saved_config[stage],
                config_sha256=fingerprint,
                input_revision=input_revision,
                parent_revision=latest_event.get("parent"),
                current=active.get(stage) == revision,
            )
        )
    return [
        dict(
            stage=stage,
            group=f"S{stage_index + 1}_{GROUP_NAMES[stage_index]}",
            versions=groups[stage],
        )
        for stage_index, stage in enumerate(STAGES)
    ]
