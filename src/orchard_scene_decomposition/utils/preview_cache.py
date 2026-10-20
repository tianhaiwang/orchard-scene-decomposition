import json
from pathlib import Path
import warnings
from ..config import STAGES

PREVIEW_LIMIT = 3


def prune_previews(folder, current, source_sha256):
    folder, current = (Path(folder), Path(current))
    if folder.resolve() != folder or current.parent != folder:
        raise ValueError("Preview cache must be inside the project directory")
    current.with_suffix(".json").touch()
    candidates = []
    for manifest in folder.glob("*.json"):
        path = manifest.with_suffix(".ply")
        if manifest.resolve() != manifest or path.resolve() != path:
            continue
        try:
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(meta, dict) or meta.get("preview") is not True:
                continue
            source = meta.get("source")
            if not isinstance(source, dict) or source.get("sha256") != source_sha256:
                continue
            stage = meta.get("snapshot_stage")
            revision = meta.get("revision") or source_sha256[:16]
            if (
                stage not in (*STAGES, "source")
                or path.name != f"{stage}-{revision}.ply"
            ):
                continue
            if path.is_file():
                candidates.append((path == current, manifest.stat().st_mtime_ns, path))
        except (OSError, ValueError):
            continue
    candidates.sort(key=lambda item: (item[0], item[1], item[2].name), reverse=True)
    for _, _, path in candidates[PREVIEW_LIMIT:]:
        try:
            path.unlink()
            for sidecar in (
                path.with_suffix(".bridge.json"),
                path.with_suffix(".json"),
            ):
                if sidecar.resolve() == sidecar:
                    sidecar.unlink(missing_ok=True)
        except OSError as exc:
            warnings.warn(
                f"Preview cleanup deferred for {path.name}: {exc}. It will be retried next time you open SuperSplat.",
                RuntimeWarning,
            )
