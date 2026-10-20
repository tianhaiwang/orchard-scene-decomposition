import argparse
import os
from pathlib import Path
import signal
from .paths import source_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    pid = int(source_path(args.pid_file).read_text(encoding="ascii"))
    command = Path("/proc") / str(pid) / "cmdline"
    if not command.exists():
        return 0
    argv = command.read_bytes().decode("utf-8").split("\x00")
    if "orchard_scene_decomposition.worker" not in argv or "--request" not in argv:
        raise RuntimeError(
            "Refusing to stop a process that is not the requested worker"
        )
    request = argv[argv.index("--request") + 1]
    if source_path(request) != source_path(args.request):
        raise RuntimeError("Worker request does not match; refusing to stop it")
    os.kill(pid, signal.SIGTERM)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
