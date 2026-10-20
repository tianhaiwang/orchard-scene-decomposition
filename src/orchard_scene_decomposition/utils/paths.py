import os
from pathlib import Path, PureWindowsPath
import subprocess


def source_path(value):
    """Return a local path; Windows paths are mapped with wslpath under WSL."""
    path = Path(value)
    windows_path = PureWindowsPath(value)
    if (
        os.name != "posix"
        or not windows_path.is_absolute()
        or not windows_path.drive.endswith(":")
    ):
        return path
    kernel_release = Path("/proc/sys/kernel/osrelease")
    if (
        not kernel_release.is_file()
        or "microsoft" not in kernel_release.read_text(encoding="utf-8").lower()
    ):
        raise ValueError("A Windows source path requires Windows or WSL")
    result = subprocess.run(
        ["wslpath", "-u", windows_path.as_posix()],
        capture_output=True,
        text=True,
        check=True,
    )
    mapped_path = result.stdout.strip()
    if not mapped_path or not Path(mapped_path).is_absolute():
        raise ValueError("WSL did not return an absolute source path")
    return Path(mapped_path)
