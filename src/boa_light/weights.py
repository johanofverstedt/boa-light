"""Model weight management.

Weights are stored in the same directory TotalSegmentator uses
(``$TOTALSEG_WEIGHTS_PATH`` or ``$TOTALSEG_HOME_DIR/nnunet/results``, default
``~/.totalsegmentator/nnunet/results``), so an existing TotalSegmentator/BOA
installation is reused without downloading anything again. ``$BOA_LIGHT_WEIGHTS``
overrides the location.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from boa_light.models import NETWORKS, Network

logger = logging.getLogger(__name__)


def weights_dir() -> Path:
    if "BOA_LIGHT_WEIGHTS" in os.environ:
        return Path(os.environ["BOA_LIGHT_WEIGHTS"])
    if "TOTALSEG_WEIGHTS_PATH" in os.environ:
        return Path(os.environ["TOTALSEG_WEIGHTS_PATH"])
    if "TOTALSEG_HOME_DIR" in os.environ:
        home = Path(os.environ["TOTALSEG_HOME_DIR"])
    else:
        home = (Path("/tmp") if str(Path.home()) == "/" else Path.home()) / ".totalsegmentator"
    return home / "nnunet" / "results"


def model_folder(net: Network, root: Path | None = None) -> Path:
    root = root or weights_dir()
    return root / net.folder / f"{net.trainer}__{net.plans}__{net.configuration}"


def is_available(task_id: int, root: Path | None = None) -> bool:
    folder = model_folder(NETWORKS[task_id], root)
    return (folder / "plans.json").is_file() and any(folder.glob("fold_*/checkpoint_final.pth"))


def _download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url) as response, target.open("wb") as fh:  # noqa: S310
        total = int(response.headers.get("content-length", 0))
        done = 0
        chunk = 1 << 20
        while True:
            buf = response.read(chunk)
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
            if total and sys.stderr.isatty():
                sys.stderr.write(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB")
        if total and sys.stderr.isatty():
            sys.stderr.write("\n")


def download(task_id: int, root: Path | None = None, force: bool = False) -> Path:
    """Download and unpack the weights of one network if they are missing."""
    net = NETWORKS[task_id]
    root = root or weights_dir()
    folder = model_folder(net, root)
    if not force and is_available(task_id, root):
        return folder
    root.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading weights for task %s (%s)", task_id, net.folder)
    with tempfile.TemporaryDirectory(dir=root, prefix=".download_") as tmp:
        zip_path = Path(tmp) / "weights.zip"
        _download(net.url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():  # guard against path traversal
                if Path(member).is_absolute() or ".." in Path(member).parts:
                    raise RuntimeError(f"Refusing to extract unsafe path {member!r}")
            extract_dir = Path(tmp) / "x"
            zf.extractall(extract_dir)
        src = next(extract_dir.rglob(net.folder), None)
        if src is None:
            raise RuntimeError(f"Archive {net.url} does not contain {net.folder}")
        dst = root / net.folder
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), dst)
    if not is_available(task_id, root):
        raise RuntimeError(f"Weights for task {task_id} incomplete after download: {folder}")
    return folder


def ensure(task_ids: list[int], root: Path | None = None) -> None:
    for tid in task_ids:
        download(tid, root)
