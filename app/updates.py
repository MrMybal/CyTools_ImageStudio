"""Verify and stage GitHub release assets. Only app/ is swapped at startup."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZipFile

from cytools_core import load_manifest
from cytools_core.storage import RuntimeLock
import requests

from settings import DATA, ROOT, RUNTIME

UPDATES = RUNTIME / "updates"
APP = ROOT / "app"
ASSET = "CyTools_ImageStudio-app.zip"
SIZE_LIMIT = 128 * 1024**2
REQUIRED = {"desktop.py", "studio_ui.py", "studio_runtime.py", "CyTool.json", "updates.py", "tool.py"}
VERSION_PATTERN = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")


def version(text):
    match = VERSION_PATTERN.fullmatch(text)
    if not match:
        raise ValueError("Updates require a numeric release version.")
    return tuple(map(int, match.groups()))


def check(repo):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or any(
            part in (".", "..") for part in repo.split("/")):
        raise ValueError("Enter the GitHub repository as owner/name.")
    with requests.get(f"https://api.github.com/repos/{repo}/releases/latest", timeout=(5, 15)) as response:
        response.raise_for_status()
        release = response.json()
    tag = release["tag_name"]
    if version(tag) <= version(load_manifest(APP / "CyTool.json")["version"]):
        return None
    item = next((item for item in release.get("assets", []) if item["name"] == ASSET), None)
    if item is None:
        raise ValueError(f"The release has no {ASSET} asset.")
    digest = item.get("digest", "")
    url = item["browser_download_url"]
    if not (url.lower().startswith(f"https://github.com/{repo.lower()}/releases/download/")
            and url.endswith("/" + ASSET) and re.fullmatch(r"sha256:[a-f0-9]{64}", digest or "")
            and 0 < item["size"] <= SIZE_LIMIT):
        raise ValueError("The GitHub release asset must have a valid provenance, size and SHA256.")
    return {"tag": tag, "url": url, "digest": digest[7:], "size": item["size"]}


def extract(archive, target, tag):
    with ZipFile(archive) as zip_file:
        entries = zip_file.infolist()
        if sum(entry.file_size for entry in entries) > SIZE_LIMIT or len(entries) > 10_000:
            raise ValueError("Update archive exceeds extraction limits.")
        names = set()
        for entry in entries:
            raw = entry.filename.rstrip("/")
            parts = PurePosixPath(raw).parts
            if (not parts or parts[0] != "app" or "\\" in raw or ":" in raw or
                    any(part in {"", ".", ".."} or part.endswith((" ", ".")) for part in raw.split("/")) or
                    stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError("Update contains a path outside app/ or an unsupported link.")
            casefold = raw.lower()
            if casefold in names:
                raise ValueError("Update contains duplicate paths.")
            names.add(casefold)
            if entry.is_dir():
                continue
            destination = target.joinpath(*parts[1:])
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(entry) as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output)
    manifest = load_manifest(target / "CyTool.json")
    if manifest["id"] != "cy.tools.imagestudio" or version(manifest["version"]) != version(tag):
        raise ValueError("Update has a different tool identity or version.")
    if not all((target / name).is_file() for name in REQUIRED):
        raise ValueError("Update is missing application entry points.")
    for path in target.rglob("*.py"):
        compile(path.read_bytes(), str(path), "exec")


def stage(release):
    UPDATES.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="fetch-", dir=UPDATES) as temp:
        area = Path(temp)
        archive = area / ASSET
        digest = hashlib.sha256()
        written = 0
        with requests.get(release["url"], timeout=(5, 20), stream=True) as response, archive.open("xb") as file:
            response.raise_for_status()
            for block in response.iter_content(1024 * 1024):
                written += len(block)
                if written > release["size"]:
                    raise ValueError("Update is larger than the GitHub asset size.")
                digest.update(block)
                file.write(block)
        if written != release["size"] or digest.hexdigest() != release["digest"]:
            raise ValueError("Update ZIP failed size or SHA256 verification.")
        target = area / "app"
        target.mkdir()
        extract(archive, target, release["tag"])
        name = "staged-" + uuid4().hex
        target.rename(UPDATES / name)
        return name


def schedule(target):
    if not re.fullmatch(r"(staged|backup)-[a-f0-9]{32}", target):
        raise ValueError("Invalid staged application identifier.")
    UPDATES.mkdir(parents=True, exist_ok=True)
    temporary = UPDATES / ("action-" + uuid4().hex + ".tmp")
    temporary.write_text(target, encoding="ascii")
    temporary.replace(UPDATES / "next.txt")


def activate(target):
    previous = UPDATES / ("backup-" + uuid4().hex)
    APP.rename(previous)
    try:
        target.rename(APP)
    except BaseException:
        previous.rename(APP)
        raise
    return previous


def apply_on_startup():
    action = UPDATES / "next.txt"
    if not action.exists():
        return
    name = action.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"(staged|backup)-[a-f0-9]{32}", name):
        raise ValueError("Invalid pending application update.")
    target = UPDATES / name
    if target.is_symlink() or not (target / "CyTool.json").is_file():
        raise ValueError("Pending application is missing.")
    job_root = DATA / "jobs"
    job_root.mkdir(parents=True, exist_ok=True)
    lock = RuntimeLock(job_root / "runtime.lock.sqlite3")
    try:
        if not action.exists():
            return
        backup = activate(target)
        action.unlink()
        (UPDATES / "previous.txt").write_text(backup.name, encoding="ascii")
    finally:
        lock.close()
    os.execv(sys.executable, [sys.executable, *sys.argv])


_pool = None
_future = None
_result = None
_staged = ""
_job = ""


def draw(studio):
    from imgui_bundle import imgui
    from ui_common import label, tr
    import cyui

    global _pool, _future, _result, _staged, _job
    if not imgui.collapsing_header(label("Application updates")):
        return
    _, studio.config["update_repo"] = imgui.input_text(label("GitHub repository (owner/name)"),
                                                       studio.config.get("update_repo", ""))
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=1)
    if _future and _future.done():
        try:
            output = _future.result()
            if _job == "check":
                _result = output
                studio.message = tr("Update available") + " " + output["tag"] if output else tr("Already up to date")
            else:
                _staged = output
                studio.message = tr("Update verified and staged")
        except Exception as error:
            studio.error = str(error)
        finally:
            _future = None
    with cyui.scoped_disabled(bool(_future)):
        if imgui.button(label("Check GitHub release")):
            _job = "check"
            _future = _pool.submit(check, studio.config.get("update_repo", "").strip())
        if _result:
            imgui.same_line()
            if imgui.button(label("Download and verify update")):
                _job = "stage"
                _future = _pool.submit(stage, _result)
    if _staged and imgui.button(label("Apply on next launch")):
        schedule(_staged)
        studio.message = tr("Restart ImageStudio to apply the update.")
    previous = UPDATES / "previous.txt"
    if previous.is_file() and imgui.button(label("Restore previous version on next launch")):
        schedule(previous.read_text(encoding="ascii"))
        studio.message = tr("Restart ImageStudio to restore the previous version.")
