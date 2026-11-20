"""Own only the children and staging areas created by ImageStudio."""
from collections import deque
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
from threading import Lock, Thread
import time

import psutil
from cytools_core import CyToolError
from cytools_core.storage import RuntimeLock
from comfy import Client, external_client
from settings import DATA, RUNTIME, endpoint, model_paths, read

COMFY_REPOSITORY = "https://github.com/Comfy-Org/ComfyUI.git"
COMFY_TAG = "v0.37.0"
COMFY_COMMIT = "73c9bad4d21e7addbe1d13bc92eee0f1431b017d"
TORCH_INDEX = {"cuda": "https://download.pytorch.org/whl/cu128",
               "cpu": "https://download.pytorch.org/whl/cpu"}
TORCH_PACKAGES = ("torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0")
CREATE_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def engine_environment():
    return {key: value for key, value in os.environ.items()
            if not key.startswith("IMAGESTUDIO_") and key.upper() not in {"PYTHONPATH", "PYTHONHOME"}}


def python_at(directory):
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def terminate(process):
    if process.poll() is not None:
        return
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    _, remaining = psutil.wait_procs(children, timeout=3)
    for child in remaining:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(remaining, timeout=5)
    if remaining:
        raise CyToolError("EngineStillRunning", "A ComfyUI child process did not stop.", recoverable=True)


def run_command(argv, context, log, *, cwd=None):
    with log.open("ab") as output:
        child = subprocess.Popen(argv, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, creationflags=CREATE_FLAGS, env=engine_environment())
        try:
            while child.poll() is None:
                context.cancellation.sleep(0.25)
        finally:
            terminate(child)
    if child.returncode:
        raise CyToolError("EngineInstallFailed", f"{Path(argv[0]).name} exited with {child.returncode}. See installer log.",
                          details={"log": str(log)}, recoverable=True)


def install(context, compute):
    import sys
    root = RUNTIME / "engines"
    root.mkdir(parents=True, exist_ok=True)
    log = context.path("outputs/comfy-install.log")
    lock = RuntimeLock(root / "install.lock")
    try:
        target = root / f"{COMFY_TAG}-{compute}"
        marker = target / "installed.json"
        if not marker.exists():
            # Build at its permanent path. Python virtual environments are not movable.
            target.mkdir(exist_ok=True)
            source = target / "ComfyUI"
            if not (source / ".git").is_dir():
                context.progress(5, currentStep="Downloading ComfyUI")
                with tempfile.TemporaryDirectory(prefix="clone-", dir=target) as staging:
                    staged = Path(staging) / "ComfyUI"
                    run_command(["git", "clone", "--depth", "1", "--branch", COMFY_TAG,
                                 COMFY_REPOSITORY, str(staged)], context, log)
                    staged.rename(source)
            revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                               creationflags=CREATE_FLAGS, text=True).strip()
            if revision != COMFY_COMMIT:
                raise CyToolError("EngineIntegrityFailure", "The ComfyUI release commit does not match the expected revision.")
            env = target / "python"
            if not python_at(env).is_file():
                context.progress(15, currentStep="Creating private ComfyUI Python")
                base_python = Path(getattr(sys, "_base_executable", sys.executable))
                if base_python.name.lower() == "pythonw.exe":
                    base_python = base_python.with_name("python.exe")
                run_command([str(base_python), "-m", "venv", str(env)], context, log)
            python = python_at(env)
            pip = [str(python), "-m", "pip", "--disable-pip-version-check", "install"]
            context.progress(30, currentStep="Installing PyTorch")
            run_command([*pip, "--index-url", TORCH_INDEX[compute], *TORCH_PACKAGES], context, log)
            context.progress(70, currentStep="Installing ComfyUI dependencies")
            run_command([*pip, "-r", str(source / "requirements.txt")], context, log)
            run_command([str(python), str(source / "main.py"), "--help"], context, log, cwd=source)
            marker.write_text(json.dumps({"commit": revision, "compute": compute}), encoding="utf-8")
        selection = RUNTIME / "engine.json"
        temporary = selection.with_suffix(".tmp")
        temporary.write_text(json.dumps({"main": str(target / "ComfyUI/main.py"),
                                         "python": str(python_at(target / "python")), "compute": compute}), encoding="utf-8")
        temporary.replace(selection)
        return {"compute": compute, "revision": COMFY_COMMIT, "log": str(log) if log.exists() else ""}
    finally:
        lock.close()


def installed_paths():
    record = RUNTIME / "engine.json"
    return json.loads(record.read_text(encoding="utf-8")) if record.exists() else {}


class Engine:
    def __init__(self):
        self.process = None
        self.area = None
        self.url = ""
        self.signature = None
        self.reader = None
        self.log_lock = Lock()
        self.log_sequence = 0
        self.messages = deque(maxlen=500)

    def stream_log(self, process):
        with process.stdout as pipe:
            for line in pipe:
                with self.log_lock:
                    self.log_sequence += 1
                    self.messages.append((self.log_sequence, line[:4000]))

    def snapshot_log(self, context, after):
        with self.log_lock:
            text = "".join(line for sequence, line in self.messages if sequence > after)
        if text:
            path = context.path("logs/comfy-engine.log")
            path.write_text(text, encoding="utf-8")
            return str(path)
        return ""

    def running(self):
        return self.process is not None and self.process.poll() is None

    def stop(self):
        if self.process is not None:
            terminate(self.process)
            self.process = None
        if self.reader is not None:
            self.reader.join(timeout=3)
            self.reader = None
        if self.area is not None:
            self.area.cleanup()
            self.area = None
        self.signature = None

    def release(self, level):
        if level in ("All", "Worker"):
            self.stop()
        elif self.running():
            with Client(self.url, owned=True) as client:
                client.free()

    def open(self, config, profile, context):
        profile = profile or config["profile"]
        if profile not in ("managed", "existing"):
            external = config["profiles"][profile]
            if not external["url"]:
                raise CyToolError("ConfigurationRequired", f"Configure the {profile} ComfyUI URL in Options.", recoverable=True)
            url = endpoint(external["url"])
            return Client(url, external_client(url, profile), targeted=external.get("targeted_interrupt", False))

        paths = installed_paths() if profile == "managed" else config
        if not (paths.get("main") and paths.get("python")):
            message = "Install the managed engine in Options." if profile == "managed" else "Select main.py and Python for the existing installation in Options."
            raise CyToolError("EngineMissing", message, recoverable=True)
        main, python = (Path(paths[key]).expanduser().resolve() for key in ("main", "python"))
        if main.name != "main.py" or not main.is_file() or not python.is_file():
            raise CyToolError("EngineMissing", "The configured ComfyUI main.py or Python is missing.", recoverable=True)
        models = Path(config["models"]).expanduser().resolve()
        models.mkdir(parents=True, exist_ok=True)
        extra = model_paths(config, main)
        memory = "cpu" if paths.get("compute") == "cpu" else config["memory"]
        signature = (main, python, models, json.dumps(extra, sort_keys=True), memory,
                     config["reserve"], config["device"], config["custom_nodes"])
        if self.running() and signature == self.signature:
            return Client(self.url, owned=True, alive=self.running)
        self.stop()

        base = RUNTIME / "comfy-state"
        base.mkdir(parents=True, exist_ok=True)
        self.area = tempfile.TemporaryDirectory(prefix="instance-", dir=base, ignore_cleanup_errors=True)
        area = Path(self.area.name)
        config_file = area / "models.yaml"
        config_file.write_text(json.dumps(extra, ensure_ascii=False), encoding="utf-8")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        command = [str(python), str(main), "--listen", "127.0.0.1", "--port", str(port), "--base-directory",
                   str(area), "--models-directory", str(models), "--extra-model-paths-config", str(config_file),
                   "--cache-none", "--disable-auto-launch"]
        if not config["custom_nodes"]:
            command.append("--disable-all-custom-nodes")
        if memory == "cpu":
            command.append("--cpu")
        else:
            command.extend(("--reserve-vram", str(config["reserve"])))
            if config["device"].strip():
                command.extend(("--cuda-device", str(int(config["device"]))))
            if memory == "low":
                command.extend(("--disable-dynamic-vram", "--lowvram"))
            if memory == "minimum":
                command.append("--novram")
        self.url = f"http://127.0.0.1:{port}"
        log_mark = self.log_sequence
        try:
            self.process = subprocess.Popen(command, cwd=main.parent, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=CREATE_FLAGS,
                                            env=engine_environment(), text=True, encoding="utf-8", errors="replace")
            self.reader = Thread(target=self.stream_log, args=(self.process,), daemon=True)
            self.reader.start()
            context.log("Starting ComfyUI with the " + memory + " memory profile.")
            until = time.monotonic() + 180
            with Client(self.url, owned=True, alive=self.running) as probe:
                while self.running() and time.monotonic() < until:
                    context.cancellation.check()
                    try:
                        probe.json("/system_stats", timeout=(0.5, 2))
                        self.signature = signature
                        return Client(self.url, owned=True, alive=self.running)
                    except CyToolError:
                        context.cancellation.sleep(0.5)
            raise CyToolError("EngineStartFailed", "ComfyUI failed to start. Check the engine log.",
                              details={"log": self.snapshot_log(context, log_mark)}, recoverable=True)
        except BaseException:
            self.stop()
            self.snapshot_log(context, log_mark)
            raise
