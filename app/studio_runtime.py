"""ImageStudio operations shared by CyUI, CLI and the native MCP host."""
from contextlib import contextmanager
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlsplit
from uuid import uuid4

from cytools_core import CyToolError, Runtime, load_manifest
from cytools_core.communication import register_operations
from cytools_core.downloads import DownloadManager
from cytools_core.shared_installation import register_operations as register_storage
from cytools_core.storage import RuntimeLock
from PIL import Image, ImageOps, UnidentifiedImageError

import graphs
from engine import Engine, install, installed_paths
from installation_ui import configure as configure_storage
from settings import CATEGORIES, DATA, ROOT, read
from selection import suggestions

APP = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpeg", ".jpg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
MODEL_SUFFIXES = {".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".sft", ".onnx"}


def prepare_image(context, source, name, mode):
    try:
        with Image.open(source) as image:
            if image.width * image.height > 40_000_000:
                raise ValueError("Image exceeds 40 megapixels.")
            if image.format not in {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}:
                raise ValueError("Unsupported image format.")
            target = context.path(f"prepared/{name}.png")
            ImageOps.exif_transpose(image).convert(mode).save(target)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise CyToolError("InvalidImage", str(error)) from error
    return target


def store_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return str(path)


def result_assets(client, context, history):
    result = []
    for node, outputs in history.get("outputs", {}).items():
        for items in outputs.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or not item.get("filename"):
                    continue
                suffix = Path(item["filename"]).suffix.lower()
                if suffix not in IMAGE_SUFFIXES | {".safetensors"} or item.get("type", "output") not in ("output", "temp"):
                    continue
                name = f"{len(result) + 1:03d}{suffix}"
                destination = context.path(f"outputs/{name}")
                client.download(item, destination, context)
                asset = {"file": str(destination), "node": str(node),
                         "mimeType": mimetypes.guess_type(name)[0] or "application/octet-stream",
                         "name": Path(item["filename"]).name}
                if suffix in IMAGE_SUFFIXES:
                    with Image.open(destination) as image:
                        asset.update(width=image.width, height=image.height)
                result.append(asset)
    return result


@contextmanager
def connected(engine, context, parameters, *, cleanup=False):
    config = read()
    log_mark = engine.log_sequence
    with engine.open(config, parameters.get("profile"), context) as client:
        try:
            yield client
        except CyToolError as error:
            if error.code == "SubmissionUncertain" and client.owned:
                engine.stop()
            raise
        finally:
            if cleanup and client.owned and config["release"] and engine.running():
                try:
                    client.free()
                except (CyToolError, ValueError) as error:
                    context.log(str(error), level="Warning")
            if client.owned:
                engine.snapshot_log(context, log_mark)


@contextmanager
def staging(engine, client, context):
    prefix = "imagestudio_" + uuid4().hex
    area = Path(engine.area.name) if client.owned else None
    try:
        yield prefix, area
    finally:
        if area is not None:
            for folder in ("input", "output", "temp"):
                target = area / folder / prefix
                try:
                    if target.is_dir() and not target.is_symlink():
                        shutil.rmtree(target)
                except OSError as error:
                    context.log(f"ComfyUI staging cleanup: {error}", level="Warning")


def run_graph(client, context, prompt, parameters, info):
    graphs.validate_nodes(prompt, info)
    graph_file = store_json(context.path("outputs/workflow.api.json"), prompt)
    prompt_id, history = client.run(prompt, context, parameters.get("timeout", 3600))
    context.progress(99, currentStep="Collecting ComfyUI outputs")
    files = result_assets(client, context, history)
    history_file = store_json(context.path("outputs/history.json"), history)
    return {"promptId": prompt_id, "workflow": graph_file, "history": history_file, "assets": files}


def document(p):
    if "file" in p:
        return graphs.read_document(p["file"])
    return p["graph"]


def create_runtime(data_dir, **options):
    configure_storage()
    runtime = Runtime(load_manifest(APP / "CyTool.json"), data_dir, **options)
    runtime.configure_inputs(DATA / "inputs", fields={
        "generate": ["image", "mask", "controls.*.image"], "run_workflow": ["file", "uploads.*.file"],
        "inspect_workflow": ["file"], "train_lora": ["dataset.*.image"], "import_model": ["file"],
    })
    engine = Engine()
    runtime.register_worker("comfyui", engine)

    def status(context, p):
        config = read()
        return {"defaultProfile": config["profile"], "managedInstalled": bool(installed_paths()),
                "existingConfigured": bool(config["main"] and config["python"]), "ownedEngineRunning": engine.running(),
                "externalConfigured": [key for key, value in config["profiles"].items() if value["url"]],
                "modelsDirectory": config["models"], "modelSearchPaths": config["roots"],
                "memoryProfile": config["memory"], "vramReserveGiB": config["reserve"]}

    def inventory(context, p):
        with connected(engine, context, p) as client:
            info = client.info()
            stats = client.json("/system_stats")
            system = stats.get("system", {})
            devices = [{key: device[key] for key in ("name", "type", "index", "vram_total", "vram_free")
                        if key in device} for device in stats.get("devices", [])]
            return {"models": graphs.catalog(info), "devices": devices,
                    "system": {key: system[key] for key in ("os", "ram_total", "ram_free", "comfyui_version") if key in system},
                    "samplers": graphs.choices(info, "KSampler", "sampler_name"),
                    "schedulers": graphs.choices(info, "KSampler", "scheduler"),
                    "guidedTraining": client.owned and all(name in info for name in
                        ("LoadImageTextDataSetFromFolder", "MakeTrainingDataset", "TrainLoraNode", "SaveLoRA")),
                    "nodeTypes": sorted(info) if p["include_nodes"] else [], "owned": client.owned}

    def generate(context, p):
        if p["width"] * p["height"] * p["batch"] > 32_000_000:
            raise CyToolError("InvalidParameters", "The requested batch exceeds 32 million pixels.")
        if p["tiled_vae"] and p["tile_overlap"] * 2 >= p["tile_size"]:
            raise CyToolError("InvalidParameters", "VAE overlap must be less than half the tile size.")
        with connected(engine, context, p, cleanup=True) as client, staging(engine, client, context) as (folder, _):
            info = client.info()
            graphs.models(graphs.Graph(), p)
            originals = {}
            if p["mode"] != "txt2img":
                originals["image"] = (p["image"], "RGB")
            if p["mode"] == "inpaint":
                originals["mask"] = (p["mask"], "L")
            originals.update({f"control_{index}": (control["image"], "RGB")
                              for index, control in enumerate(p["controls"])})
            uploaded = {}
            for key, (source, mode) in originals.items():
                context.cancellation.check()
                uploaded[key] = client.upload(prepare_image(context, source, key, mode), folder)
            prompt, seed = graphs.generate(p, info, uploaded, f"{folder}/image")
            outcome = run_graph(client, context, prompt, p, info)
            if not outcome["assets"]:
                raise CyToolError("NoImageOutput", "ComfyUI finished but provided no downloadable image.")
            return {**outcome, "seed": seed}

    def inspect(context, p):
        with connected(engine, context, p) as client:
            info = client.info()
            prompt = graphs.workflow(document(p), info)
            kinds = sorted({node["class_type"] for node in prompt.values()})
            return {"nodes": len(prompt), "nodeTypes": kinds,
                    "missingNodes": [kind for kind in kinds if kind not in info],
                    "editableInputs": [{"node": node, "type": value["class_type"], "input": key, "value": initial}
                                       for node, value in prompt.items() for key, initial in value["inputs"].items()
                                       if not isinstance(initial, (list, dict))]}

    def workflow(context, p):
        with connected(engine, context, p, cleanup=True) as client, staging(engine, client, context) as (folder, _):
            info = client.info()
            prompt = graphs.workflow(document(p), info)
            graphs.validate_nodes(prompt, info)
            for item in [*p["overrides"], *p["uploads"]]:
                key, field = item["node"], item["input"]
                if key not in prompt or field not in graphs.inputs(info, prompt[key]["class_type"]):
                    raise CyToolError("InvalidWorkflowInput", f"Node {key} has no input {field}.")
            for item in p["overrides"]:
                prompt[item["node"]]["inputs"][item["input"]] = item["value"]
            for index, item in enumerate(p["uploads"]):
                context.cancellation.check()
                image = prepare_image(context, item["file"], f"workflow_{index}", "RGBA")
                prompt[item["node"]]["inputs"][item["input"]] = client.upload(image, folder)
            if client.owned:
                for node in prompt.values():
                    definition = info.get(node["class_type"], {})
                    if not definition.get("output_node"):
                        continue
                    for key in ("filename_prefix", "prefix"):
                        value = node["inputs"].get(key)
                        if isinstance(value, str):
                            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(value.replace("\\", "/")).name).strip(".")
                            node["inputs"][key] = f"{folder}/{safe[:100] or 'output'}"
            return run_graph(client, context, prompt, p, info)

    def train(context, p):
        with connected(engine, context, p, cleanup=True) as client, staging(engine, client, context) as (folder, area):
            if area is None:
                raise CyToolError("UnsupportedTraining", "Guided training requires an ImageStudio-launched engine. Use a training workflow for an external server.")
            directory = area / "input" / folder
            directory.mkdir(parents=True, exist_ok=True)
            for number, entry in enumerate(p["dataset"], 1):
                context.cancellation.check()
                prepared = prepare_image(context, entry["image"], f"sample_{number:03d}", "RGB")
                shutil.copyfile(prepared, directory / prepared.name)
                (directory / f"{prepared.stem}.txt").write_text(entry["caption"], encoding="utf-8")
            info = client.info()
            prompt = graphs.train(p, info, folder, f"{folder}/trained")
            outcome = run_graph(client, context, prompt, p, info)
            for file in sorted((area / "output" / folder).glob("*.safetensors")):
                target = context.path(f"outputs/{file.name}")
                copy_stream(context, file, target)
                outcome["assets"].append({"file": str(target), "name": file.name, "node": "SaveLoRA",
                                          "mimeType": "application/octet-stream"})
            if not any(item["file"].endswith(".safetensors") for item in outcome["assets"]):
                raise CyToolError("NoLoRAOutput", "Training finished but SaveLoRA did not produce weights.")
            return outcome

    def release(context, p):
        profile = p.get("profile") or read()["profile"]
        if p["stop"] and profile not in ("managed", "existing"):
            raise CyToolError("InvalidParameters", "Only an ImageStudio-launched engine can be stopped.")
        if profile in ("managed", "existing"):
            if p["stop"]:
                engine.stop()
            if not engine.running():
                return {"requested": False, "engineStopped": True}
            engine.release("Model")
        else:
            with connected(engine, context, p) as client:
                client.free()
        return {"requested": True, "engineStopped": False}

    def suggest(context, p):
        with connected(engine, context, p) as client:
            return {"candidates": suggestions(client.info(), p["mode"], p["family"])}

    handlers = {"studio_status": status, "inspect_backend": inventory, "generate": generate, "suggest_models": suggest,
                "run_workflow": workflow, "inspect_workflow": inspect, "train_lora": train,
                "download_model": download_model, "import_model": import_model, "release_memory": release,
                "install_engine": lambda ctx, p: install(ctx, p["compute"])}
    for name, handler in handlers.items():
        runtime.register(name, handler)
    register_operations(runtime)
    register_storage(runtime, ROOT)
    return runtime


def model_target(category, filename):
    if category not in CATEGORIES or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,140}", filename):
        raise CyToolError("InvalidParameters", "Use a supported category and a simple model filename.")
    if Path(filename).suffix.lower() not in MODEL_SUFFIXES or category == "diffusers":
        raise CyToolError("InvalidParameters", "Select a supported single-file model format.")
    root = Path(read()["models"]).expanduser().resolve()
    target = root / category / filename
    if target.resolve() != target or not target.is_relative_to(root):
        raise CyToolError("AccessDenied", "The requested file is redirected from the model library.")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def hub_token():
    default = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "token"
    path = Path(os.environ.get("HF_TOKEN_PATH", str(default)))
    token = os.environ.get("IMAGESTUDIO_HF_TOKEN") or os.environ.get("HF_TOKEN", "")
    return token.strip() or (path.read_text(encoding="utf-8").strip() if path.is_file() else "")


def download_model(context, p):
    if not p["accept_license"]:
        raise CyToolError("LicenseAcceptanceRequired", "Read and accept the model provider's terms.")
    url = urlsplit(p["url"])
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise CyToolError("InvalidParameters", "Use a direct HTTPS URL without credentials, query or fragment.")
    target = model_target(p["category"], p["filename"])
    lock = RuntimeLock(target.with_name("." + target.name + ".lock"))
    try:
        if target.exists() and not target.with_name(target.name + ".installed.json").is_file():
            raise CyToolError("ModelAlreadyExists", "A model with this name already exists. Pick a different filename.")
        if p.get("size") and not target.exists():
            partial = target.with_name(target.name + ".part")
            reusable = partial.stat().st_size if partial.is_file() else 0
            if shutil.disk_usage(target.parent).free + reusable < p["size"]:
                raise CyToolError("InsufficientDisk", "There is not enough free space in the model library.", recoverable=True)
        token = hub_token() if url.netloc == "huggingface.co" else ""
        last = 0.0

        def report(current, total):
            nonlocal last
            now = time.monotonic()
            if now - last >= 0.5:
                context.progress(min(99, current / total * 100) if total else 0,
                                 currentStep=f"{current / 1048576:.1f} MiB downloaded")
                last = now

        manager = DownloadManager(target.parent)
        result = manager.download(target.name, p["url"], sha256=p.get("sha256"), size=p.get("size"),
                                  revision=p.get("revision"), cancellation=context.cancellation, progress=report,
                                  headers={"Authorization": f"Bearer {token}"} if token else None)
        return {"file": result["path"], "category": p["category"], "bytes": target.stat().st_size,
                "sha256Verified": bool(p.get("sha256"))}
    finally:
        lock.close()


def copy_stream(context, source, target):
    with Path(source).open("rb") as incoming, target.open("xb") as outgoing:
        while block := incoming.read(1024 * 1024):
            context.cancellation.check()
            outgoing.write(block)


def import_model(context, p):
    target = model_target(p["category"], p["filename"])
    lock = RuntimeLock(target.with_name("." + target.name + ".lock"))
    pending = target.with_name(f".{target.name}.{uuid4().hex}.part")
    try:
        if target.exists():
            raise CyToolError("ModelAlreadyExists", "A model with this name already exists.")
        if Path(p["file"]).suffix.lower() != target.suffix.lower():
            raise CyToolError("InvalidParameters", "The installed model filename must keep the original extension.")
        copy_stream(context, p["file"], pending)
        pending.rename(target)
        return {"file": str(target), "category": p["category"], "bytes": target.stat().st_size,
                "sha256Verified": False}
    finally:
        pending.unlink(missing_ok=True)
