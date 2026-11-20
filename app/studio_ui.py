"""Native CyUI workspace; all execution is scheduled by the shared CyTools runtime."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
from time import monotonic
import webbrowser

from imgui_bundle import hello_imgui, imgui

import cyui
from advanced_ui import Panel
from canvas_ui import ImageCanvas
from cytools_core.transports.http import make_server
from generation_config import recipe, save as save_recipe
from settings import CATEGORIES, DATA, endpoint, read, save
from studio_runtime import create_runtime
from ui_common import configure_generation, journal, label, options_panel, preferences, tr

TERMINAL = {"Failed", "Cancelled", "Completed"}
PROFILE_NAMES = {
    "managed": "ImageStudio installation", "existing": "Installed ComfyUI",
    "local": "Local server", "remote": "Remote server",
}
MODEL_FILES = {"checkpoint": "checkpoints", "diffusion": "diffusion_models",
               "encoder": "text_encoders", "encoder2": "text_encoders", "vae": "vae"}
TASKS = {"txt2img": "Text to image", "img2img": "Image to image", "inpaint": "Inpainting"}
FAMILIES = {"checkpoint": "Checkpoint · SD / SDXL / all-in-one", "flux": "FLUX.1", "flux2": "FLUX.2",
            "qwen": "Qwen Image", "qwen_edit": "Qwen Image Edit", "sd3": "Stable Diffusion 3"}
FIELD_LABELS = {
    "checkpoint": "Checkpoint", "diffusion": "Diffusion model", "encoder": "Text encoder",
    "encoder2": "Second text encoder", "vae": "VAE", "dtype": "Weight precision", "encoder_device": "Encoder device",
    "sampler": "Sampler", "scheduler": "Scheduler", "width": "Width", "height": "Height", "steps": "Steps",
    "batch": "Images", "cfg": "CFG", "guidance": "FLUX guidance", "denoise": "Denoise",
    "inpaint_method": "Inpainting method", "tiled_vae": "Tiled VAE decode", "tile_size": "Tile size",
    "tile_overlap": "Tile overlap", "timeout": "Timeout (seconds)", "seed": "Seed (-1 = random)",
    "batch_size": "Batch size", "learning_rate": "Learning rate", "rank": "Rank",
    "grad_accumulation_steps": "Gradient accumulation", "gradient_checkpointing": "Gradient checkpointing",
    "optimizer": "Optimizer", "training_dtype": "Training precision", "offloading": "CPU weight offload",
    "existing_lora": "Continue from LoRA",
    "checkpoint_depth": "Checkpoint depth", "lora_dtype": "LoRA precision", "loss_function": "Loss function",
    "bypass_mode": "Adapter hooks for quantized models", "quantized_backward": "Quantized backward",
}


def choose(name, value, values, shown=None):
    display = shown or [tr(str(item)) for item in values]
    with cyui.scoped_id(name):
        if imgui.begin_combo(label(name), display[values.index(value)] if value in values else str(value)):
            for option, text in zip(values, display):
                if imgui.selectable(text, value == option)[0]:
                    value = option
            imgui.end_combo()
    return value


def dialog(kind, **kwargs):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        return getattr(filedialog, kind)(parent=root, **kwargs)
    finally:
        root.destroy()


def path_field(name, value="", *, kind="askopenfilename"):
    width = imgui.get_content_region_avail().x
    imgui.set_next_item_width(max(80, width * 0.54))
    _, value = imgui.input_text(label(name), value)
    imgui.same_line()
    if imgui.small_button(tr("Browse") + "##" + name):
        value = dialog(kind) or value
    return value


def update_theme():
    theme = {"dark": "violet", "light": "light-grey", "classic": "navy"}[preferences.values["theme"]]
    cyui.apply(theme, "comfortable")
    imgui.get_style().font_scale_main = preferences.values["font_scale"]


class Studio:
    def __init__(self, runtime, actor, session):
        self.runtime, self.actor, self.session = runtime, actor, session
        self.config = read()
        self.forms = {op["id"]: {key: copy.deepcopy(spec["default"]) for key, spec in op["inputSchema"]["properties"].items()
                                 if "default" in spec} for op in runtime.descriptor["operations"]
                      if "properties" in op["inputSchema"]}
        self.schemas = {op["id"]: op["inputSchema"] for op in runtime.descriptor["operations"]}
        self.forms["train_lora"]["dataset"] = []
        self.forms["generate"]["prompt"] = ""
        self.forms["run_workflow"]["file"] = ""
        self.forms["download_model"].update(category="checkpoints", filename="", url="")
        self.forms["import_model"].update(category="loras", filename="", file="")
        self.profile = self.config["profile"]
        self.operation = "generate"
        self.active = None
        self.result_job = None
        self.results = []
        self.result_index = -1
        self.inventory = {}
        self.inspection = {}
        self.source, self.preview = ImageCanvas(), ImageCanvas()
        self.canvas_output = False
        self.error = ""
        self.message = ""
        self.http = None
        self.frames = 0
        self.last_poll = 0.0
        self.root_draft = ""
        self.root_type = "library"
        self.advanced = Panel(runtime, actor)
        self.library = json.loads(Path(__file__).with_name("model_catalog.json").read_text(encoding="utf-8"))
        self.catalog_choice = -1
        self.force_tab = ""
        self.suggestions = []
        self.current_theme = None
        self.hf_token = ""
        self.server_token = {"local": "", "remote": ""}

    def capture(self):
        return {"operationId": self.operation, "parameters": self.payload(self.operation)}

    def restore(self, value):
        operation = value["operationId"]
        if operation not in ("generate", "run_workflow", "train_lora"):
            self.advanced.restore(value)
            return
        self.operation = operation
        properties = self.schemas[operation]["properties"]
        self.forms[operation] = {key: copy.deepcopy(spec["default"]) for key, spec in properties.items() if "default" in spec}
        self.forms[operation].update(value["parameters"])
        if value["parameters"].get("profile"):
            self.profile = value["parameters"]["profile"]
        self.force_tab = {"generate": "Studio", "run_workflow": "Workflows", "train_lora": "LoRA training"}[operation]
        if operation == "generate":
            file = value["parameters"].get("image")
            if file:
                self.source.load(file)
                if value["parameters"].get("mask"):
                    self.source.load_mask(value["parameters"]["mask"])
                self.canvas_output = False
            else:
                self.source.reset()

    def payload(self, operation):
        data = copy.deepcopy(self.forms[operation])
        if "profile" in self.schemas[operation].get("properties", {}):
            data["profile"] = self.profile
        if operation == "generate":
            if self.source.path:
                data["image"] = str(self.source.path)
            if data["mode"] == "inpaint":
                data["mask"] = self.source.save_mask() or data.get("mask", "")
        if operation == "run_workflow" and data.get("graph"):
            data.pop("file", None)
        return {key: value for key, value in data.items() if value != "" or key in ("prompt", "negative")}

    def submit(self, operation, data=None):
        if self.active:
            return
        try:
            params = self.payload(operation) if data is None else data
            job = self.runtime.submit(self.actor, self.session, operation, params)
            self.active = job
            self.last_poll = 0
            self.error = ""
            self.message = ""
            if preferences.values["autosave_generation"] and operation in ("generate", "train_lora", "run_workflow"):
                item = recipe(self.runtime.descriptor, operation, params, job_id=job["jobId"])
                save_recipe(DATA / "configurations/autosave" / (job["jobId"] + ".json"), item)
        except Exception as exc:
            self.error = str(exc)

    def poll(self):
        if not self.active or monotonic() - self.last_poll < 0.15:
            return
        self.last_poll = monotonic()
        self.active = self.runtime.get_job(self.actor, self.active["jobId"])
        if self.active["state"] not in TERMINAL:
            return
        job, self.active = self.active, None
        operation = job["operationId"]
        if job["state"] != "Completed":
            self.error = (job.get("error") or {}).get("message", job["state"])
            return
        result = job["outputs"]
        self.message = tr("Completed")
        if operation == "inspect_backend":
            profile = (job.get("parameters") or {}).get("profile", self.profile)
            self.inventory[profile] = result
        elif operation == "suggest_models":
            self.suggestions = result["candidates"]
            if not self.suggestions:
                self.message = tr("No matching model recipe on this server.")
        elif operation == "inspect_workflow":
            self.inspection = result
        elif operation in ("generate", "run_workflow", "train_lora"):
            self.result_job = job
            self.results = result["assets"]
            self.result_index = next((index for index, asset in enumerate(self.results) if asset["mimeType"].startswith("image/")), -1)
            if self.result_index >= 0:
                self.show_result(self.result_index)
                self.force_tab = "Studio"
        elif operation in ("import_model", "download_model"):
            self.message = result["file"]

    def show_result(self, index):
        self.result_index = index
        self.preview.load(self.results[index]["file"])
        self.canvas_output = True

    def show_job(self):
        if self.active:
            progress = self.active["progress"]
            imgui.progress_bar((progress.get("percent") or 0) / 100, (300, 0),
                               tr(progress.get("currentStep") or self.active["state"]))
            imgui.same_line()
            if imgui.small_button(label("Cancel")):
                self.runtime.cancel(self.actor, self.active["jobId"])
        if self.message:
            cyui.text_dim(self.message)
        if self.error:
            imgui.push_style_color(imgui.Col_.text, cyui.vec(cyui.palette().error))
            imgui.text_wrapped(self.error)
            imgui.pop_style_color()

    def options(self, category):
        return [entry["name"] for entry in self.inventory.get(self.profile, {}).get("models", {}).get(category, [])]

    def named(self, name, current, category):
        options = self.options(category)
        if not options:
            return imgui.input_text(label(name), current)[1]
        return choose(name, current, ["", *options], [tr("Select a file"), *options])

    def models_form(self, form):
        previous = form["family"]
        form["family"] = choose("Model family", previous, list(FAMILIES), list(FAMILIES.values()))
        if form["family"] != previous and form is self.forms["generate"]:
            if form["family"] in ("qwen", "qwen_edit", "flux", "flux2"):
                form["cfg"] = 1.0 if form["family"] in ("flux", "flux2") else 2.5
            if form["family"] == "qwen_edit":
                form["mode"] = "img2img"
        fields = ("checkpoint",) if form["family"] == "checkpoint" else ("diffusion", "encoder", "vae")
        if form["family"] == "flux":
            fields += ("encoder2",)
        for field in fields:
            form[field] = self.named(FIELD_LABELS[field], form.get(field, ""), MODEL_FILES[field])

    def field(self, op, name):
        form = self.forms[op]
        spec = self.schemas[op]["properties"][name]
        value = form.get(name, spec.get("default", ""))
        title = FIELD_LABELS.get(name, name)
        if name in ("sampler", "scheduler"):
            entries = self.inventory.get(self.profile, {}).get(name + "s", [])
            if entries:
                form[name] = choose(title, value, entries, entries)
                return
        if "enum" in spec:
            form[name] = choose(title, value, spec["enum"])
        elif spec["type"] == "integer" and name != "seed":
            form[name] = imgui.input_int(label(title), value)[1]
        elif spec["type"] == "number":
            form[name] = imgui.input_float(label(title), value, format="%.6g")[1]
        elif spec["type"] == "boolean":
            form[name] = imgui.checkbox(label(title), value)[1]
        else:
            changed, entry = imgui.input_text(label(title), str(value))
            if changed:
                try:
                    form[name] = int(entry) if spec["type"] == "integer" else entry
                except ValueError:
                    form[name] = entry

    def canvas_panel(self):
        source = self.source
        mode = self.forms["generate"]["mode"]
        if imgui.small_button(label("Open image")):
            path = dialog("askopenfilename")
            if path:
                source.load(path)
                self.forms["generate"].pop("mask", None)
                self.canvas_output = False
        imgui.same_line()
        if imgui.small_button(label("Fit view")):
            (self.preview if self.canvas_output else source).fit = True
        if self.results:
            imgui.same_line()
            pictures = [index for index, asset in enumerate(self.results) if asset["mimeType"].startswith("image/")]
            if pictures:
                before = self.result_index if self.canvas_output else -1
                index = choose("Canvas view", before, [-1, *pictures],
                               [tr("Source image"), *(self.results[i]["name"] for i in pictures)])
                if index != before:
                    if index >= 0:
                        self.show_result(index)
                    else:
                        self.canvas_output = False
        if self.canvas_output and self.result_index >= 0:
            asset = self.results[self.result_index]
            if imgui.small_button(label("Use as source")):
                source.load(asset["file"])
                self.forms["generate"].pop("mask", None)
                self.canvas_output = False
                if self.forms["generate"]["mode"] == "txt2img":
                    self.forms["generate"]["mode"] = "img2img"
            imgui.same_line()
            if imgui.small_button(label("Save image")):
                filename = dialog("asksaveasfilename", initialfile=asset["name"],
                                  defaultextension=Path(asset["name"]).suffix)
                if filename:
                    shutil.copyfile(asset["file"], filename)
            imgui.same_line()
            imgui.text(f"{asset['width']} × {asset['height']}")
        paint = not self.canvas_output and mode == "inpaint" and source.path is not None
        if paint:
            if imgui.small_button(label("Load mask")):
                mask = dialog("askopenfilename")
                if mask:
                    source.load_mask(mask)
            imgui.same_line()
            if imgui.small_button(label("Undo mask")):
                source.undo_mask()
                self.forms["generate"].pop("mask", None)
            imgui.same_line()
            if imgui.small_button(label("Clear mask")):
                source.clear()
                self.forms["generate"].pop("mask", None)
            imgui.same_line()
            _, source.erase = imgui.checkbox(label("Erase"), source.erase)
            imgui.set_next_item_width(145)
            _, source.brush = imgui.slider_int(label("Brush size"), source.brush, 3, 250)
        cyui.text_dim(tr("Wheel to zoom · Middle drag or Space + drag to pan"))
        view = self.preview if self.canvas_output else source
        view.draw(paint=paint, empty=tr("Open an image or generate one to begin"))

    def rows(self, key, label_singular, limit, add, draw):
        items = self.forms["generate"][key]
        if not imgui.collapsing_header(label(label_singular + " (" + str(len(items)) + ")")):
            return
        if imgui.small_button(label("Add " + label_singular)) and len(items) < limit:
            items.append(add())
        removed = None
        for index, item in enumerate(items):
            with cyui.scoped_id(index):
                imgui.separator()
                if imgui.small_button(label("Remove")):
                    removed = index
                draw(item)
        if removed is not None:
            items.pop(removed)

    def lora_row(self, item):
        item["name"] = self.named("LoRA file", item["name"], "loras")
        _, item["model_strength"] = imgui.slider_float(label("Model strength"), item["model_strength"], -2, 2)
        _, item["clip_strength"] = imgui.slider_float(label("Encoder strength"), item["clip_strength"], -2, 2)

    def control_row(self, item):
        item["model"] = self.named("ControlNet model", item["model"], "controlnet")
        item["image"] = path_field("Control image", item["image"])
        if self.source.path and imgui.small_button(label("Use canvas source")):
            item["image"] = str(self.source.path)
        item["preprocess"] = choose("Preprocessor", item["preprocess"], ["none", "canny"], [tr("Prepared map"), "Canny"])
        for key, title, bound in (("strength", "Strength", 2), ("start", "Start", 1), ("end", "End", 1)):
            _, item[key] = imgui.slider_float(label(title), item[key], 0, bound)
        if item["preprocess"] == "canny":
            _, item["low"] = imgui.slider_float(label("Canny low"), item["low"], 0, 1)
            _, item["high"] = imgui.slider_float(label("Canny high"), item["high"], 0, 1)

    def generation(self):
        self.operation = "generate"
        width = imgui.get_content_region_avail().x
        inspector = min(max(345, width * 0.31), max(345, width - 250))
        imgui.begin_child("canvas", (max(160, width - inspector - 10), 0))
        try:
            self.canvas_panel()
        finally:
            imgui.end_child()
        imgui.same_line()
        imgui.begin_child("generation", (0, 0), imgui.ChildFlags_.borders)
        try:
            self.generation_inspector()
        finally:
            imgui.end_child()

    def generation_inspector(self):
        p = self.forms["generate"]
        cyui.section_title(tr("Generate image"))
        p["mode"] = choose("Task", p["mode"], list(TASKS), [tr(value) for value in TASKS.values()])
        self.models_form(p)
        with cyui.scoped_disabled(bool(self.active)):
            if imgui.small_button(label("Suggest installed models")):
                self.submit("suggest_models", {"profile": self.profile, "mode": p["mode"]})
        candidates = [item for item in self.suggestions if item["parameters"]["mode"] == p["mode"]]
        if candidates:
            selected = choose("Suggested recipe", -1, [-1, *range(len(candidates))],
                              [tr("Select a recipe"), *(item["name"] for item in candidates)])
            if selected >= 0:
                p.update(candidates[selected]["parameters"])
        imgui.separator()
        imgui.text(tr("Prompt"))
        _, p["prompt"] = imgui.input_text_multiline("##prompt", p["prompt"], (-1, 100))
        imgui.text(tr("Negative prompt"))
        _, p["negative"] = imgui.input_text_multiline("##negative", p["negative"], (-1, 56))
        cyui.text_dim(f"{p['width']} × {p['height']}   ·   {p['steps']} {tr('Steps').lower()}")
        with cyui.scoped_disabled(bool(self.active)):
            if cyui.primary_button(label("Generate"), (-1, 0)):
                self.submit("generate")
        if imgui.collapsing_header(label("Sampling")):
            if self.source.path and imgui.small_button(label("Use source dimensions")):
                stride = 16 if p["family"] == "flux2" else 8
                p["width"], p["height"] = [min(4096, max(64, round(side / stride) * stride)) for side in self.source.size]
            for key in ("width", "height", "batch", "steps", "cfg", "seed", "sampler", "scheduler"):
                self.field("generate", key)
            if p["mode"] != "txt2img":
                self.field("generate", "denoise")
            if p["mode"] == "inpaint":
                self.field("generate", "inpaint_method")
            if p["family"] == "flux":
                self.field("generate", "guidance")
            if p["family"] != "checkpoint":
                self.field("generate", "dtype")
                self.field("generate", "encoder_device")
            self.field("generate", "tiled_vae")
            if p["tiled_vae"]:
                self.field("generate", "tile_size")
                self.field("generate", "tile_overlap")
        self.rows("loras", "LoRA", 12, lambda: {"name": "", "model_strength": 1., "clip_strength": 1.}, self.lora_row)
        self.rows("controls", "ControlNet", 8, lambda: {"model": "", "image": "", "preprocess": "none",
                  "strength": 1., "start": 0., "end": 1., "low": 0.4, "high": 0.8}, self.control_row)

    def workflows(self):
        self.operation = "run_workflow"
        p = self.forms["run_workflow"]
        cyui.section_title(tr("ComfyUI workflow"))
        previous = p.get("file", "")
        p["file"] = path_field("Workflow JSON or PNG", previous)
        if previous != p["file"]:
            self.inspection = {}
            p["overrides"], p["uploads"] = [], []
            p.pop("graph", None)
        with cyui.scoped_disabled(bool(self.active)):
            if imgui.button(label("Inspect workflow")):
                data = {key: val for key, val in self.payload(self.operation).items() if key in ("profile", "file", "graph")}
                self.submit("inspect_workflow", data)
            imgui.same_line()
            if cyui.primary_button(label("Run workflow")):
                self.submit("run_workflow")
        if self.inspection.get("missingNodes"):
            imgui.text_wrapped(tr("Missing nodes") + " · " + ", ".join(self.inspection["missingNodes"]))
        fields = self.inspection.get("editableInputs", [])
        overrides = {(item["node"], item["input"]): item["value"] for item in p["overrides"]}
        uploads = {(item["node"], item["input"]): item["file"] for item in p["uploads"]}
        for item in fields:
            node, field, starting = item["node"], item["input"], item["value"]
            key = (node, field)
            with cyui.scoped_id(node + "." + field):
                name = f"{node} · {item['type']} / {field}"
                current = overrides.get(key, starting)
                if field == "image" and item["type"] in ("LoadImage", "LoadImageMask"):
                    original = uploads.get(key, "")
                    path = path_field("Upload local image", original)
                    if path:
                        uploads[key] = path
                    elif original:
                        uploads.pop(key, None)
                if isinstance(starting, bool):
                    changed, current = imgui.checkbox(name, current)
                elif isinstance(starting, float):
                    changed, current = imgui.input_float(name, current)
                else:
                    changed, entry = imgui.input_text(name, str(current))
                    if changed:
                        try:
                            current = int(entry) if isinstance(starting, int) else entry
                        except ValueError:
                            changed = False
                if changed:
                    if current == starting:
                        overrides.pop(key, None)
                    else:
                        overrides[key] = current
        if fields:
            p["overrides"] = [{"node": node, "input": field, "value": value} for (node, field), value in overrides.items()]
            p["uploads"] = [{"node": node, "input": field, "file": file} for (node, field), file in uploads.items()]

    def model_library(self):
        cyui.section_title(tr("Available models"))
        current = self.inventory.get(self.profile, {})
        if not current:
            cyui.text_dim(tr("Refresh the selected server to list its models."))
        for category, entries in current.get("models", {}).items():
            if imgui.collapsing_header(f"{category}  ·  {len(entries)}"):
                for model in entries:
                    imgui.text(model["name"])
        imgui.separator()
        cyui.section_title(tr("Download weights"))
        p = self.forms["download_model"]
        before = self.catalog_choice
        self.catalog_choice = choose("Catalog", before, [-1, *range(len(self.library))],
                                     [tr("Custom URL"), *(item["name"] for item in self.library)])
        selection = self.library[self.catalog_choice] if self.catalog_choice >= 0 else None
        if before != self.catalog_choice:
            p.clear()
            p.update(selection["download"] if selection else {"category": "checkpoints", "filename": "", "url": ""})
            p["accept_license"] = False
        if selection:
            cyui.text_dim(f"{selection['license']}  ·  {selection['estimated']}")
            if imgui.small_button(label("Open model card and license")):
                webbrowser.open(selection["source"])
        else:
            p["category"] = choose("Model category", p["category"], [key for key in CATEGORIES if key != "diffusers"])
            _, p["filename"] = imgui.input_text(label("File name"), p["filename"])
            _, p["url"] = imgui.input_text(label("Direct HTTPS URL"), p["url"])
            _, p["sha256"] = imgui.input_text(label("Expected SHA256"), p.get("sha256", ""))
        _, p["accept_license"] = imgui.checkbox(label("I accept the provider terms"), p["accept_license"])
        if imgui.collapsing_header(label("Hugging Face access")):
            _, self.hf_token = imgui.input_text(label("Hugging Face token"), self.hf_token, flags=imgui.InputTextFlags_.password)
            if imgui.small_button(label("Use token for this session")):
                os.environ["IMAGESTUDIO_HF_TOKEN"] = self.hf_token.strip()
                self.hf_token = ""
            imgui.same_line()
            if imgui.small_button(label("Open Hugging Face token settings")):
                webbrowser.open("https://huggingface.co/settings/tokens")
        with cyui.scoped_disabled(not p["accept_license"] or bool(self.active)):
            if cyui.primary_button(label("Download / resume")):
                self.submit("download_model")
        cyui.text_dim(tr("Downloads go to the writable library. External servers manage their own model files."))
        if imgui.collapsing_header(label("Install a local model file")):
            self.model_import()

    def model_import(self):
        p = self.forms["import_model"]
        p["category"] = choose("Model category", p["category"], [key for key in CATEGORIES if key != "diffusers"])
        old = p.get("file", "")
        p["file"] = path_field("Model file", old)
        if old != p["file"] and p["file"]:
            p["filename"] = Path(p["file"]).name
        _, p["filename"] = imgui.input_text(label("File name"), p["filename"])
        with cyui.scoped_disabled(not p["file"] or bool(self.active)):
            if imgui.button(label("Install in library")):
                self.submit("import_model")

    def training(self):
        self.operation = "train_lora"
        p = self.forms["train_lora"]
        cyui.section_title(tr("Train a LoRA"))
        self.models_form(p)
        cyui.text_dim(tr("Guided training uses an ImageStudio-launched engine and native ComfyUI training nodes."))
        for key in ("steps", "rank", "learning_rate", "batch_size", "grad_accumulation_steps", "optimizer",
                    "training_dtype", "seed", "gradient_checkpointing", "offloading", "existing_lora"):
            self.field("train_lora", key)
        if imgui.collapsing_header(label("Training memory and precision")):
            for key in ("checkpoint_depth", "lora_dtype", "loss_function", "bypass_mode", "quantized_backward"):
                self.field("train_lora", key)
        if imgui.button(label("Add training images")):
            files = dialog("askopenfilenames", filetypes=[(tr("Images"), "*.png *.jpg *.jpeg *.webp")])
            for file in files[:max(0, 256 - len(p["dataset"]))]:
                captions = Path(file).with_suffix(".txt")
                caption = ""
                if captions.is_file():
                    with captions.open(encoding="utf-8-sig") as text:
                        caption = text.read(2000)
                p["dataset"].append({"image": file, "caption": caption})
        removed = None
        for index, row in enumerate(p["dataset"]):
            with cyui.scoped_id(index):
                imgui.text(Path(row["image"]).name)
                imgui.same_line()
                if imgui.small_button(label("Remove")):
                    removed = index
                _, row["caption"] = imgui.input_text(label("Caption"), row["caption"])
        if removed is not None:
            p["dataset"].pop(removed)
        ready = self.profile in ("managed", "existing") and bool(p["dataset"])
        with cyui.scoped_disabled(not ready or bool(self.active)):
            if cyui.primary_button(label("Train LoRA")):
                self.submit("train_lora")
        if self.result_job and self.result_job["operationId"] == "train_lora":
            for index, asset in enumerate(self.results):
                if not asset["file"].endswith(".safetensors"):
                    continue
                with cyui.scoped_id(index):
                    imgui.text(asset["name"])
                    imgui.same_line()
                    if imgui.small_button(label("Add trained LoRA to library")):
                        self.submit("import_model", {"file": asset["file"], "category": "loras", "filename": asset["name"]})
            self.export()

    def export(self):
        if self.result_job and imgui.small_button(label("Export job files")):
            target = dialog("askdirectory")
            if target:
                result = self.runtime.export_job(self.actor, self.result_job["jobId"], target)
                self.message = json.dumps(result, ensure_ascii=False)

    def settings_panel(self):
        config = self.config
        cyui.section_title(tr("ComfyUI connections"))
        config["profile"] = choose("Default backend", config["profile"], list(PROFILE_NAMES),
                                   [tr(value) for value in PROFILE_NAMES.values()])
        if imgui.collapsing_header(label("Installed ComfyUI"), imgui.TreeNodeFlags_.default_open):
            config["main"] = path_field("ComfyUI main.py", config["main"])
            config["python"] = path_field("ComfyUI Python", config["python"])
            _, config["custom_nodes"] = imgui.checkbox(label("Load trusted custom nodes"), config["custom_nodes"])
        for key in ("local", "remote"):
            with cyui.scoped_id(key):
                if imgui.collapsing_header(label(PROFILE_NAMES[key])):
                    row = config["profiles"][key]
                    _, row["url"] = imgui.input_text(label("ComfyUI base URL"), row["url"])
                    _, row["targeted_interrupt"] = imgui.checkbox(label("Server supports prompt-specific cancellation"), row["targeted_interrupt"])
                    _, self.server_token[key] = imgui.input_text(label("Server bearer token"), self.server_token[key],
                                                               flags=imgui.InputTextFlags_.password)
                    if imgui.small_button(label("Use token for this session")):
                        os.environ["IMAGESTUDIO_COMFY_TOKEN_" + key.upper()] = self.server_token[key].strip()
                        self.server_token[key] = ""
        if imgui.collapsing_header(label("ImageStudio installation")):
            self.field("install_engine", "compute")
            with cyui.scoped_disabled(bool(self.active)):
                if imgui.button(label("Install / resume ComfyUI")):
                    self.submit("install_engine")
        cyui.section_title(tr("Model search folders"))
        config["models"] = path_field("Writable library", config["models"], kind="askdirectory")
        self.root_type = choose("Folder type", self.root_type, ["library", *CATEGORIES],
                                [tr("ComfyUI models library"), *CATEGORIES])
        self.root_draft = path_field("Existing folder", self.root_draft, kind="askdirectory")
        if imgui.small_button(label("Add search folder")):
            directory = Path(self.root_draft).expanduser()
            if not self.root_draft or not directory.is_dir():
                self.error = tr("Choose an existing folder.")
            else:
                record = {"path": str(directory.resolve()), "category": self.root_type}
                if record not in config["roots"]:
                    config["roots"].append(record)
                self.root_draft = ""
        removed = None
        for index, root in enumerate(config["roots"]):
            with cyui.scoped_id(index):
                if imgui.small_button(label("Remove")):
                    removed = index
                imgui.same_line()
                imgui.text_wrapped(root["category"] + " · " + root["path"])
        if removed is not None:
            config["roots"].pop(removed)
        cyui.text_dim(tr("Search folders are referenced in place by engines launched here."))
        cyui.section_title(tr("RAM and VRAM"))
        config["memory"] = choose("Memory profile", config["memory"], ["balanced", "low", "minimum", "cpu"],
                                  [tr("Balanced"), tr("Low VRAM"), tr("Minimum VRAM"), "CPU"])
        if config["memory"] != "cpu":
            _, config["reserve"] = imgui.slider_float(label("Reserved VRAM (GiB)"), config["reserve"], 0, 12)
            _, config["device"] = imgui.input_text(label("CUDA device index (blank = default)"), config["device"])
        _, config["release"] = imgui.checkbox(label("Unload weights after every job"), config["release"])
        with cyui.scoped_disabled(bool(self.active)):
            if imgui.button(label("Save connection and model settings")):
                for row in config["profiles"].values():
                    if row["url"]:
                        row["url"] = endpoint(row["url"])
                if config["device"].strip() and not config["device"].strip().isdigit():
                    raise ValueError(tr("CUDA device index must be a nonnegative integer."))
                if not config["models"].strip() or not Path(config["models"]).expanduser().is_absolute():
                    raise ValueError(tr("Choose an absolute writable model directory."))
                save(config)
                self.profile = config["profile"]
                self.inventory.clear()
                self.message = tr("Settings saved")
            imgui.same_line()
            if imgui.button(label("Release memory")):
                self.submit("release_memory", {"profile": self.profile})
            if self.profile in ("managed", "existing"):
                imgui.same_line()
                if imgui.button(label("Stop owned engine")):
                    self.submit("release_memory", {"profile": self.profile, "stop": True})

    def mcp(self):
        cyui.section_title("MCP")
        if not self.http:
            if imgui.button(label("Start local MCP service")):
                self.http = make_server(self.runtime, access_profile="free")
        else:
            imgui.text(self.http.mcp_url)
            imgui.same_line()
            if imgui.button(label("Copy MCP URL")):
                imgui.set_clipboard_text(self.http.mcp_url)
            imgui.same_line()
            if imgui.button(label("Stop MCP service")):
                self.http.server_close()
                self.http = None

    def preferences_panel(self):
        self.settings_panel()
        imgui.separator()
        self.mcp()
        imgui.separator()
        if imgui.collapsing_header(label("Language, theme, recipes and licenses"), imgui.TreeNodeFlags_.default_open):
            options_panel()
        from updates import draw
        draw(self)

    def tab(self, title, draw):
        desired = title == self.force_tab
        opened, _ = imgui.begin_tab_item(label(title), flags=imgui.TabItemFlags_.set_selected if desired else 0)
        if opened:
            if desired:
                self.force_tab = ""
            try:
                draw()
            except (OSError, ValueError, RuntimeError) as error:
                self.error = str(error)
            finally:
                imgui.end_tab_item()

    def draw(self):
        self.frames += 1
        if self.frames == 1:
            self.force_tab = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--tab=")), "")
            sample = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--preview=")), "")
            if sample:
                self.source.load(sample)
        journal.poll()
        self.poll()
        theme = (preferences.values["theme"], preferences.values["font_scale"])
        if self.current_theme != theme:
            update_theme()
            self.current_theme = theme
        cyui.heading("CyTools ImageStudio", level=1)
        imgui.same_line()
        imgui.set_next_item_width(190)
        old = self.profile
        self.profile = choose("Backend", old, list(PROFILE_NAMES), [tr(value) for value in PROFILE_NAMES.values()])
        if old != self.profile:
            self.suggestions.clear()
        imgui.same_line()
        with cyui.scoped_disabled(bool(self.active)):
            if imgui.small_button(label("Refresh models")):
                self.submit("inspect_backend")
        data = self.inventory.get(self.profile, {})
        if data.get("devices"):
            imgui.same_line()
            device = data["devices"][0]
            ram = data.get("system", {}).get("ram_free", 0) / 1024**3
            usage = f"{ram:.1f} GiB RAM"
            if device.get("type") != "cpu":
                vram = device.get("vram_free", 0) / 1024**3
                usage = f"{vram:.1f} GiB VRAM  ·  {usage}"
            cyui.text_dim(f"{device.get('name', 'ComfyUI')}  |  {usage}")
        self.show_job()
        if self.result_job:
            self.export()
        if imgui.begin_tab_bar("##studio-tabs"):
            self.tab("Studio", self.generation)
            self.tab("Workflows", self.workflows)
            self.tab("Models", self.model_library)
            self.tab("LoRA training", self.training)
            self.tab("Options", self.preferences_panel)
            self.advanced.draw_tab()
            journal.draw_tab()
            imgui.end_tab_bar()

    def close(self):
        if self.http:
            self.http.server_close()
        self.source.close()
        self.preview.close()
        self.advanced.close()


def main():
    DATA.joinpath("settings").mkdir(parents=True, exist_ok=True)
    runner = hello_imgui.RunnerParams()
    runner.app_window_params.window_title = "CyTools_ImageStudio"
    runner.app_window_params.window_geometry.size = (1370, 850)
    runner.ini_filename = str(DATA / "settings/imgui.ini")
    runner.callbacks.load_additional_fonts = lambda: cyui.load_fonts(17, icons=False)
    from cytools_core import CyToolError
    runtime = create_runtime(DATA / "jobs")
    runtime.start()
    try:
        permissions = ("network", "filesystemExternal", "manageResources")
        actor = runtime.authenticate(runtime.create_client("ImageStudio desktop", permissions)["token"])
        session = runtime.create_session(actor, temporary=False)["sessionId"]
        studio = Studio(runtime, actor, session)
        journal.bind(runtime, actor)
        configure_generation(runtime, studio.capture, studio.restore)
        language = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--language=")), "")
        if language in ("en", "fr"):
            preferences.values["language"] = language

        def frame():
            studio.draw()
            if "--smoke-test" in sys.argv and studio.frames >= 12:
                runner.app_shall_exit = True

        runner.callbacks.show_gui = frame
        runner.callbacks.before_exit = studio.close
        hello_imgui.run(runner)
        if "--smoke-test" in sys.argv:
            from standalone_log import save_screenshot
            target = DATA / "reports" / (f"studio-{preferences.values['language']}.png")
            target.parent.mkdir(parents=True, exist_ok=True)
            save_screenshot(target, hello_imgui.final_app_window_screenshot())
    finally:
        while True:
            try:
                runtime.close(timeout=5)
                break
            except CyToolError as error:
                if error.code != "Busy":
                    raise
