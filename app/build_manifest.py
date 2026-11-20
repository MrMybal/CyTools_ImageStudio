"""Generate CyTool.json, the shared contract for desktop, CLI and native MCP."""
import json
from pathlib import Path
from settings import CATEGORIES


def obj(properties, required=(), **extra):
    return dict(type="object", properties=properties, required=list(required), additionalProperties=False, **extra)


def str_(description="", default=None, **extra):
    field = dict(type="string", description=description, **extra)
    if default is not None:
        field["default"] = default
    return field


def enum(values, default=None, description=""):
    return str_(description, default, enum=list(values))


def num(default, low, high, **extra):
    kind = "integer" if all(isinstance(value, int) for value in (default, low, high)) else "number"
    return dict(type=kind, default=default, minimum=low, maximum=high, **extra)


def flag(default):
    return dict(type="boolean", default=default)


def arr(item, maximum, *, minimum=0, default=False):
    field = dict(type="array", items=item, minItems=minimum, maxItems=maximum)
    if default:
        field["default"] = []
    return field


PROFILE = {"profile": enum(["managed", "existing", "local", "remote"], description="Configured ComfyUI profile. Omit for the studio default.")}
TIME = {"timeout": num(3600, 15, 86400, description="ComfyUI execution deadline in seconds.")}
FILE = str_("Path on the ImageStudio host, imported automatically.", minLength=1, **{"x-cy-kind": "File"})
IMAGE = {**FILE, "x-cy-kind": "Image"}
MODEL = dict(
    family=enum(["checkpoint", "flux", "flux2", "qwen", "qwen_edit", "sd3"], "checkpoint",
                "Select the graph architecture matching the weights."),
    checkpoint=str_("Exact checkpoint name from inspect_backend."),
    diffusion=str_("Exact diffusion model name from inspect_backend."),
    encoder=str_("Exact encoder. FLUX.1 takes CLIP-L here."),
    encoder2=str_("FLUX.1 T5 encoder."),
    vae=str_("Exact VAE name from inspect_backend."),
    dtype=enum(["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], "default"),
    encoder_device=enum(["default", "cpu"], "default"),
)
LORA = obj(dict(name=str_("Exact LoRA name from inspect_backend.", minLength=1),
                model_strength=num(1., -8., 8.), clip_strength=num(1., -8., 8.)), ["name"])
CONTROL = obj(dict(model=str_("Exact ControlNet name from inspect_backend.", minLength=1), image=IMAGE,
                   strength=num(1., 0., 10.), start=num(0., 0., 1.), end=num(1., 0., 1.),
                   preprocess=enum(["none", "canny"], "none"), low=num(0.4, 0., 1.), high=num(0.8, 0., 1.)),
              ["model", "image"])
GENERATION = obj(dict(
    **PROFILE, **TIME, **MODEL,
    mode=enum(["txt2img", "img2img", "inpaint"], "txt2img"),
    prompt=str_("Positive prompt or Qwen Image Edit instruction.", maxLength=20000, **{"x-cy-kind": "MultilineString"}),
    negative=str_("Negative prompt.", "", maxLength=20000),
    image=IMAGE, mask={**IMAGE, "description": "Grayscale mask. White regions are repainted."},
    width=num(1024, 64, 4096, multipleOf=8), height=num(1024, 64, 4096, multipleOf=8),
    steps=num(25, 1, 150), cfg=num(7., 0., 30.), guidance=num(3.5, 0., 30.),
    seed=num(-1, -1, 9007199254740991, description="Use -1 for a random seed returned in the job outputs."),
    sampler=str_("KSampler sampler from inspect_backend.", "euler"),
    scheduler=str_("KSampler scheduler from inspect_backend.", "normal"),
    batch=num(1, 1, 8), denoise=num(0.65, 0., 1.),
    inpaint_method=enum(["noise_mask", "conditioning"], "noise_mask", "Use conditioning for dedicated inpaint models."),
    tiled_vae=flag(False), tile_size=num(512, 64, 4096, multipleOf=32),
    tile_overlap=num(64, 0, 1024, multipleOf=16),
    loras=arr(LORA, 12, default=True), controls=arr(CONTROL, 8, default=True),
), ["prompt"], allOf=[
    {"if": {"properties": {"mode": {"enum": ["img2img", "inpaint"]}}, "required": ["mode"]},
     "then": {"required": ["image"]}},
    {"if": {"properties": {"mode": {"const": "inpaint"}}, "required": ["mode"]},
     "then": {"required": ["mask"]}},
])
NODE = dict(node=str_("Node id.", minLength=1), input=str_("Node input name.", minLength=1))
SOURCE = dict(**PROFILE, file=FILE, graph={"type": "object", "description": "API prompt or editor graph."})
HAS_SOURCE = {"oneOf": [{"required": ["file"]}, {"required": ["graph"]}]}
WORKFLOW = obj(dict(**SOURCE, **TIME,
    overrides=arr(obj(dict(**NODE, value={}), ["node", "input", "value"]), 300, default=True),
    uploads=arr(obj(dict(**NODE, file=IMAGE), ["node", "input", "file"]), 48, default=True)), **HAS_SOURCE)
TRAIN = obj(dict(
    **PROFILE, **TIME, **MODEL,
    dataset=arr(obj(dict(image=IMAGE, caption=str_("Training caption.", "", maxLength=2000)), ["image"]), 256, minimum=1),
    steps=num(100, 1, 100000), rank=num(8, 1, 128), learning_rate=num(0.0005, 0.0000001, 1.),
    batch_size=num(1, 1, 32), grad_accumulation_steps=num(1, 1, 1024),
    optimizer=enum(["AdamW", "Adam", "SGD", "RMSprop"], "AdamW"), seed=num(0, 0, 9007199254740991),
    training_dtype=enum(["bf16", "fp32", "none"], "bf16"), gradient_checkpointing=flag(True),
    offloading=flag(False), existing_lora=str_("Optional adapter to continue.", "[None]"),
    checkpoint_depth=num(1, 1, 5), lora_dtype=enum(["bf16", "fp32"], "bf16"),
    loss_function=enum(["MSE", "L1", "Huber", "SmoothL1"], "MSE"),
    bypass_mode=flag(False), quantized_backward=flag(False),
), ["dataset"])
TARGET = dict(category=enum(key for key in CATEGORIES if key != "diffusers"),
              filename=str_("Destination basename with its model extension.", minLength=1, maxLength=141))
DOWNLOAD = obj(dict(**TARGET, url=str_("Direct HTTPS URL without credentials.", minLength=1),
                    sha256=str_("Provider SHA256.", pattern="^[0-9a-fA-F]{64}$"),
                    size={"type": "integer", "minimum": 1}, revision=str_("Provider version or commit."),
                    accept_license=flag(False)), ["category", "filename", "url"])
ASSET = obj(dict(file=str_("Job-owned absolute path."), name=str_(), node=str_(), mimeType=str_(),
                 width={"type": "integer"}, height={"type": "integer"}), ["file", "name", "node", "mimeType"])
RESULT = obj(dict(promptId=str_(), workflow=str_("Effective API graph."), history=str_("ComfyUI history."),
                  assets=arr(ASSET, 1024), seed={"type": "integer"}), ["promptId", "workflow", "history", "assets"])
MODEL_RESULT = obj(dict(file=str_("Persistent model path."), category=str_(), bytes={"type": "integer"},
                        sha256Verified={"type": "boolean"}), ["file", "category", "bytes", "sha256Verified"])
OPEN = {"type": "object"}
NET = ("network",)
FS = ("filesystemExternal",)


def operation(key, title, description, inputs, outputs, permissions=(), progress=False):
    return dict(id=key, name=title, description=description, inputSchema=inputs, outputSchema=outputs,
                supportedBackends=[], permissions=list(permissions), resourceRequirements={"ramBytes": 32 * 1024 * 1024},
                canRunAsync=True, canCancel=True, supportsProgress=progress)


def build():
    path = Path(__file__).with_name("CyTool.json")
    data = dict(schemaVersion=1, protocolVersion="1.0", id="cy.tools.imagestudio",
                name="CyTools_ImageStudio", version="0.1.0", vendor="Cyberalien", author="Cyberalien",
                description="AI image generation and editing with local or remote ComfyUI, an inpainting canvas, ControlNet, LoRA training, and MCP automation.",
                license="AGPL-3.0-only", licenseUrl="https://spdx.org/licenses/AGPL-3.0-only.html",
                repository="https://github.com/MrMybal/CyTools_ImageStudio",
                executable="runtime/python/Scripts/python.exe app/tool.py", standaloneExecutable="CyTools_ImageStudio.exe",
                categories=["image"], tags=["ComfyUI", "image-generation", "ControlNet", "LoRA"],
                capabilities=["jobs", "cancellation", "text-to-image", "image-to-image", "inpainting", "workflow", "lora-training"],
                backends=[], interfaces=["ImGui", "CLI", "JSONL", "HTTP", "MCP"],
                platforms=[dict(os=os, architecture="x64", status="Experimental", computeBackends=["CPU", "CUDA"])
                           for os in ("Windows", "Linux")],
                runtimeRequirements=[], concurrency=dict(policy="Sequential", maxWorkers=1))
    data["operations"] = [
        operation("studio_status", "Studio status", "Read local backend configuration and model search paths.", obj({}), OPEN),
        operation("inspect_backend", "Inspect ComfyUI", "List installed models, loaders, supported nodes, samplers and device memory. Call before choosing a generation model.",
                  obj(dict(**PROFILE, include_nodes=flag(False))), OPEN, NET),
        operation("suggest_models", "Suggest image models", "Return named recipes with all required components visible on the selected ComfyUI. File names and loaders indicate compatibility, which the model loader verifies at execution.",
                  obj(dict(**PROFILE, mode=enum(["txt2img", "img2img", "inpaint"], "txt2img"),
                           family=enum(["auto", *MODEL["family"]["enum"]], "auto"))),
                  obj({"candidates": arr(obj(dict(name=str_(), parameters=OPEN, basis=str_()),
                                             ["name", "parameters", "basis"]), 10000)}, ["candidates"]), NET),
        operation("generate", "Generate image", "Run text-to-image, image-to-image or inpainting with LoRA and ControlNet. Use the family and exact installed components from inspect_backend.",
                  GENERATION, RESULT, NET + FS, True),
        operation("inspect_workflow", "Inspect workflow", "Read node classes, missing nodes and widget values.", obj(SOURCE, **HAS_SOURCE), OPEN, NET + FS),
        operation("run_workflow", "Run workflow", "Submit a ComfyUI workflow with input overrides or automatic image uploads.", WORKFLOW, RESULT, NET + FS, True),
        operation("train_lora", "Train LoRA", "Train from captioned images using native nodes on an ImageStudio-launched ComfyUI.", TRAIN, RESULT, NET + FS, True),
        operation("download_model", "Download model", "Download or resume model weights after reviewing their license.", DOWNLOAD, MODEL_RESULT, NET + FS, True),
        operation("import_model", "Install local model", "Install a file such as a locally trained LoRA into the writable library.",
                  obj(dict(**TARGET, file=FILE), ["category", "filename", "file"]), MODEL_RESULT, FS),
        operation("release_memory", "Release memory", "Request cache release, or stop the ImageStudio-launched process.",
                  obj(dict(**PROFILE, stop=flag(False))), OPEN, NET + ("manageResources",)),
        operation("install_engine", "Install ComfyUI", "Provision the pinned ComfyUI and its private PyTorch without model weights.",
                  obj(dict(compute=enum(["cuda", "cpu"], "cuda"))), OPEN, NET + FS, True),
    ]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
