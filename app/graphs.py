"""ComfyUI graph construction and import without model libraries."""
import copy
import json
from pathlib import Path
from secrets import randbits
from cytools_core import CyToolError

MODEL_INPUTS = {
    "checkpoints": [("CheckpointLoaderSimple", "ckpt_name")],
    "diffusion_models": [("UNETLoader", "unet_name"), ("UnetLoaderGGUF", "unet_name")],
    "text_encoders": [("CLIPLoader", "clip_name"), ("CLIPLoaderGGUF", "clip_name")],
    "vae": [("VAELoader", "vae_name")],
    "loras": [("LoraLoader", "lora_name")],
    "controlnet": [("ControlNetLoader", "control_net_name")],
    "upscale_models": [("UpscaleModelLoader", "model_name")],
}


def inputs(info, node):
    groups = info.get(node, {}).get("input", {})
    return {**groups.get("required", {}), **groups.get("optional", {})}


def choices(info, node, key):
    spec = inputs(info, node).get(key, [])
    if not spec:
        return []
    return spec[0] if isinstance(spec[0], list) else spec[1].get("options", []) if len(spec) > 1 else []


def catalog(info):
    result = {}
    for category, loaders in MODEL_INPUTS.items():
        values = {}
        for node, key in loaders:
            for name in choices(info, node, key):
                values.setdefault(name, []).append(node)
        result[category] = [{"name": name, "loaders": nodes} for name, nodes in sorted(values.items())]
    return result


class Graph:
    def __init__(self):
        self.nodes = {}

    def add(self, kind, **values):
        key = str(len(self.nodes) + 1)
        self.nodes[key] = {"class_type": kind, "inputs": values}
        return [key, 0]


def slots(link, count):
    return tuple([link[0], index] for index in range(count))


def models(graph, p):
    family = p["family"]
    if family == "checkpoint":
        if not p.get("checkpoint"):
            raise CyToolError("ModelRequired", "Select an installed ComfyUI checkpoint.", recoverable=True)
        return slots(graph.add("CheckpointLoaderSimple", ckpt_name=p["checkpoint"]), 3)

    if not all(p.get(key) for key in ("diffusion", "encoder", "vae")):
        raise CyToolError("ModelRequired", "Select the diffusion model, text encoder and VAE for this architecture.")
    dtype = p.get("dtype", "default")
    model = graph.add("UNETLoader", unet_name=p["diffusion"], weight_dtype=dtype)
    if family == "flux":
        if not p.get("encoder2"):
            raise CyToolError("ModelRequired", "FLUX.1 requires both CLIP-L and T5 text encoders.")
        clip = graph.add("DualCLIPLoader", clip_name1=p["encoder"], clip_name2=p["encoder2"], type="flux",
                         device=p.get("encoder_device", "default"))
    else:
        clip = graph.add("CLIPLoader", clip_name=p["encoder"],
                         type={"qwen": "qwen_image", "qwen_edit": "qwen_image", "flux2": "flux2", "sd3": "sd3"}[family],
                         device=p.get("encoder_device", "default"))
    return model, clip, graph.add("VAELoader", vae_name=p["vae"])


def generate(p, info, uploaded, prefix):
    graph = Graph()
    model, clip, vae = models(graph, p)
    for lora in p["loras"]:
        model, clip = slots(graph.add("LoraLoader", model=model, clip=clip, lora_name=lora["name"],
                                      strength_model=lora["model_strength"], strength_clip=lora["clip_strength"]), 2)
    family, mode = p["family"], p["mode"]
    if family == "qwen_edit" and mode != "img2img":
        raise CyToolError("InvalidParameters", "Qwen Image Edit requires Image to image mode.")
    image = graph.add("LoadImage", image=uploaded["image"]) if "image" in uploaded else None
    if image and mode != "txt2img":
        image = graph.add("ImageScale", image=image, width=p["width"], height=p["height"],
                          upscale_method="lanczos", crop="disabled")
    if family == "qwen_edit":
        positive = graph.add("TextEncodeQwenImageEditPlus", clip=clip, prompt=p["prompt"], image1=image, vae=vae)
        negative = graph.add("TextEncodeQwenImageEditPlus", clip=clip, prompt=p["negative"], image1=image, vae=vae)
    else:
        positive = graph.add("CLIPTextEncode", clip=clip, text=p["prompt"])
        negative = graph.add("CLIPTextEncode", clip=clip, text=p["negative"])
    if family == "flux":
        positive = graph.add("FluxGuidance", conditioning=positive, guidance=p["guidance"])

    if mode == "txt2img" or family == "qwen_edit":
        latent_node = {"flux": "EmptySD3LatentImage", "flux2": "EmptyFlux2LatentImage", "qwen": "EmptySD3LatentImage",
                       "qwen_edit": "EmptySD3LatentImage", "sd3": "EmptySD3LatentImage"}.get(family, "EmptyLatentImage")
        latent = graph.add(latent_node, width=p["width"], height=p["height"], batch_size=p["batch"])
    else:
        latent = graph.add("VAEEncode", pixels=image, vae=vae)
        if mode == "inpaint":
            mask_image = graph.add("LoadImage", image=uploaded["mask"])
            mask_image = graph.add("ImageScale", image=mask_image, width=p["width"], height=p["height"],
                                   upscale_method="nearest-exact", crop="disabled")
            mask = graph.add("ImageToMask", image=mask_image, channel="red")
            if p["inpaint_method"] == "conditioning":
                positive, negative, latent = slots(graph.add("InpaintModelConditioning", positive=positive,
                    negative=negative, pixels=image, vae=vae, mask=mask, noise_mask=True), 3)
            else:
                latent = graph.add("SetLatentNoiseMask", samples=latent, mask=mask)
            if p["batch"] > 1:
                latent = graph.add("RepeatLatentBatch", samples=latent, amount=p["batch"])
        elif p["batch"] > 1:
            latent = graph.add("RepeatLatentBatch", samples=latent, amount=p["batch"])
    for index, control in enumerate(p["controls"]):
        if control["start"] > control["end"]:
            raise CyToolError("InvalidParameters", "ControlNet start must be at or before its end.")
        hint = graph.add("LoadImage", image=uploaded[f"control_{index}"])
        if control["preprocess"] == "canny":
            hint = graph.add("Canny", image=hint, low_threshold=control["low"], high_threshold=control["high"])
        net = graph.add("ControlNetLoader", control_net_name=control["model"])
        values = {"vae": vae} if "vae" in inputs(info, "ControlNetApplyAdvanced") else {}
        positive, negative = slots(graph.add("ControlNetApplyAdvanced", positive=positive, negative=negative,
            image=hint, control_net=net, strength=control["strength"], start_percent=control["start"],
            end_percent=control["end"], **values), 2)

    seed = randbits(53) if p["seed"] < 0 else p["seed"]
    sampled = graph.add("KSampler", seed=seed, steps=p["steps"], cfg=p["cfg"], sampler_name=p["sampler"],
                        scheduler=p["scheduler"], model=model, positive=positive, negative=negative,
                        latent_image=latent, denoise=1.0 if mode == "txt2img" or family == "qwen_edit" else p["denoise"])
    tiled = p["tiled_vae"]
    image = graph.add("VAEDecodeTiled" if tiled else "VAEDecode", samples=sampled, vae=vae,
                      **({"tile_size": p["tile_size"], "overlap": p["tile_overlap"]} if tiled else {}))
    graph.add("SaveImage", images=image, filename_prefix=prefix)
    return graph.nodes, seed


def train(p, info, folder, prefix):
    graph = Graph()
    model, clip, vae = models(graph, p)
    images, texts = slots(graph.add("LoadImageTextDataSetFromFolder", folder=folder), 2)
    latent, conditioning = slots(graph.add("MakeTrainingDataset", images=images, texts=texts, vae=vae, clip=clip), 2)
    required = info.get("TrainLoraNode", {}).get("input", {}).get("required", {})
    args = {key: spec[1]["default"] for key, spec in required.items()
            if len(spec) > 1 and isinstance(spec[1], dict) and "default" in spec[1]}
    for key in ("batch_size", "grad_accumulation_steps", "steps", "learning_rate", "rank", "optimizer",
                "seed", "training_dtype", "gradient_checkpointing", "offloading", "existing_lora",
                "checkpoint_depth", "lora_dtype", "loss_function", "bypass_mode", "quantized_backward"):
        if key in inputs(info, "TrainLoraNode"):
            args[key] = p[key]
    result = graph.add("TrainLoraNode", model=model, latents=latent, positive=conditioning, **args)
    outputs = info.get("TrainLoraNode", {}).get("output", [])
    try:
        lora = [result[0], outputs.index("LORA_MODEL")]
    except ValueError as error:
        raise CyToolError("UnsupportedTraining", "TrainLoraNode on this server does not expose LORA_MODEL.") from error
    values = {"steps": [result[0], outputs.index("INT")]} if "INT" in outputs else {}
    graph.add("SaveLoRA", lora=lora, prefix=prefix, **values)
    return graph.nodes


def read_document(path):
    file = Path(path)
    if file.stat().st_size > 32 * 1024**2:
        raise CyToolError("InvalidWorkflow", "Workflow exceeds 32 MiB.")
    if file.suffix.lower() == ".png":
        from PIL import Image
        with Image.open(file) as image:
            text = image.info.get("prompt") or image.info.get("workflow")
        if not text:
            raise CyToolError("InvalidWorkflow", "This PNG has no embedded ComfyUI workflow.")
        return json.loads(text)
    return json.loads(file.read_text(encoding="utf-8-sig"))


def workflow(document, info):
    if not isinstance(document, dict):
        raise CyToolError("InvalidWorkflow", "Expected a ComfyUI JSON object.")
    if "nodes" in document:
        return convert_ui(document, info)
    if "prompt" in document:
        document = document["prompt"]
    if not isinstance(document, dict) or not document or not all(
            isinstance(node, dict) and isinstance(node.get("class_type"), str)
            and isinstance(node.get("inputs"), dict) for node in document.values()):
        raise CyToolError("InvalidWorkflow", "Expected a ComfyUI API prompt or editor graph.")
    return copy.deepcopy(document)


def convert_ui(document, info):
    """Convert standard graphs. The frontend API export is authoritative for extensions."""
    if document.get("definitions", {}).get("subgraphs"):
        raise CyToolError("ApiExportRequired", "Export this subgraph from ComfyUI using Save (API Format).")
    nodes = {str(node["id"]): node for node in document["nodes"]}
    raw_links = document.get("links", [])
    links = {link[0]: (str(link[1]), link[2]) for link in raw_links if isinstance(link, list) and len(link) >= 5}
    if len(links) != len(raw_links):
        raise CyToolError("ApiExportRequired", "Export this link format from ComfyUI using Save (API Format).")

    def resolve(link_id, source=None, seen=frozenset()):
        if link_id in seen or link_id not in links:
            raise CyToolError("InvalidWorkflow", "The editor graph contains an invalid link.")
        node_id, socket = links[link_id]
        node = nodes[node_id]
        if node["type"] == "PrimitiveNode":
            return node.get("widgets_values", [None])[0]
        if node["type"] == "Reroute":
            return resolve(node["inputs"][0]["link"], seen=seen | {link_id})
        if node.get("mode", 0) != 0:
            raise CyToolError("ApiExportRequired", "Connected bypassed or muted nodes require Save (API Format).")
        return [node_id, socket]

    result = {}
    for key, node in nodes.items():
        kind = node["type"]
        if kind in ("Reroute", "PrimitiveNode", "Note", "MarkdownNote") or node.get("mode", 0) != 0:
            continue
        if kind not in info:
            raise CyToolError("MissingNodes", f"Node {kind} is not installed on the selected server.")
        values = {entry["name"]: resolve(entry["link"]) for entry in node.get("inputs", [])
                  if entry.get("link") is not None}
        widgets = node.get("widgets_values", [])
        if isinstance(widgets, dict):
            values.update({name: value for name, value in widgets.items() if name not in values})
        else:
            remaining = iter(widgets)
            attached = {entry["name"] for entry in node.get("inputs", []) if entry.get("widget")}
            for name, spec in inputs(info, kind).items():
                kind_value = spec[0]
                if not (isinstance(kind_value, list) or kind_value in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")):
                    continue
                if len(spec) > 1 and spec[1].get("forceInput") and name not in attached:
                    continue
                if name in values and name not in attached:
                    continue
                sentinel = object()
                value = next(remaining, sentinel)
                if value is sentinel:
                    break
                if name not in values:
                    values[name] = value
                if name in ("seed", "noise_seed", "rand_seed") and len(spec) > 1 and spec[1].get("control_after_generate"):
                    next(remaining, None)
            if any(value not in ("fixed", "randomize", "increment", "decrement", "image") for value in remaining):
                raise CyToolError("ApiExportRequired", f"Node {kind} has frontend-only widgets. Use Save (API Format).")
        result[key] = {"class_type": kind, "inputs": values}
    return result


def validate_nodes(prompt, info):
    missing = sorted({node["class_type"] for node in prompt.values()} - info.keys())
    if missing:
        raise CyToolError("MissingNodes", "Install these nodes on the selected ComfyUI server: " + ", ".join(missing))
