"""Conservative model recipes derived from the selected server's installed components."""
from graphs import catalog


def suggestions(info, task, preferred):
    library = catalog(info)

    def names(category, loader):
        return [entry["name"] for entry in library[category] if loader in entry["loaders"]]

    def match(category, loader, predicate):
        return next((name for name in names(category, loader) if predicate(name.casefold())), "")

    candidates = []

    def add(family, name, **parameters):
        if preferred in ("auto", family):
            candidates.append({"name": name, "parameters": {"family": family, "mode": task, **parameters},
                               "basis": "filename_and_loader"})

    qwen_encoder = match("text_encoders", "CLIPLoader", lambda name: "qwen_2.5_vl_7b" in name)
    qwen_vae = match("vae", "VAELoader", lambda name: "qwen_image_vae" in name)
    if qwen_encoder and qwen_vae:
        is_edit = task == "img2img"
        required = "TextEncodeQwenImageEditPlus" if is_edit else "EmptySD3LatentImage"
        if task != "inpaint" and required in info:
            for name in names("diffusion_models", "UNETLoader"):
                lower = name.casefold()
                if "qwen_image" in lower and ("edit" in lower) == is_edit and "layered" not in lower:
                    add("qwen_edit" if is_edit else "qwen", name, diffusion=name, encoder=qwen_encoder,
                        vae=qwen_vae, cfg=2.5, steps=25, sampler="euler", scheduler="simple")

    flux_l = match("text_encoders", "CLIPLoader", lambda name: "clip_l." in name)
    flux_t5 = match("text_encoders", "CLIPLoader", lambda name: "t5xxl" in name)
    flux_vae = match("vae", "VAELoader", lambda name: name.rsplit("/", 1)[-1] == "ae.safetensors")
    if flux_l and flux_t5 and flux_vae and "DualCLIPLoader" in info:
        for name in names("diffusion_models", "UNETLoader"):
            lower = name.casefold()
            if "flux1" not in lower or not any(f"flux1{sep}{model}" in lower for sep in ("-", "_") for model in ("dev", "schnell")):
                continue
            schnell = "schnell" in lower
            add("flux", name, diffusion=name, encoder=flux_l, encoder2=flux_t5, vae=flux_vae, cfg=1,
                steps=4 if schnell else 24, guidance=0.0 if schnell else 3.5, sampler="euler", scheduler="simple")

    for name in names("checkpoints", "CheckpointLoaderSimple"):
        lower = name.casefold()
        known_sd = any(key in lower for key in ("v1-5-pruned", "stable-diffusion", "sdxl", "sd_xl"))
        schnell = "flux1-schnell" in lower
        if not known_sd and not schnell:
            continue
        if "inpaint" in lower and task != "inpaint" or "refiner" in lower:
            continue
        is_inpaint = task == "inpaint" and "inpaint" in lower
        add("checkpoint", name, checkpoint=name, steps=4 if schnell else 25, cfg=1 if schnell else 7,
            sampler="euler", scheduler="simple" if schnell else "normal",
            inpaint_method="conditioning" if is_inpaint else "noise_mask")
    return candidates
