"""Local machine configuration. Remote callers select a configured profile by name."""
import copy
import json
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNTIME = ROOT / "runtime"
SETTINGS = DATA / "settings" / "studio.json"

CATEGORIES = {
    "checkpoints": ("checkpoints", "Stable-diffusion"),
    "diffusion_models": ("diffusion_models", "unet"),
    "text_encoders": ("text_encoders", "clip"),
    "vae": ("vae", "VAE"),
    "loras": ("loras", "Lora", "LyCORIS"),
    "controlnet": ("controlnet", "ControlNet", "t2i_adapter"),
    "clip_vision": ("clip_vision",),
    "embeddings": ("embeddings",),
    "upscale_models": ("upscale_models", "ESRGAN", "RealESRGAN"),
    "vae_approx": ("vae_approx",),
    "style_models": ("style_models",),
    "model_patches": ("model_patches",),
    "diffusers": ("diffusers",),
}

DEFAULTS = {
    "profile": "managed",
    "profiles": {
        "local": {"url": "http://127.0.0.1:8188", "targeted_interrupt": False},
        "remote": {"url": "", "targeted_interrupt": False},
    },
    "main": "", "python": "", "roots": [], "models": str(DATA / "models"),
    "memory": "balanced", "reserve": 1.0, "device": "",
    "custom_nodes": False, "release": True, "language": "en", "theme": "mid-grey",
    "scale": 1.0, "autosave": False,
}


def read():
    config = copy.deepcopy(DEFAULTS)
    if SETTINGS.is_file():
        config.update(json.loads(SETTINGS.read_text(encoding="utf-8")))
    return config


def save(config):
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    temporary = SETTINGS.with_name(f".studio-{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(SETTINGS)
    finally:
        temporary.unlink(missing_ok=True)


def endpoint(value):
    url = urlsplit(value.strip())
    if (url.scheme not in ("http", "https") or not url.hostname or url.username or
            url.password or url.query or url.fragment or any(c.isspace() for c in value)):
        raise ValueError("Enter a ComfyUI HTTP(S) base URL without credentials or query parameters.")
    _ = url.port
    return url.geturl().rstrip("/")


def model_paths(config, main):
    """ComfyUI accepts one absolute path per line per model category."""
    paths = {key: [] for key in CATEGORIES}
    roots = [{"path": str(main.parent / "models"), "category": "library"}, *config["roots"]]
    for item in roots:
        root = Path(item["path"]).expanduser().resolve()
        if "\n" in str(root) or "\r" in str(root) or not root.is_dir():
            continue
        key = item["category"]
        if key in CATEGORIES:
            paths[key].append(str(root))
        elif key == "library":
            for category, aliases in CATEGORIES.items():
                paths[category].extend(str(base / name) for base in (root, root / "models")
                                       for name in aliases if (base / name).is_dir())
    if config.get("custom_nodes") and (main.parent / "custom_nodes").is_dir():
        paths["custom_nodes"] = [str(main.parent / "custom_nodes")]
    return {"imagestudio": {key: "\n".join(dict.fromkeys(values)) for key, values in paths.items() if values}}
