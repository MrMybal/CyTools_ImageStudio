"""Optional hardware smoke test. Reuse the locally installed ComfyUI and model weights."""
import argparse
import json
from contextlib import ExitStack
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image, ImageDraw
from settings import DATA, SETTINGS, read, save
from tool import create_runtime


def record_lora(report, source):
    path = Path(source)
    with path.open("rb") as weights:
        digest = hashlib.file_digest(weights, "sha256").hexdigest()
    (report / "trained-lora-checksum.json").write_text(json.dumps({
        "name": path.name, "sha256": digest, "bytes": path.stat().st_size
    }, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--library", action="append", required=True)
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--sd15", action="store_true")
    parser.add_argument("--managed-cpu", action="store_true")
    args = parser.parse_args()
    config = read()
    if not SETTINGS.is_file():
        config.update(profile="existing", main=args.main, python=args.python,
                      roots=[{"path": root, "category": "library"} for root in args.library])
        save(config)
    report = DATA / "reports"
    report.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        temp = stack.enter_context(TemporaryDirectory(prefix="smoke-", dir=report))
        if args.managed_cpu:
            stack.enter_context(patch("engine.RUNTIME", Path(temp) / "engine-test"))
            stack.enter_context(patch("studio_runtime.read", return_value={
                **config, "profile": "managed", "models": str(Path(temp) / "models"), "roots": []
            }))
        if args.sd15:
            stack.enter_context(patch("studio_runtime.read", return_value={**config, "models": str(Path(temp) / "models")}))
        runtime = stack.enter_context(create_runtime(Path(temp) / "jobs"))
        permissions = ("network", "filesystemExternal", "manageResources")
        actor = runtime.authenticate(runtime.create_client("ImageStudio hardware smoke", permissions)["token"])
        session = runtime.create_session(actor, temporary=False)["sessionId"]

        def invoke(name, parameters, timeout):
            job = runtime.submit(actor, session, name, parameters)
            actual = runtime.wait(actor, job["jobId"], timeout=timeout)
            summary = {"state": actual["state"], "error": actual["error"], "outputs": actual["outputs"]}
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            suffix = "-sd15" if args.sd15 else "-managed-cpu" if args.managed_cpu else ""
            (report / (name + suffix + ".json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            if actual["state"] != "Completed":
                for path in Path(actual["workspace"]).rglob("*.log"):
                    (report / (name + suffix + "-" + path.name)).write_bytes(path.read_bytes())
                raise SystemExit(1)
            return actual

        if args.managed_cpu:
            invoke("install_engine", {"compute": "cpu"}, 1000)
            outcome = invoke("inspect_backend", {"profile": "managed"}, 200)
            assert any(device["type"] == "cpu" for device in outcome["outputs"]["devices"])
            invoke("release_memory", {"profile": "managed", "stop": True}, 40)
            return

        if args.sd15:
            checkpoint = "v1-5-pruned-emaonly.safetensors"
            invoke("download_model", {"category": "checkpoints", "filename": checkpoint, "accept_license": True,
                "url": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/f03de32/" + checkpoint,
                "sha256": "6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa"}, 1200)
        invoke("inspect_backend", {"profile": "existing"}, 190)
        if args.qwen:
            source = Path(temp) / "source.png"
            with Image.new("RGB", (384, 384), "#21283e") as bitmap:
                draw = ImageDraw.Draw(bitmap)
                draw.ellipse((60, 60, 324, 324), fill="#c5813d")
                bitmap.save(source)
            job = invoke("generate", {
                "profile": "existing", "family": "qwen_edit", "mode": "img2img",
                "diffusion": "qwen_image_edit_2511_fp8mixed.safetensors", "encoder": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "vae": "qwen_image_vae.safetensors", "image": str(source), "seed": 174,
                "prompt": "Turn the orange circle into a bright blue sphere and preserve the dark blue background.",
                "width": 384, "height": 384, "steps": 4, "cfg": 2.5,
            }, 500)
            source_out = Path(job["outputs"]["assets"][0]["file"])
            (report / "qwen-edit.png").write_bytes(source_out.read_bytes())
        if args.train:
            example = Path(temp) / "training.png"
            with Image.new("RGB", (128, 128), "#20384e") as bitmap:
                ImageDraw.Draw(bitmap).ellipse((16, 16, 112, 112), fill="#e19163")
                bitmap.save(example)
            job = invoke("train_lora", {
                "profile": "existing", "family": "qwen_edit",
                "diffusion": "qwen_image_edit_2511_fp8mixed.safetensors", "encoder": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "vae": "qwen_image_vae.safetensors", "dataset": [{"image": str(example), "caption": "A peach colored circle."}],
                "rank": 1, "steps": 1, "offloading": True, "training_dtype": "none"
            }, 700)
            record_lora(report, job["outputs"]["assets"][0]["file"])
        if args.sd15:
            base = {"profile": "existing", "family": "checkpoint", "checkpoint": checkpoint}
            params = {**base, "width": 192, "height": 192, "steps": 4, "seed": 371,
                      "prompt": "A blue ceramic sculpture of a small sphere, studio photograph"}
            job = invoke("generate", params, 300)
            source = job["outputs"]["assets"][0]["file"]
            (report / "sd15-text2img.png").write_bytes(Path(source).read_bytes())
            mask = Path(temp) / "mask.png"
            with Image.new("L", (192, 192), 0) as bitmap:
                ImageDraw.Draw(bitmap).rectangle((60, 60, 132, 132), fill=255)
                bitmap.save(mask)
            out = invoke("generate", {**params, "mode": "inpaint", "image": source, "mask": str(mask),
                         "prompt": "An orange ceramic sphere"}, 300)
            (report / "sd15-inpaint.png").write_bytes(Path(out["outputs"]["assets"][0]["file"]).read_bytes())
            trained = invoke("train_lora", {**base, "rank": 1, "steps": 2,
                             "dataset": [{"image": source, "caption": "A small blue sphere"}]}, 420)
            record_lora(report, trained["outputs"]["assets"][0]["file"])
        invoke("release_memory", {"profile": "existing", "stop": True}, 45)


if __name__ == "__main__":
    main()
