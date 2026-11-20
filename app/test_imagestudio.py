"""Behavioral tests for image jobs and the ComfyUI transport boundary."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import time
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image

from build_manifest import GENERATION, TRAIN
from cytools_core import CyToolError, validate_parameters
from comfy import Client
import graphs
from settings import DEFAULTS, model_paths
from selection import suggestions
from tool import create_runtime
import updates


def node(**properties):
    return {"input": {"required": properties}}


class Fixture(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, data=None, status=200):
        self.send_response(status)
        self.end_headers()
        if data is not None:
            self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == "/object_info":
            self.reply(self.server.info)
        elif self.path == "/system_stats":
            self.reply({"system": {"ram_total": 1024, "api_key": "hidden"}, "devices": []})
        elif self.path == "/queue":
            running = [[1, self.server.current]] if self.server.hold and self.server.current else []
            self.reply({"queue_running": running, "queue_pending": []})
        elif self.path.startswith("/history/"):
            if self.server.hold:
                self.reply({})
            else:
                ident = self.path.rsplit("/", 1)[-1]
                self.reply({ident: {"status": {"completed": True, "status_str": "success"},
                                   "outputs": {"2": {"images": [{"filename": "server.png", "type": "output"}]}}}})
        elif self.path.startswith("/view?"):
            png = BytesIO()
            Image.new("RGB", (64, 64), "#499").save(png, format="PNG")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(png.getvalue())
        else:
            self.reply(status=404)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path == "/upload/image":
            self.server.uploaded.append(raw)
            self.reply({"subfolder": "unique", "name": "input.png"})
            return
        payload = json.loads(raw)
        if self.path == "/prompt":
            self.server.prompt = payload["prompt"]
            self.server.current = payload["prompt_id"]
            self.server.submitted.set()
            self.reply({"prompt_id": self.server.current})
        elif self.path == "/interrupt":
            self.server.interrupted.append(payload)
            self.reply()
        elif self.path in ("/free", "/queue"):
            self.reply()
        else:
            self.reply(status=404)


@contextmanager
def endpoint_fixture():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
    server.info = {
        "LoadImage": node(image=[["input.png"]]),
        "SaveImage": {**node(images=["IMAGE"], filename_prefix=["STRING", {"default": "Image"}]), "output_node": True},
    }
    server.uploaded = []
    server.hold = False
    server.current = ""
    server.prompt = {}
    server.submitted = Event()
    server.interrupted = []
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class ImageStudioTests(unittest.TestCase):
    def test_model_roots_and_component_recipe(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "mirror"
            unet = library / "models" / "unet"
            unet.mkdir(parents=True)
            direct = root / "Other Studio"
            direct.mkdir()
            config = {**DEFAULTS, "roots": [{"path": str(library), "category": "library"},
                                             {"path": str(direct), "category": "loras"}]}
            paths = model_paths(config, root / "main.py")["imagestudio"]
            self.assertEqual(paths["diffusion_models"], str(unet.resolve()))
            self.assertEqual(paths["loras"], str(direct.resolve()))
        info = {
            "UNETLoader": node(unet_name=[["qwen_image_edit_2511_fp8mixed.safetensors"]]),
            "CLIPLoader": node(clip_name=[["qwen_2.5_vl_7b_fp8_scaled.safetensors"]]),
            "VAELoader": node(vae_name=[["qwen_image_vae.safetensors"]]),
            "TextEncodeQwenImageEditPlus": node(),
        }
        self.assertEqual(suggestions(info, "txt2img", "auto"), [])
        candidate = suggestions(info, "img2img", "auto")[0]["parameters"]
        self.assertEqual(candidate["family"], "qwen_edit")
        self.assertEqual(candidate["vae"], "qwen_image_vae.safetensors")

    def test_generate_inpainting_applies_mask_lora_control_and_tile(self):
        params = validate_parameters({"checkpoint": "sd_xl_base.safetensors", "mode": "inpaint", "prompt": "A book",
             "image": "input.png", "mask": "mask.png", "controls": [{"model": "canny.safetensors", "image": "guide.png",
                 "preprocess": "canny", "low": 0.25, "start": 0.1, "end": 0.8}],
             "loras": [{"name": "shading.safetensors", "model_strength": 0.55}], "seed": 420,
             "tiled_vae": True, "tile_size": 256, "tile_overlap": 32, "inpaint_method": "conditioning",
             "batch": 2, "denoise": 0.42}, GENERATION)
        result, seed = graphs.generate(params, {}, {"image": "i", "mask": "m", "control_0": "g"}, "folder/image")
        self.assertEqual(seed, 420)
        by_type = {node["class_type"]: node["inputs"] for node in result.values()}
        self.assertEqual(by_type["LoraLoader"]["strength_model"], 0.55)
        self.assertEqual(by_type["ControlNetApplyAdvanced"]["end_percent"], 0.8)
        self.assertEqual(by_type["Canny"]["low_threshold"], 0.25)
        self.assertEqual(by_type["VAEDecodeTiled"]["tile_size"], 256)
        self.assertEqual(by_type["KSampler"]["denoise"], 0.42)
        self.assertIn("InpaintModelConditioning", by_type)
        self.assertEqual(by_type["RepeatLatentBatch"]["amount"], 2)

    def test_training_slot_depends_on_server_schema(self):
        params = validate_parameters({"checkpoint": "sd15.safetensors", "steps": 9, "rank": 4,
                                     "dataset": [{"image": "sample.png", "caption": "A book"}]}, TRAIN)
        info = {"TrainLoraNode": {**node(rank=["INT", {"default": 8}], steps=["INT", {"default": 1}],
                                         algorithm=[["LoRA"], {"default": "LoRA"}]),
                                  "output": ["MODEL", "LORA_MODEL", "LOSS_MAP", "INT"]}}
        graph = graphs.train(params, info, "input_folder", "train/adapter")
        trainer, save = list(graph.values())[-2:]
        self.assertEqual(trainer["inputs"]["steps"], 9)
        self.assertEqual(trainer["inputs"]["rank"], 4)
        self.assertEqual(save["inputs"]["lora"][1], 1)
        self.assertEqual(save["inputs"]["steps"][1], 3)

    def test_editor_workflow_conversion(self):
        info = {"LoadImage": node(image=[["test.png"]]),
                "SaveImage": node(images=["IMAGE"], filename_prefix=["STRING", {"default": "Default"}])}
        doc = {"nodes": [{"id": 1, "type": "LoadImage", "widgets_values": ["test.png", "image"]},
                         {"id": 2, "type": "SaveImage", "inputs": [{"name": "images", "link": 1}],
                          "widgets_values": ["Folder/result"]}], "links": [[1, 1, 0, 2, 0, "IMAGE"]]}
        prompt = graphs.workflow(doc, info)
        self.assertEqual(prompt["2"]["inputs"], {"images": ["1", 0], "filename_prefix": "Folder/result"})
        with self.assertRaisesRegex(CyToolError, "subgraph"):
            graphs.workflow({**doc, "definitions": {"subgraphs": [{}]}}, info)

    def test_remote_workflow_import_override_and_output(self):
        graph = {"1": {"class_type": "LoadImage", "inputs": {"image": "server_image.png"}},
                 "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "default"}}}
        with endpoint_fixture() as (server, url), TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "entrée source.png"
            Image.new("RGB", (8, 8), "blue").save(source)
            config = {**DEFAULTS, "profiles": {"remote": {"url": url}}}
            with patch("studio_runtime.read", return_value=config), create_runtime(folder / "jobs") as runtime:
                actor = runtime.authenticate(runtime.create_client("test", ("network", "filesystemExternal"))["token"])
                task = runtime.create_session(actor)["sessionId"]
                job = runtime.submit(actor, task, "run_workflow", {"profile": "remote", "graph": graph,
                      "uploads": [{"node": "1", "input": "image", "file": str(source)}],
                      "overrides": [{"node": "2", "input": "filename_prefix", "value": "renamed"}]})
                done = runtime.wait(actor, job["jobId"], timeout=20)
                self.assertEqual(done["state"], "Completed", done["error"])
                result = Path(done["outputs"]["assets"][0]["file"])
                self.assertTrue(result.is_relative_to(Path(done["workspace"])))
                self.assertEqual(done["outputs"]["assets"][0]["width"], 64)
                self.assertEqual(server.prompt["1"]["inputs"]["image"], "unique/input.png")
                self.assertEqual(server.prompt["2"]["inputs"]["filename_prefix"], "renamed")
                runtime.finish_task(actor, task)
                self.assertFalse(result.exists())
                self.assertTrue(source.exists())

    def test_targeted_interrupt_holds_until_engine_stops(self):
        with endpoint_fixture() as (server, url):
            server.hold = True
            cancel = Event()
            observed = []
            class Token:
                def check(self):
                    if cancel.is_set():
                        raise CyToolError("Cancelled", "Cancelled")
            class Context:
                cancellation = Token()
                def log(self, *_args, **_kw):
                    pass
                def progress(self, *_args, **_kw):
                    pass
            def work():
                try:
                    with Client(url, targeted=True) as client:
                        client.run({"1": {"class_type": "SaveImage", "inputs": {}}}, Context(), 30)
                except CyToolError as error:
                    observed.append(error.code)
            worker = Thread(target=work, daemon=True)
            worker.start()
            try:
                self.assertTrue(server.submitted.wait(5))
                cancel.set()
                deadline = time.monotonic() + 5
                while not server.interrupted and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertEqual(server.interrupted, [{"prompt_id": server.current}])
                self.assertTrue(worker.is_alive())
            finally:
                server.hold = False
                worker.join(timeout=5)
            self.assertEqual(observed, ["Cancelled"])

    def test_archive_rejects_traversal_and_can_restore(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            archive = folder / "bad.zip"
            with ZipFile(archive, "w") as zip_file:
                zip_file.writestr("app/../outside.py", "broken")
            with self.assertRaises(ValueError):
                updates.extract(archive, folder / "target", "v0.1.0")
            app, store = folder / "app", folder / "updates"
            app.mkdir()
            store.mkdir()
            new = store / "staged"
            new.mkdir()
            (app / "version").write_text("old")
            (new / "version").write_text("new")
            data = folder / "assets"
            data.write_text("user")
            with patch.object(updates, "APP", app), patch.object(updates, "UPDATES", store):
                backup = updates.activate(new)
                self.assertEqual((app / "version").read_text(), "new")
                updates.activate(backup)
                self.assertEqual((app / "version").read_text(), "old")
                self.assertEqual(data.read_text(), "user")


if __name__ == "__main__":
    unittest.main()
