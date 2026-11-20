"""HTTP/API adapter. Heavy inference always lives in a separate ComfyUI process."""
import json
import os
import time
from urllib.parse import quote, urlsplit
from uuid import uuid4

import requests
import websocket
from cytools_core import CyToolError

TIMEOUT = (5, 25)


class Client:
    def __init__(self, url, token="", *, owned=False, targeted=False, alive=lambda: True):
        self.url = url
        self.owned = owned
        self.targeted = targeted or owned
        self.alive = alive
        self.token = token
        self.http = requests.Session()
        if urlsplit(url).hostname in ("127.0.0.1", "localhost", "::1"):
            self.http.trust_env = False
        if token:
            self.http.headers["Authorization"] = f"Bearer {token}"
        self.http.headers["User-Agent"] = "CyTools_ImageStudio/0.1"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.http.close()

    def request(self, method, route, **kwargs):
        try:
            response = self.http.request(method, self.url + route, timeout=kwargs.pop("timeout", TIMEOUT),
                                         allow_redirects=False, **kwargs)
        except requests.RequestException as error:
            raise CyToolError("ComfyUnavailable", f"ComfyUI {route} is unavailable ({type(error).__name__}).",
                              recoverable=True) from error
        if not 200 <= response.status_code < 300:
            message = f"ComfyUI {route} returned HTTP {response.status_code}."
            if route == "/prompt" and response.status_code == 400:
                try:
                    body = response.json()
                    lines = [str(body.get("error", {}).get("message", ""))]
                    for node, item in body.get("node_errors", {}).items():
                        lines.extend(f"{node} ({item.get('class_type', '')}): {entry.get('details') or entry.get('message')}"
                                     for entry in item.get("errors", []))
                    message += " " + "\n".join(lines)[:1800]
                except (ValueError, AttributeError):
                    pass
            response.close()
            code = "AccessRequired" if response.status_code in (401, 403) else "ComfyRequestFailed"
            raise CyToolError(code, message, recoverable=True)
        return response

    def json(self, route, data=None, **kwargs):
        with self.request("GET" if data is None else "POST", route, json=data, **kwargs) as response:
            if response.content:
                return response.json()
            return None

    def info(self):
        return self.json("/object_info", timeout=(5, 60))

    def upload(self, file, folder, name=None):
        with file.open("rb") as stream:
            with self.request("POST", "/upload/image", files={"image": (name or file.name, stream)},
                              data={"subfolder": folder, "type": "input", "overwrite": "false"},
                              timeout=(5, 90)) as response:
                item = response.json()
        return "/".join(part for part in (item.get("subfolder"), item["name"]) if part)

    def download(self, entry, destination, context):
        params = {key: entry[key] for key in ("filename", "subfolder", "type") if key in entry}
        with self.request("GET", "/view", params=params, stream=True) as response:
            with destination.open("xb") as output:
                for block in response.iter_content(1024 * 1024):
                    context.cancellation.check()
                    output.write(block)

    def queue_state(self, prompt_id):
        queue = self.json("/queue")
        return next((state for state in ("queue_running", "queue_pending")
                     if any(len(item) > 1 and item[1] == prompt_id for item in queue.get(state, []))), "")

    def free(self):
        self.json("/free", {"unload_models": True, "free_memory": True})

    def drain(self, prompt_id, context):
        """Keep Core's reservation until the accepted prompt has actually left the queue."""
        interrupted = False
        deleted = False
        warned = False
        while self.alive():
            try:
                state = self.queue_state(prompt_id)
                if not state:
                    return
                if state == "queue_pending" and not deleted:
                    self.json("/queue", {"delete": [prompt_id]})
                    deleted = True
                if state == "queue_running" and not interrupted and self.targeted:
                    self.json("/interrupt", {"prompt_id": prompt_id})
                    interrupted = True
                elif state == "queue_running" and not self.targeted and not warned:
                    context.log("Waiting for the external prompt to finish. This profile has no targeted interrupt.")
                    warned = True
            except (CyToolError, ValueError):
                if not warned:
                    context.log("ComfyUI connection lost. Waiting to confirm the prompt has stopped.", level="Warning")
                    warned = True
            time.sleep(0.6)

    def run(self, prompt, context, deadline_seconds):
        client_id, local_id = str(uuid4()), str(uuid4())
        socket = None
        try:
            url = self.url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            socket = websocket.create_connection(f"{url}/ws?clientId={client_id}", timeout=1.0,
                                                 suppress_origin=True, header=headers,
                                                 redirect_limit=0)
        except (websocket.WebSocketException, OSError, ValueError):
            pass
        prompt_id = None
        try:
            context.cancellation.check()
            try:
                reply = self.json("/prompt", {"prompt": prompt, "client_id": client_id, "prompt_id": local_id},
                                  timeout=(5, 120))
            except CyToolError as error:
                if error.code == "ComfyUnavailable" and self.owned:
                    # The owned engine can be stopped by the caller if submission was ambiguous.
                    raise CyToolError("SubmissionUncertain", "ComfyUI did not confirm prompt submission.",
                                      recoverable=True) from error
                raise
            prompt_id = reply["prompt_id"]
            context.log(f"ComfyUI prompt {prompt_id}")
            context.progress(0, currentStep="Queued in ComfyUI")
            started = time.monotonic()
            next_poll = 0.0
            missing_since = None
            completed = set()
            current = None
            while True:
                context.cancellation.check()
                now = time.monotonic()
                if now - started > deadline_seconds:
                    raise CyToolError("ComfyTimeout", "The ComfyUI job exceeded its time limit.", recoverable=True)
                if now >= next_poll:
                    next_poll = now + 1
                    try:
                        history = self.json("/history/" + quote(prompt_id, safe="")) or {}
                        if prompt_id in history:
                            status = history[prompt_id].get("status", {})
                            if status.get("completed") or status.get("status_str") == "success":
                                return prompt_id, history[prompt_id]
                            errors = [data for event, data in status.get("messages", [])
                                      if event in ("execution_error", "execution_interrupted")]
                            description = errors[-1] if errors else {}
                            message = str(description.get("exception_message") or "ComfyUI stopped the workflow.")[:1600]
                            if "out of memory" in message.lower() or "outofmemory" in str(description.get("exception_type", "")).lower():
                                raise CyToolError("ComfyOutOfMemory", message, recoverable=True)
                            raise CyToolError("ComfyExecutionFailed", message)
                        if self.queue_state(prompt_id):
                            missing_since = None
                        elif missing_since is None:
                            missing_since = now
                        elif now - missing_since > 5:
                            raise CyToolError("ComfyPromptMissing", "The prompt left the queue without a result.")
                    except CyToolError as error:
                        if error.code != "ComfyUnavailable" or not self.alive():
                            raise
                if socket is None:
                    time.sleep(0.25)
                    continue
                try:
                    message = socket.recv()
                    if not message:
                        socket.close()
                        socket = None
                        continue
                    if not isinstance(message, str):
                        continue
                    event = json.loads(message)
                except websocket.WebSocketTimeoutException:
                    continue
                except (websocket.WebSocketException, OSError, ValueError):
                    socket.close()
                    socket = None
                    continue
                data = event.get("data", {})
                if data.get("prompt_id") != prompt_id:
                    continue
                kind = event.get("type")
                if kind == "execution_cached":
                    completed.update(data.get("nodes", []))
                elif kind == "executing":
                    if current is not None:
                        completed.add(current)
                    current = data.get("node")
                    if current in prompt:
                        context.progress(min(98, len(completed) * 100 / len(prompt)),
                                         currentStep=prompt[current]["class_type"])
                elif kind == "progress":
                    step, total = data.get("value", 0), data.get("max", 1)
                    fraction = step / max(1, total)
                    context.progress(min(98, (len(completed) + fraction) * 100 / len(prompt)),
                                     currentStep=f"{prompt.get(current, {}).get('class_type', 'ComfyUI')} {step}/{total}")
        except BaseException:
            if prompt_id:
                self.drain(prompt_id, context)
            raise
        finally:
            if socket is not None:
                socket.close()


def external_client(url, profile):
    token = os.getenv("IMAGESTUDIO_COMFY_TOKEN_" + profile.upper(), "")
    if token and not (url.startswith("https://") or urlsplit(url).hostname in ("127.0.0.1", "localhost", "::1")):
        raise ValueError("A ComfyUI bearer token requires HTTPS or a loopback endpoint.")
    return token
