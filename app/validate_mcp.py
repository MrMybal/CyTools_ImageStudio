"""Exercise the native stdio MCP host with the Python MCP conformance client."""
import argparse
import asyncio
import json
from pathlib import Path
import socket
import subprocess
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

APP = Path(__file__).resolve().parent


def result(reply):
    if reply.isError:
        raise AssertionError(reply.content[0].text)
    return json.loads(reply.content[0].text)


async def test(python, root):
    params = StdioServerParameters(command=str(python), args=[str(APP / "tool.py"), "--data-dir", str(root), "mcp"])
    async with stdio_client(params) as (receive, transmit):
        async with ClientSession(receive, transmit) as session:
            await session.initialize()
            tool_names = set()
            cursor = None
            while True:
                page = await session.list_tools(cursor=cursor)
                tool_names.update(tool.name for tool in page.tools)
                cursor = page.nextCursor
                if not cursor:
                    break
            required = {"op_studio_status", "op_suggest_models", "op_generate", "op_run_workflow", "op_train_lora"}
            assert required <= tool_names, required - tool_names
            job = result(await session.call_tool("op_studio_status", {"parameters": {}}))
            async with asyncio.timeout(15):
                while job["state"] not in ("Completed", "Failed", "Cancelled"):
                    await asyncio.sleep(.1)
                    job = result(await session.call_tool("cy_get_job", {"jobId": job["jobId"]}))
            assert job["state"] == "Completed", job
            assert job["outputs"]["defaultProfile"] in ("managed", "existing", "local", "remote")
            result(await session.call_tool("cy_finish_task", {}))
            print(json.dumps({"mcp": "native stdio", "tools": len(tool_names), "studio_status": job["state"]}))


async def test_http(python, root, image=None):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with (root / "native-http.log").open("w+", encoding="utf-8") as log:
        worker = (
            "import sys\n"
            "from tool import create_runtime\n"
            "from cytools_core.transports.http import make_server\n"
            "with create_runtime(sys.argv[1]) as runtime:\n"
            "    server = make_server(runtime, int(sys.argv[2]), access_profile='free')\n"
            "    try: sys.stdin.buffer.read(1)\n"
            "    finally: server.server_close()\n"
        )
        command = [str(python), "-c", worker, str(root / "http-runtime"), str(port)]
        child = subprocess.Popen(command, cwd=APP, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
        try:
            async with asyncio.timeout(12):
                while True:
                    if child.poll() is not None:
                        log.seek(0)
                        raise AssertionError(log.read())
                    try:
                        _, writer = await asyncio.open_connection("127.0.0.1", port)
                    except OSError:
                        await asyncio.sleep(.10)
                    else:
                        writer.close()
                        await writer.wait_closed()
                        break
            async with streamable_http_client(f"http://127.0.0.1:{port}/mcp") as (receive, transmit, _):
                async with ClientSession(receive, transmit) as session:
                    await session.initialize()
                    job = result(await session.call_tool("op_studio_status", {"parameters": {}}))
                    async with asyncio.timeout(15):
                        while job["state"] not in ("Completed", "Failed", "Cancelled"):
                            await asyncio.sleep(.1)
                            job = result(await session.call_tool("cy_get_job", {"jobId": job["jobId"]}))
                    assert job["state"] == "Completed", job
                    if image is not None:
                        async def invoke(name, parameters):
                            job = result(await session.call_tool("op_" + name, {"parameters": parameters}))
                            async with asyncio.timeout(200):
                                while job["state"] not in ("Completed", "Failed", "Cancelled"):
                                    await asyncio.sleep(.25)
                                    job = result(await session.call_tool("cy_get_job", {"jobId": job["jobId"]}))
                            assert job["state"] == "Completed", job
                            return job["outputs"]

                        try:
                            choice = await invoke("suggest_models", {"profile": "existing", "mode": "img2img"})
                            assert any(item["parameters"]["family"] == "qwen_edit" for item in choice["candidates"]), choice
                            graph = {
                                "1": {"class_type": "LoadImage", "inputs": {"image": "pending.png"}},
                                "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "MCP"}},
                            }
                            generated = await invoke("run_workflow", {"profile": "existing", "graph": graph,
                                "uploads": [{"node": "1", "input": "image", "file": str(image)}]})
                            asset = generated["assets"][0]
                            assert Path(asset["file"]).is_file() and asset["width"] == 384, asset
                            print(json.dumps({"mcp": "native HTTP", "workflow": "Completed", "width": asset["width"],
                                              "height": asset["height"], "installed_recipes": len(choice["candidates"])}))
                        finally:
                            stopped = await invoke("release_memory", {"profile": "existing", "stop": True})
                            assert stopped["engineStopped"]
                    result(await session.call_tool("cy_finish_task", {}))
                    print(json.dumps({"mcp": "native HTTP", "profile": "Free", "studio_status": job["state"]}))
        finally:
            child.stdin.close()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--workflow-image", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="imagestudio-mcp-") as directory:
        asyncio.run(test(args.python.resolve(), Path(directory) / "stdio"))
        image = args.workflow_image.resolve() if args.workflow_image else None
        asyncio.run(test_http(args.python.resolve(), Path(directory), image))
