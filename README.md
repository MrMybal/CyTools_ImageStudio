# CyTools_ImageStudio

AI image generation and editing with local or remote ComfyUI, an inpainting canvas, ControlNet, LoRA training, and MCP automation.

[Guide en français](LISEZ-MOI.md) · [Developer documentation](app/README.md) · [Tested capabilities](app/CapabilityCoverage.md)

## Features

- Create images from text, use an image as a starting point, or paint a mask on the CyUI canvas for inpainting.
- Add LoRA adapters and ControlNet inputs. Train a LoRA from captioned images on a compatible ComfyUI engine launched by ImageStudio.
- Launch a managed ComfyUI installation, reuse an existing installation, or connect to an HTTP(S) server. Import ComfyUI workflows and adjust their inputs before running them.
- Reference model directories across multiple drives. The download catalog includes Stable Diffusion, FLUX.1, and Qwen Image components. The selected ComfyUI server determines which models and nodes can run.
- Automate model discovery, generation, workflows and LoRA training through the native CyToolsCore MCP service.

## Quick start on Windows x64

Install Python 3.11 or newer and obtain the Windows [CyToolsCore platform wheel](app/README.md#cytoolscore-platform-wheel) for version 0.12.0. From the repository root, run the installer and open the included Windows launcher.

```powershell
powershell -ExecutionPolicy Bypass -File .\app\Install.ps1 -CoreWheel C:\path\to\cytools_core-0.12.0-py3-none-win_amd64.whl
.\CyTools_ImageStudio.exe
```

1. In **Options → ComfyUI connections**, choose an installed ComfyUI, an existing server, or the managed installation. Git is required to install the managed engine.
2. In **Options → Model search folders**, add an existing ComfyUI library or assign a standalone folder to a category such as checkpoints or LoRAs. Save the settings and select **Refresh models**. An external server uses its own libraries.
3. In **Studio**, select a task and model. Use **Suggest installed models** to find a matching recipe, or choose the components explicitly.

The repository includes the Windows launcher and application source. The installer creates a private Python environment. Models are supplied through existing libraries or downloads in the **Models** tab. The Linux launcher and setup script are documented in the [French guide](LISEZ-MOI.md#installation-sur-une-autre-machine).

## MCP

Open **Options → MCP**, start the local service, and copy its URL into your client. The service shares the desktop's runtime and queue. A standalone stdio MCP entry point is also available.

Clients can call `op_inspect_backend` and `op_suggest_models`, then submit `op_generate`, `op_run_workflow`, or `op_train_lora`. Use `cy_get_job` for progress and `cy_export_job` to keep files before finishing the task.

## Validation

Windows x64 testing covered Qwen Image Edit, Stable Diffusion 1.5 text-to-image and mask inpainting, SD 1.5 LoRA training, an MCP-submitted ComfyUI workflow, and installing and starting a fresh managed CPU engine. See the [feature matrix](app/CapabilityCoverage.md) for test limits, including GPU memory requirements.

## License

ImageStudio is licensed under [AGPL-3.0-only](LICENSE). See [component notices](LICENCES/SOURCES.txt) for the engine, SDKs, UI and model licenses.
