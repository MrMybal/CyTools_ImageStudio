# CyTools_ImageStudio development

The desktop, CLI and MCP entry points use the same CyToolsCore 0.12.0 runtime. The app uses the native CyMCPSDK communication wheel, Dear ImGui and the CyUI Python modules. ComfyUI runs independently, either as a process owned by the runtime or as a configured HTTP(S) endpoint.

## Build and run

On Windows x64, place the platform Core wheel in `app/packages` or pass `-CoreWheel` to `app/Install.ps1`. Use a local Python 3.11+ and run the launcher build from a Visual Studio 2022 x64 developer environment.

```powershell
powershell -ExecutionPolicy Bypass -File app/Install.ps1
app\Build-Windows.cmd
.\runtime\python\Scripts\python.exe app\build_manifest.py
.\runtime\python\Scripts\cytools.exe validate app\CyTool.json
.\runtime\python\Scripts\python.exe -m unittest discover -s app -p test_imagestudio.py -v
```

The default settings are under `data/settings`. Model and image assets, job state, caches, and private Python installations are excluded from source distribution. Build the Core platform wheel using the SDK’s native instructions for each operating system.

### CyToolsCore platform wheel

Use a CyToolsCore 0.12.0 wheel built for the target operating system and architecture. For a source build, follow the [CyToolsCore native communication guide](https://github.com/MrMybal/CyToolsCore/blob/main/Docs/NativeCommunication.md) to build the native library and package the wheel. Pass that file to `app/Install.ps1 -CoreWheel` on Windows or to `sh app/install_linux.sh` on Linux.

The Windows executable loads `app/desktop.py` with the Python environment installed under `runtime/python`. Rebuilding the executable requires the Visual Studio C++ toolchain. Normal installs can use the executable from the repository.

- `studio_runtime.py` registers jobs, imports and workers.
- `engine.py` installs and launches ComfyUI and writes its model path configuration.
- `comfy.py` connects to the self-hosted ComfyUI API and tracks the accepted prompt through completion.
- `graphs.py` builds image and training graphs and converts standard editor workflows.
- `studio_ui.py` and `canvas_ui.py` provide the CyUI workspace.

## MCP

Call `op_inspect_backend` first to obtain exact model names, available node types, samplers, schedulers and memory measurements. `op_suggest_models` assembles recipes for recognized families from model filenames and the available loaders. Always select a workflow matching the model architecture. An arbitrary custom model can run through a valid ComfyUI API workflow when its nodes and dependencies are installed.

Outputs are files in the job workspace. `cy_export_job` saves a copy before `cy_finish_task` removes the temporary workspace. Profile, model names, paths and other generation settings can be saved in a generation recipe. Provider credentials are supplied from the current process session, the standard Hugging Face token store or environment variables. Variables for external ComfyUI include `IMAGESTUDIO_COMFY_TOKEN_LOCAL` and `IMAGESTUDIO_COMFY_TOKEN_REMOTE`.

## GitHub updates

Set the application's GitHub repository as `owner/name` in Options. Application releases use a version tag such as `v0.1.1` and a single `CyTools_ImageStudio-app.zip` asset with `app/` as its only root. GitHub must provide a SHA256 digest for the ZIP. The updater verifies the size, digest, manifest identity, version and Python syntax before staging. The launcher swaps the application directory under the runtime lock on restart. The preceding app directory is retained for restoration.

Use this asset format for updates compatible with the installed Core wheel and launcher. Use a fresh installation for incompatible runtime or platform prerequisites. The ComfyUI engine and model downloads have their own versioning. The in-app updater consumes tagged release assets. Source commits are available from the repository.

## Sources

- The ImageStudio application is licensed under [GNU Affero General Public License v3.0](../LICENSE) (AGPL-3.0-only). Copyright (C) 2026 Cyberalien. The [ImageStudio repository](https://github.com/MrMybal/CyTools_ImageStudio) is the project source location.
- [CyToolsCore](https://github.com/MrMybal/CyToolsCore) and [CyMCPSDK](https://github.com/MrMybal/CyMCPSDK)
- [CyUI](https://github.com/MrMybal/CyUI), 1.2.0 at `6ef01cfcdb197f3a89d64618ce5ef8818c704728`
- [ComfyUI server API](https://docs.comfy.org/development/comfyui-server/comms_routes) and [ComfyUI v0.37.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.37.0)
