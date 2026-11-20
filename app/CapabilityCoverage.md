# ImageStudio feature coverage

| Engine capability | CyTools operation | Desktop control | Verification |
| --- | --- | --- | --- |
| Server devices, memory, object definitions and model names | `inspect_backend` | Refresh models | Real Windows ComfyUI 0.37.0 instance with RTX 3090. Reads two configured local model libraries in place. |
| Recognized installed-model recipes | `suggest_models` | Suggest installed models | Tests identify Qwen Edit with its three model components and reject the same recipe for text to image. |
| Text to image, image to image and inpainting | `generate` | Studio canvas and inspector | Stable Diffusion 1.5 generated and inpainted 192 × 192 PNGs. Qwen Image Edit produced a 384 × 384 PNG. Graph tests cover batch, denoise and VAE tiling. |
| LoRA stacks and ControlNet | `generate` | LoRA and ControlNet sections | Graph tests verify adapter strength, control interval and Canny thresholds reach the emitted graph. |
| ComfyUI API and editor workflows | `inspect_workflow`, `run_workflow` | Workflows | HTTP integration test with an imported absolute Unicode input, node override, exported image and task cleanup. Subgraphs and extension-defined widgets use frontend API export. |
| Native LoRA training | `train_lora` | LoRA training | Stable Diffusion 1.5 produced rank-1 LoRA weights in a two-step smoke run. The Qwen Edit training attempt exceeded the RTX 3090's 24 GiB VRAM. |
| Model paths and model downloads | `download_model`, `import_model` | Models and Options | ComfyUI read two machine libraries in place. The 4.27 GB SD 1.5 test checkpoint passed SHA256 verification in a temporary library. |
| Managed process and existing installation | `install_engine`, `release_memory` | Options | Windows existing GPU installation and fresh managed CPU installation both started and stopped. The managed installer used ComfyUI v0.37.0 and private PyTorch 2.11. |
| Cancellation and isolation | Core jobs and ComfyUI queue | Cancel | Test confirms targeted interrupt and retains the job until it leaves the queue. Older external servers wait for a running prompt to complete. |
| Desktop and MCP | CyUI and native Core transports | Studio, Advanced, MCP | Windows launcher from a different working directory, English and French renders, native stdio and local HTTP. MCP suggested installed Qwen components and executed a real ComfyUI workflow that returned a 384 × 384 PNG. |
| GitHub update and restoration | Startup maintenance | Options | Test rejects path traversal and swaps/restores only the application folder. Live GitHub delivery requires an app repository with the documented assets. |

Guided graph families include compatible Stable Diffusion checkpoints, FLUX.1, FLUX.2, Qwen Image, Qwen Image Edit and SD3. Model formats requiring other loaders or specialized preprocessing can be used through an API workflow. The target ComfyUI version and installed nodes and weights determine whether a graph runs. The Windows x64 launcher and RTX 3090 route are verified. Live FLUX and ControlNet inference, a real external server, fresh managed CUDA installation, CPU model inference, and Linux desktop remain unverified.
