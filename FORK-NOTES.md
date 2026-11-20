# Fork notes (christopherbearman/xllama)

Experimental changes on top of upstream xllama 1.7.0, packaged as **1.8.0**.
Not console-benchmarked by upstream.

## Image generation

- **LCM DreamShaper v7** (`lcm-dreamshaper-v7`, catalogue `kind: diffusion`):
  an SD1.5-class Latent Consistency Model, DirectML/Olive fp16 ONNX export
  (`softwareweaver/LCM_Dreamshaper_v7_Olive_Onnx`, MIT). Turn on **Better
  quality: LCM DreamShaper v7** in the Image dialog; the first use downloads
  it (~2.1 GB). 4 steps.
- `uwp/diffuse.cpp` detects an LCM model from `scheduler/scheduler_config.json`
  and feeds the UNet's `timestep_cond` (guidance embedding) using
  `include/xllama/diffusion/lcm_scheduler.h`, checked against diffusers'
  `LCMScheduler` in `tests/test_fork_additions.cpp`.
- **The chat model is unloaded before every image** (GUI and API), so a large
  chat model no longer causes `bad allocation`. The next chat turn reloads it.
- LAN API: `POST /v1/images/generations` with `"model": "lcm-dreamshaper-v7"`
  (anything containing `lcm` or `dreamshaper`) uses LCM; optional
  `guidance_scale` (1-20, default 8). Other names keep SD-Turbo.

## Chat

- **Qwen3.5 4B** (`qwen35-4b`, Q4_K_M, Apache-2.0) added as the best-quality
  coding model, `n_ctx` 8192.
- **Web search:** start a message with `/web ` (or `/search `), on the console
  or through the LAN API. The console fetches Bing RSS results and the model
  answers from the top 5 with numbered citations. Bing's RSS terms allow
  personal, non-commercial use.

## Remote diagnostics (LAN API, trusted network only)

| Route                   | What it returns                                        |
| ----------------------- | ------------------------------------------------------ |
| `GET /v1/debug/log`     | `{"log": ...}` the last ~400 lines of `xllama.log`     |
| `GET /v1/debug/status`  | resident model, app memory limit/usage, image progress |
| `POST /v1/debug/unload` | drops the resident chat model to free memory           |
