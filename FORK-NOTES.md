# Fork notes (christopherbearman/xllama)

## Use the Xbox as a server

Open **`http://<console-ip>:11434/ui`** in any browser on the network (Mac,
phone, tablet). It is a full chat app served by xllama itself: streaming
chat, model picker and downloads, Web search and Image modes, code blocks with
copy / save / live HTML preview, chats saved in that browser, and a status,
log and settings panel. Source: `uwp/web/index.html`; after editing run
`python3 scripts/embed-web-ui.py` to regenerate `uwp/web-ui-html.h`.

Server behaviour (1.9.0 / 1.10.0):

- The console shows a **server screen** (address + status only); "Show
  console chat" brings back the old UI ("console-chat.flag" remembers it).
- **GPU decode** on by default for `qwen35-4b` and `phi4-mini` (catalogue
  `gpu_layers: 99`): ~1.7x / ~2x faster on Series X.
- **Conversation memory reuse** for Qwen3-family no-think models: history
  replays the empty think block, so follow-up turns prefill only the new
  message (`ChatFormat::history_keeps_gen_suffix`).
- `max_tokens` larger than a model's context is clamped to half the context.

- API requests **queue** (up to 5 minutes) instead of answering `busy`.
- xllama holds a **DisplayRequest** while open, so the console stays awake.
- `GET /v1/debug/status` reports `version` and `uptime_s`.
- xllama must stay **open on screen** (UWP apps stop in the background) and
  must be set to **App type: Game** after every install.

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
  or through the LAN API. The console searches and the model answers from the
  top 5 results with numbered citations. Providers: **Tavily** (recommended,
  free tier) or **Brave Search**, set with
  `POST /v1/config/web-search {"key": "tvly-..."}` (or `"brave:<key>"`). With
  no key it falls back to Bing RSS, whose results are poor for questions.

## /image command

Start a message with `/image ` (or `/img `) on the console or through
`/v1/chat/completions` (e.g. Open WebUI) and the console generates a picture
with LCM DreamShaper v7 (SD-Turbo if LCM is not downloaded). Through the API the
reply is a markdown image with a `data:` URI, so chat clients show it inline;
earlier embedded images are replaced with `[image]` in incoming history so they
do not fill the context. On the console screen, view it with **Image**.

## Remote diagnostics (LAN API, trusted network only)

| Route                   | What it returns                                        |
| ----------------------- | ------------------------------------------------------ |
| `GET /v1/debug/log`     | `{"log": ...}` the last ~400 lines of `xllama.log`     |
| `GET /v1/debug/status`  | resident model, app memory limit/usage, image progress |
| `POST /v1/debug/unload` | drops the resident chat model to free memory           |
