# Series X fork (christopherbearman/xllama), November 2026

This fork turns xllama into a LAN coding and image assistant on one Xbox Series
X in Developer Mode, and moves most of a 9B model's work onto the console's GPU.
Every number below was measured on that console unless it says otherwise; the
upstream project measures on Series S, so its tables in
[benchmarks.md](benchmarks.md) are not replaced by these.

Platform rules (memory, disk, blocked APIs) stay in
[uwp-constraints.md](uwp-constraints.md); the D3D12 backend's history up to
October is in [gguf-gpu-decode.md](gguf-gpu-decode.md). This page owns what the
fork added on top.

## What the fork adds

| Area               | Change                                                                                                                                                                                                                                                                  | Where                                                                          |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Chat web app       | Chat page served by the console at `/ui`: Build mode (coding preset, plan-first thinking, automatic page check and patch-mode fix), image mode with an optional prompt-writing step, auto-continue at the length limit, live progress, model nicknames, in-page preview | `uwp/web/index.html` (embedded by `scripts/embed-web-ui.py`)                   |
| Models             | Qwen3.5 4B and 9B (IQ3_S "Big Queef", IQ4_XS "Big Queef+"), Phi-4 mini, LFM2.5; q8_0 KV cache via catalogue `kv_q8`                                                                                                                                                     | `uwp/models/manifest.json`                                                     |
| GPU matmul kernels | IQ3_XXS, IQ4_XS and Q5_K matrix-vector kernels; 8-column tiled kernels for prefill                                                                                                                                                                                      | `shaders/ggml_d3d12_mmv_*.hlsl`, `ggml_d3d12_mmt_*.hlsl`                       |
| GPU attention      | `FLASH_ATTN_EXT` on the GPU for F16 and Q8_0 caches                                                                                                                                                                                                                     | `shaders/ggml_d3d12_fa.hlsl`                                                   |
| GPU small ops      | ADD, MUL, RMS_NORM, SILU, SWIGLU, and q8_K quantization of GPU-produced inputs, so a layer's matmuls chain in one submission                                                                                                                                            | `shaders/ggml_d3d12_ew.hlsl`                                                   |
| CPU                | `CONCAT` split across threads (patch 0002)                                                                                                                                                                                                                              | `patches/0002-cpu-concat-row-threads.patch`                                    |
| KV reuse           | Text-extension reuse, continue-in-place, and a recurrent-state checkpoint for hybrid models (Qwen3.5, LFM2)                                                                                                                                                             | `src/bridge/session.cpp`                                                       |
| Prompt budget      | History is kept by shrinking the reply cap before any turn is dropped (`fit_prompt_keep_history`)                                                                                                                                                                       | `include/xllama/prompt_budget.h`                                               |
| Diagnostics        | Per-op profiling, GPU phase counters, a 60 s GPU submission timeout that names its ops, hardware exceptions logged with a D3D12 breadcrumb                                                                                                                              | `src/bridge/op_profile.cpp`, `src/bridge/ggml_d3d12.cpp`, `uwp/api-server.cpp` |

Every new GPU path checks itself against its host emulation on first use and
stays off if they disagree. The log says which (`d3d12: flash attention on
(check …)`, `d3d12: small ops on GPU on (check …)`).

## Switches (LocalState files)

Each switch is a file in the app's `LocalState` folder, read at startup. Upload
or delete them through Device Portal, then restart the app.

| File                              | Effect                                                                            |
| --------------------------------- | --------------------------------------------------------------------------------- |
| `d3d12-fa-off.txt`                | Attention back on the CPU                                                         |
| `d3d12-ew-off.txt`                | Small ops and GPU quantization back on the CPU                                    |
| `d3d12-ew-ops.txt`                | Only the listed small ops on the GPU: any of `add mul silu swiglu rms_norm quant` |
| `d3d12-tile-off.txt`              | Single-column matmul kernels for prefill                                          |
| `d3d12-dispatch-timing.txt` = `1` | Per-dispatch GPU timing (adds barriers; diagnostic only)                          |

`POST /v1/debug/profile {"on":true}` turns on per-op CPU/GPU timing for the
next requests; `{"on":false}` turns it off. `GET /v1/debug/log` returns the last
lines of `xllama.log`.

## Measured on the Series X

Big Queef+ (Qwen3.5-9B IQ4_XS, 4.04 GB) with a q8_0 KV cache, Game designation.

| Build                      | Prompt read                        | Writing                                 | Notes                                      |
| -------------------------- | ---------------------------------- | --------------------------------------- | ------------------------------------------ |
| 1.10.16 (before)           | ~44 tok/s; 5,675 tokens took 239 s | ~12 tok/s fresh, ~8 tok/s in long chats | Attention on the CPU                       |
| 1.10.17 (GPU attention)    | 2,297 tokens in 30.1 s (76 tok/s)  | ~11.6 tok/s at 2.3k tokens              |                                            |
| 1.10.25 (small ops on GPU) | 2,090 tokens in 23.2 s (90 tok/s)  | ~11.4 tok/s at 2.1k tokens              | 177 GPU handovers per token, down from 237 |

The op profile that chose this work (build 1.10.16, 6,304-token prompt):
attention was 61.6 % of prefill time and 51.5 % of each decoded token; the
delta-net `CONCAT` was 14.0 % of prefill, on one CPU thread.

Recurrent checkpoint: 50.3 MiB, saved in 11–12 ms. On a follow-up it restored
4,893 tokens and read 33 new ones in 0.5 s, where a full re-read of 4,926 tokens
takes about 66 s.

Memory with Big Queef+ loaded: 4,365 MB working set after the context is made,
against the 5,120 MB Game limit.

## Faults found and fixed

| Symptom                                                        | Cause                                                                                                                                     | Fixed in                                     |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| Every successful code fix ran twice ("It repeated itself")     | The repeat guard compared a corrected page with the previous one                                                                          | 1.10.18                                      |
| Empty or one-word replies with small ops on                    | A quantized input was reused after the graph allocator had put a different tensor at the same address                                     | 1.10.22                                      |
| "malformed request body" on Big Queef+                         | The API's catch-all hid an access violation                                                                                               | 1.10.23 (now logged); cause fixed in 1.10.25 |
| Access violation at `MUL 'attn_norm-0'`                        | A submission of small ops only ran before any matmul had created the activation buffer; happens when the embedding lookup runs on the CPU | 1.10.25                                      |
| A 50 MB checkpoint allocation could fail near the memory limit | No room left under 5 GB                                                                                                                   | 1.10.23 skips the checkpoint instead         |

## Open items

- **Writing speed is now GPU-bound** (about 54 ms GPU per token of about 87 ms
  at 2k tokens). Attention at decode uses one GPU group per head, 16 in all. A
  split-K pass and its combine kernel are in `ggml_d3d12_fa.hlsl` with a tested
  emulation (`d3d12_fa_split_len`, `d3d12_fa_emulate`), but the backend does not
  use them yet.
- The GPU reads `CUSTOM` (CPU-visible) heaps at about 0.4× the speed of
  `DEFAULT` heaps (Series S measurement, [gguf-gpu-decode.md](gguf-gpu-decode.md)
  D1c). Attention and small ops read the KV cache and activations from such heaps.
- A request sent while a model is still loading can fail; the web app retries,
  scripts must too.
- `max_tokens` under 32 answers "the conversation is full".
- No GPU watchdog: a stuck shader holds the GPU at about 100 % until the app is
  stopped. The 60 s submission timeout reports it but cannot free the GPU.

## Build, install and test loop

CI on `windows-2022` builds the `xllama-appx-llamacpp` package (about 8–10
minutes). Install the `.msix` and its `.cer` through Device Portal, then set
App type to Game in Dev Home. Host tests run on Linux:
`cmake --build <dir> --target xllama-tests` then `xllama-tests`.

Device Portal calls that drive the console from a script (upload a switch file,
restart the app, read performance) are in [device-portal.md](device-portal.md).
What we learned about running another operating system on the console is in
[research/xbox-other-os.md](research/xbox-other-os.md).
