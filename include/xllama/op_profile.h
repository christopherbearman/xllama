// Fork: per-op-type timing of the llama graph (which CPU work dominates).
// Off by default. POST /v1/debug/profile {"on":true} turns it on and unloads the
// resident model; the next session installs an eval callback that observes
// every node, so each op is computed (and timed) on its own. Results are logged
// after each prefill and decode as the top op types by time.
#pragma once

#include <atomic>
#include <string>

struct ggml_tensor;

namespace xllama {

std::atomic<bool>& op_profile_enabled();

// ggml_backend_sched_eval_callback-compatible.
bool op_profile_cb(ggml_tensor* t, bool ask, void* user_data);

// Clears the counters; returns the "[xllama] op-profile ..." report of what was
// recorded since the last reset (empty when nothing was).
std::string op_profile_take(const char* phase);

} // namespace xllama
