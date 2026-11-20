#include "xllama/op_profile.h"

#ifdef XLLAMA_USE_LLAMA

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <vector>

namespace xllama {
namespace {
struct Acc {
    double ms = 0.0;
    long long n = 0;
};
std::mutex g_mu;
std::map<std::string, Acc> g_acc;
std::chrono::steady_clock::time_point g_t0;

std::string key_for(const ggml_tensor* t) {
    std::string k = ggml_op_desc(t);
    if (t->op == GGML_OP_MUL_MAT && t->src[0])
        k += std::string(":") + ggml_type_name(t->src[0]->type) +
             (t->src[1] && t->src[1]->ne[1] > 1 ? "x" + std::to_string(t->src[1]->ne[1]) : "");
    if (t->buffer) {
        ggml_backend_buffer_type_t bt = ggml_backend_buffer_get_type(t->buffer);
        if (bt && ggml_backend_buft_name(bt))
            k += std::string(" @") + ggml_backend_buft_name(bt);
    }
    return k;
}
} // namespace

std::atomic<bool>& op_profile_enabled() {
    static std::atomic<bool> on{false};
    return on;
}

bool op_profile_cb(ggml_tensor* t, bool ask, void*) {
    if (ask) {
        // Observe every node: each is computed on its own between the two calls.
        g_t0 = std::chrono::steady_clock::now();
        return true;
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g_t0).count();
    std::lock_guard<std::mutex> lk(g_mu);
    Acc& a = g_acc[key_for(t)];
    a.ms += ms;
    ++a.n;
    return true; // keep computing
}

std::string op_profile_take(const char* phase) {
    std::vector<std::pair<std::string, Acc>> v;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        v.assign(g_acc.begin(), g_acc.end());
        g_acc.clear();
    }
    if (v.empty())
        return {};
    double total = 0.0;
    for (const auto& e : v)
        total += e.second.ms;
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    std::string out;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "[xllama] op-profile phase=%s total_ms=%.1f ops=%zu\n", phase,
                  total, v.size());
    out += buf;
    for (size_t i = 0; i < v.size() && i < 25; ++i) {
        std::snprintf(buf, sizeof(buf), "[xllama] op-profile %5.1f%% %9.1f ms %7lld x  %s\n",
                      total > 0 ? 100.0 * v[i].second.ms / total : 0.0, v[i].second.ms,
                      v[i].second.n, v[i].first.c_str());
        out += buf;
    }
    return out;
}

} // namespace xllama

#else // no llama.cpp in this build: profiling is a no-op

namespace xllama {
std::atomic<bool>& op_profile_enabled() {
    static std::atomic<bool> on{false};
    return on;
}
bool op_profile_cb(ggml_tensor*, bool, void*) {
    return true;
}
std::string op_profile_take(const char*) {
    return {};
}
} // namespace xllama

#endif
