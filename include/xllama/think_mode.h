#pragma once
// Fork addition: per-request thinking for Qwen3.x on the LAN API.
//
// Qwen3.x normally gets an empty "<think></think>" prefill (no-think mode) so
// it answers straight away. A request can ask for thinking instead; the
// prompt's trailing prefill is then swapped for an open "<think>\n" so the
// model plans first. The reply streams as "<think>\n…plan…</think>\n\nanswer",
// the form Open WebUI and the built-in web page both fold away.
#include <string>

namespace xllama {

constexpr char kThinkOpenPrefill[] = "<think>\n";

// Replaces a trailing no-think gen_suffix with an open think tag. Returns
// false (prompt untouched) when the format has no gen_suffix (not Qwen3) or
// the prompt does not end with it.
inline bool apply_thinking_prefill(std::string& prompt, const std::string& gen_suffix) {
    if (gen_suffix.empty() || prompt.size() < gen_suffix.size())
        return false;
    if (prompt.compare(prompt.size() - gen_suffix.size(), gen_suffix.size(), gen_suffix) != 0)
        return false;
    prompt.resize(prompt.size() - gen_suffix.size());
    prompt += kThinkOpenPrefill;
    return true;
}

} // namespace xllama
