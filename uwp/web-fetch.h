// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Fork addition: fetch web-search results on the console (see
// include/xllama/web_search.h). Blocking — call from a worker thread, never
// the UI thread.
#pragma once

#include <string>

namespace xllama::uwp {

// The user turn to send the model for "/web <query>": numbered Bing results
// followed by the question, or a short failure note plus the question.
std::string web_search_prompt(const std::string& query);

} // namespace xllama::uwp
