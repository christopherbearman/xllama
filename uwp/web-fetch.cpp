// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#ifdef XLLAMA_UWP

// clang-format off
#include "pch.h"
// clang-format on

    #include "web-fetch.h"
    #include "xllama/platform.h"
    #include "xllama/web_search.h"

namespace xllama::uwp {

std::string web_search_prompt(const std::string& query) {
    using namespace winrt::Windows::Web::Http;
    try {
        Filters::HttpBaseProtocolFilter filter;
        filter.AllowAutoRedirect(true);
        HttpClient client(filter);
        client.DefaultRequestHeaders().UserAgent().TryParseAdd(
            L"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like "
            L"Gecko) Chrome/130.0 Safari/537.36");
        winrt::Windows::Foundation::Uri uri(winrt::to_hstring(::xllama::bing_rss_url(query)));
        const std::string xml = winrt::to_string(client.GetStringAsync(uri).get());
        const auto results = ::xllama::parse_bing_rss(xml, 5);
        ::xllama::log_output("[xllama] web: '" + query + "' -> " + std::to_string(results.size()) +
                             " results\n");
        return ::xllama::build_web_prompt(query, results);
    } catch (winrt::hresult_error const& e) {
        ::xllama::log_output("[xllama] web: search failed: " + winrt::to_string(e.message()) +
                             "\n");
    } catch (const std::exception& e) {
        ::xllama::log_output(std::string("[xllama] web: search failed: ") + e.what() + "\n");
    } catch (...) {
        ::xllama::log_output("[xllama] web: search failed (unknown error)\n");
    }
    return "(The web search for \"" + query +
           "\" failed - the console could not reach the search service. Say so, then answer "
           "as best you can.)\n\nQuestion: " +
           query;
}

} // namespace xllama::uwp

#endif // XLLAMA_UWP
