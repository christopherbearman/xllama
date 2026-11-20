// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#ifdef XLLAMA_UWP

// clang-format off
#include "pch.h"
// clang-format on

    #include "web-fetch.h"
    #include "xllama/path_utils.h"
    #include "xllama/platform.h"
    #include "xllama/web_search.h"

    #include <winrt/Windows.Data.Json.h>
    #include <winrt/Windows.Storage.Streams.h>

    #include <cstdio>
    #include <stdexcept>

namespace xllama::uwp {

namespace {

using namespace winrt::Windows::Web::Http;
using namespace winrt::Windows::Data::Json;

constexpr const char* kKeyFile = "web-search-key.txt";

std::string read_key_file() {
    std::string out;
    FILE* f = _wfopen(winrt::to_hstring(::xllama::resolve_local_path(kKeyFile)).c_str(), L"rb");
    if (!f)
        return out;
    char buf[512];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        out.append(buf, n);
    fclose(f);
    return out;
}

HttpClient make_client() {
    Filters::HttpBaseProtocolFilter filter;
    filter.AllowAutoRedirect(true);
    HttpClient client(filter);
    client.DefaultRequestHeaders().UserAgent().TryParseAdd(
        L"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like "
        L"Gecko) Chrome/130.0 Safari/537.36");
    return client;
}

std::string str_field(JsonObject const& o, const wchar_t* key) {
    if (!o.HasKey(key) || o.GetNamedValue(key).ValueType() != JsonValueType::String)
        return {};
    return winrt::to_string(o.GetNamedString(key));
}

// Tavily: POST https://api.tavily.com/search, Bearer key -> results[].{title,url,content}
std::vector<WebResult> search_tavily(const std::string& query, const std::string& key) {
    HttpClient client = make_client();
    HttpRequestMessage req(HttpMethod::Post(),
                           winrt::Windows::Foundation::Uri(L"https://api.tavily.com/search"));
    req.Headers().Authorization(
        Headers::HttpCredentialsHeaderValue(L"Bearer", winrt::to_hstring(key)));
    req.Content(HttpStringContent(winrt::to_hstring(::xllama::tavily_request_json(query, 5)),
                                  winrt::Windows::Storage::Streams::UnicodeEncoding::Utf8,
                                  L"application/json"));
    auto resp = client.SendRequestAsync(req).get();
    const std::wstring body(resp.Content().ReadAsStringAsync().get().c_str());
    if (!resp.IsSuccessStatusCode())
        throw std::runtime_error("tavily HTTP " + std::to_string((int)resp.StatusCode()) + ": " +
                                 winrt::to_string(body).substr(0, 200));
    std::vector<WebResult> out;
    JsonObject root = JsonObject::Parse(body);
    if (!root.HasKey(L"results"))
        return out;
    for (auto const& v : root.GetNamedArray(L"results")) {
        if (v.ValueType() != JsonValueType::Object)
            continue;
        auto o = v.GetObject();
        out.push_back({str_field(o, L"title"), str_field(o, L"url"), str_field(o, L"content")});
    }
    return out;
}

// Brave: GET api.search.brave.com/res/v1/web/search, X-Subscription-Token ->
// web.results[].{title,url,description}
std::vector<WebResult> search_brave(const std::string& query, const std::string& key) {
    HttpClient client = make_client();
    HttpRequestMessage req(HttpMethod::Get(),
                           winrt::Windows::Foundation::Uri(winrt::to_hstring(
                               "https://api.search.brave.com/res/v1/web/search?count=5&q=" +
                               ::xllama::url_encode(query))));
    req.Headers().TryAppendWithoutValidation(L"X-Subscription-Token", winrt::to_hstring(key));
    req.Headers().TryAppendWithoutValidation(L"Accept", L"application/json");
    auto resp = client.SendRequestAsync(req).get();
    const std::wstring body(resp.Content().ReadAsStringAsync().get().c_str());
    if (!resp.IsSuccessStatusCode())
        throw std::runtime_error("brave HTTP " + std::to_string((int)resp.StatusCode()));
    std::vector<WebResult> out;
    JsonObject root = JsonObject::Parse(body);
    if (!root.HasKey(L"web"))
        return out;
    auto web = root.GetNamedObject(L"web");
    if (!web.HasKey(L"results"))
        return out;
    for (auto const& v : web.GetNamedArray(L"results")) {
        if (v.ValueType() != JsonValueType::Object)
            continue;
        auto o = v.GetObject();
        out.push_back({str_field(o, L"title"), str_field(o, L"url"),
                       ::xllama::decode_xml_text(str_field(o, L"description"))});
        if (out.size() >= 5)
            break;
    }
    return out;
}

// Keyless fallback: Bing RSS. Result quality is poor for natural-language
// questions (measured 2026-11-20); configure a key for real use.
std::vector<WebResult> search_bing_rss(const std::string& query) {
    HttpClient client = make_client();
    winrt::Windows::Foundation::Uri uri(winrt::to_hstring(::xllama::bing_rss_url(query)));
    return ::xllama::parse_bing_rss(winrt::to_string(client.GetStringAsync(uri).get()), 5);
}

} // namespace

std::string web_search_provider() {
    return ::xllama::parse_web_search_key(read_key_file()).provider;
}

bool save_web_search_key(const std::string& text) {
    const auto cfg = ::xllama::parse_web_search_key(text);
    FILE* f = _wfopen(winrt::to_hstring(::xllama::resolve_local_path(kKeyFile)).c_str(), L"wb");
    if (!f)
        return false;
    const std::string line = cfg.provider == "bing" ? std::string() : cfg.provider + ":" + cfg.key;
    fwrite(line.data(), 1, line.size(), f);
    fclose(f);
    return true;
}

std::string web_search_prompt(const std::string& query) {
    const auto cfg = ::xllama::parse_web_search_key(read_key_file());
    try {
        std::vector<WebResult> results;
        if (cfg.provider == "tavily")
            results = search_tavily(query, cfg.key);
        else if (cfg.provider == "brave")
            results = search_brave(query, cfg.key);
        else
            results = search_bing_rss(query);
        ::xllama::log_output("[xllama] web(" + cfg.provider + "): '" + query + "' -> " +
                             std::to_string(results.size()) + " results\n");
        return ::xllama::build_web_prompt(query, results);
    } catch (winrt::hresult_error const& e) {
        ::xllama::log_output("[xllama] web(" + cfg.provider +
                             "): search failed: " + winrt::to_string(e.message()) + "\n");
    } catch (const std::exception& e) {
        ::xllama::log_output("[xllama] web(" + cfg.provider + "): search failed: " + e.what() +
                             "\n");
    } catch (...) {
        ::xllama::log_output("[xllama] web(" + cfg.provider + "): search failed (unknown)\n");
    }
    return "(The web search for \"" + query +
           "\" failed - the console could not reach the search service. Say so, then answer "
           "as best you can.)\n\nQuestion: " +
           query;
}

} // namespace xllama::uwp

#endif // XLLAMA_UWP
