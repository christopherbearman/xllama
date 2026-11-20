// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Web search helpers (fork addition, experimental). WinRT-free and host-tested
// (tests/test_web_search.cpp); the UWP side only fetches the URL.
//
// A chat message that starts with "/web " (or "/search ") is a web-search
// request: the console fetches Bing's RSS results for the rest of the message
// and the model answers from a short numbered context instead of the bare
// question. Bing RSS is used because it needs no API key and is structured
// (title, link, description per <item>); its feed terms allow personal,
// non-commercial use.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace xllama {

struct WebResult {
    std::string title;
    std::string url;
    std::string snippet;
};

// "/web <query>" or "/search <query>" (case-insensitive, leading spaces ok).
// Returns true and sets |query| (trimmed, non-empty) on a match.
inline bool parse_web_command(const std::string& text, std::string* query) {
    size_t i = 0;
    while (i < text.size() && std::isspace((unsigned char)text[i]))
        ++i;
    for (const char* cmd : {"/web", "/search"}) {
        const size_t n = std::char_traits<char>::length(cmd);
        if (text.size() < i + n)
            continue;
        bool match = true;
        for (size_t k = 0; k < n; ++k)
            if (std::tolower((unsigned char)text[i + k]) != cmd[k])
                match = false;
        if (!match)
            continue;
        size_t j = i + n;
        if (j < text.size() && !std::isspace((unsigned char)text[j]))
            continue; // "/website" is not the command
        while (j < text.size() && std::isspace((unsigned char)text[j]))
            ++j;
        std::string q = text.substr(j);
        while (!q.empty() && std::isspace((unsigned char)q.back()))
            q.pop_back();
        if (q.empty())
            return false;
        if (query)
            *query = q;
        return true;
    }
    return false;
}

inline std::string url_encode(const std::string& s) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out.push_back((char)c);
        } else if (c == ' ') {
            out.push_back('+');
        } else {
            out.push_back('%');
            out.push_back(hex[c >> 4]);
            out.push_back(hex[c & 15]);
        }
    }
    return out;
}

inline std::string bing_rss_url(const std::string& query) {
    return "https://www.bing.com/search?format=rss&q=" + url_encode(query);
}

// Decode the XML/HTML entities that occur in RSS text, and strip tags.
inline std::string decode_xml_text(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        const char c = in[i];
        if (c == '<') { // drop inline tags (e.g. <strong>)
            const size_t e = in.find('>', i);
            if (e == std::string::npos)
                break;
            i = e;
            continue;
        }
        if (c != '&') {
            out.push_back(c);
            continue;
        }
        const size_t semi = in.find(';', i);
        if (semi == std::string::npos || semi - i > 10) {
            out.push_back(c);
            continue;
        }
        const std::string ent = in.substr(i + 1, semi - i - 1);
        unsigned long cp = 0;
        bool ok = true;
        if (ent == "amp")
            cp = '&';
        else if (ent == "lt")
            cp = '<';
        else if (ent == "gt")
            cp = '>';
        else if (ent == "quot")
            cp = '"';
        else if (ent == "apos")
            cp = '\'';
        else if (ent.size() > 1 && ent[0] == '#')
            cp = (ent[1] == 'x' || ent[1] == 'X') ? std::strtoul(ent.c_str() + 2, nullptr, 16)
                                                  : std::strtoul(ent.c_str() + 1, nullptr, 10);
        else
            ok = false;
        if (!ok || cp == 0) {
            out.push_back(c);
            continue;
        }
        // UTF-8 encode the code point.
        if (cp < 0x80) {
            out.push_back((char)cp);
        } else if (cp < 0x800) {
            out.push_back((char)(0xC0 | (cp >> 6)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char)(0xE0 | (cp >> 12)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char)(0xF0 | (cp >> 18)));
            out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        }
        i = semi;
    }
    return out;
}

// Text between the first <tag> and </tag> inside [from, to). Empty if absent.
inline std::string xml_child(const std::string& xml, size_t from, size_t to, const char* tag) {
    const std::string open = std::string("<") + tag + ">";
    const std::string close = std::string("</") + tag + ">";
    const size_t a = xml.find(open, from);
    if (a == std::string::npos || a >= to)
        return {};
    const size_t b = xml.find(close, a);
    if (b == std::string::npos || b > to)
        return {};
    std::string v = xml.substr(a + open.size(), b - a - open.size());
    // CDATA sections appear in some feeds.
    if (v.rfind("<![CDATA[", 0) == 0 && v.size() >= 12 && v.compare(v.size() - 3, 3, "]]>") == 0)
        return v.substr(9, v.size() - 12);
    return decode_xml_text(v);
}

inline std::vector<WebResult> parse_bing_rss(const std::string& xml, size_t max_results = 5) {
    std::vector<WebResult> out;
    size_t pos = 0;
    while (out.size() < max_results) {
        const size_t a = xml.find("<item>", pos);
        if (a == std::string::npos)
            break;
        const size_t b = xml.find("</item>", a);
        if (b == std::string::npos)
            break;
        WebResult r;
        r.title = xml_child(xml, a, b, "title");
        r.url = xml_child(xml, a, b, "link");
        r.snippet = xml_child(xml, a, b, "description");
        if (!r.title.empty() || !r.snippet.empty())
            out.push_back(std::move(r));
        pos = b + 7;
    }
    return out;
}

// Search provider from the stored key text: "tavily:<key>" or a bare
// "tvly-..." key -> Tavily; "brave:<key>" -> Brave; anything else -> keyless
// Bing RSS fallback.
struct WebSearchConfig {
    std::string provider = "bing";
    std::string key;
};

inline WebSearchConfig parse_web_search_key(std::string text) {
    auto trim = [](std::string s) {
        while (!s.empty() && std::isspace((unsigned char)s.back()))
            s.pop_back();
        size_t i = 0;
        while (i < s.size() && std::isspace((unsigned char)s[i]))
            ++i;
        return s.substr(i);
    };
    text = trim(text);
    WebSearchConfig c;
    if (text.rfind("tavily:", 0) == 0) {
        c.provider = "tavily";
        c.key = trim(text.substr(7));
    } else if (text.rfind("brave:", 0) == 0) {
        c.provider = "brave";
        c.key = trim(text.substr(6));
    } else if (text.rfind("tvly-", 0) == 0) {
        c.provider = "tavily";
        c.key = text;
    }
    if (c.key.empty())
        c.provider = "bing";
    return c;
}

// Minimal JSON string escape for request bodies.
inline std::string json_escape(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20) {
                char b[8];
                std::snprintf(b, sizeof(b), "\\u%04x", c);
                out += b;
            } else {
                out.push_back((char)c);
            }
        }
    }
    return out;
}

inline std::string tavily_request_json(const std::string& query, int max_results) {
    return "{\"query\":\"" + json_escape(query) +
           "\",\"max_results\":" + std::to_string(max_results) +
           ",\"search_depth\":\"basic\",\"include_answer\":false}";
}

// The user turn the model sees: numbered sources, then the question. Each
// snippet is capped so five results stay well inside a 2048-token context.
inline std::string build_web_prompt(const std::string& query, const std::vector<WebResult>& rs,
                                    size_t max_snippet_chars = 400) {
    if (rs.empty())
        return "(A web search for \"" + query +
               "\" returned no results. Say so, then answer as best you can.)\n\n" + query;
    std::string s = "Web search results for \"" + query + "\":\n\n";
    for (size_t i = 0; i < rs.size(); ++i) {
        std::string snip = rs[i].snippet;
        if (snip.size() > max_snippet_chars)
            snip = snip.substr(0, max_snippet_chars) + "...";
        s += "[" + std::to_string(i + 1) + "] " + rs[i].title + "\n" + rs[i].url + "\n" + snip +
             "\n\n";
    }
    s += "Using these results, answer the question below. Cite sources as [1], [2]. If the "
         "results do not contain the answer, say so.\n\nQuestion: " +
         query;
    return s;
}

} // namespace xllama
