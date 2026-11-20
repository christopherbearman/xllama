// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Fork additions: LCM scheduler (vs diffusers LCMScheduler, LCM_Dreamshaper_v7
// config) and the web-search helpers.

#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <vector>

#include "xllama/chat_prompt.h"
#include "xllama/diffusion/lcm_scheduler.h"
#include "xllama/web_search.h"

using xllama::diffusion::LcmScheduler;

TEST_CASE("LCM scheduler timesteps match diffusers") {
    const std::vector<std::pair<int, std::vector<double>>> want = {
        {1, {999}},
        {2, {999, 499}},
        {4, {999, 759, 499, 259}},
        {8, {999, 879, 759, 639, 499, 379, 259, 139}},
    };
    for (const auto& [n, ts] : want) {
        LcmScheduler s;
        s.set_timesteps(n);
        REQUIRE(s.timesteps().size() == ts.size());
        for (size_t i = 0; i < ts.size(); ++i)
            CHECK(s.timesteps()[i] == ts[i]);
    }
}

TEST_CASE("LCM scheduler 4-step trajectory matches diffusers") {
    std::vector<double> x = {
        -2.310411800234176,  -0.3732508612577643, -1.0608166785462863, 0.9995093547811761,
        -0.8840250045832653, -1.2755469302148053, -0.623224579866079,  -0.8664416019125772,
        -1.2956271403277857, 1.52363162310635,    0.3236605664983738,  2.0177260314861085,
        1.1357423400213507,  -1.226881339001083,  0.07138847120511693, 0.3380174030507378};
    const std::vector<std::vector<double>> eps = {
        {0.15351883531544036, -0.6332746088675096, -1.2609246557194385, -0.726951062065411,
         -0.019964750907762653, 0.2102998901016947, 0.17718935139366584, -0.830510051008449,
         1.0111892074660187, -0.2426793824588577, -0.7730112772958181, -1.5951814880783068,
         -0.6870036940435463, 1.4880743158106724, -0.4484158047114511, -0.8910030351707375},
        {-0.09174174090816388, 0.5563227724483105, -0.009446277892507904, -1.21768777910687,
         -1.2654653080284903, -1.1195351830271254, 1.1664857205254855, 0.9262324572968778,
         0.6539244967350809, 0.14821544548810084, -1.1461092231976013, 1.8727763768553416,
         -0.30991497424174674, 0.24018023403782535, -1.3487610270963772, 0.2443990933828689},
        {-3.1452962731231335, -0.11373127757597257, 1.6962024805143952, -0.6652034805381666,
         -0.587222598359879, 2.832104167511146, 0.9039973689601122, 0.947836532432321,
         -1.3809366518587674, -2.0547510174380235, 0.04757655385024031, -0.3855687975933702,
         -0.27503491919216155, 0.886589630091246, 1.4857416299332185, -0.007855617400547969},
        {-1.1287986397999445, -0.09376858138313218, 0.998343579451126, -0.08262167465143246,
         0.1825319491801699, 0.9509038124935927, -1.7631617835191395, 1.0107316682629284,
         0.7494883300195011, -0.560555214098722, 0.07424681473465156, 1.9936457271364452,
         0.44986533984459376, 1.5531453898519594, 0.05398594882285299, 1.4744415042312724},
    };
    const std::vector<std::vector<double>> noise = {
        {-0.7012303945657874, 0.326255796287562, -0.8738640219900763, 0.3941269998738486,
         0.8915170821381018, 1.1731555158319225, -0.39490340565105597, 0.44642257211843217,
         -0.1726536808458421, 0.975120345181078, 0.4164541966117019, -0.4986513835421579,
         -0.4754698199105609, 0.28634406331767814, -1.3181871188381729, 0.8887144617352294},
        {-0.5918823592164963, 0.6195092629710962, -1.4872950971304264, 1.3626823808042956,
         0.4188191254194787, 0.34341487809757076, -0.34859980650048067, -1.2090699446332276,
         0.20710026569896173, -0.11727393443191467, -0.017436740784596904, -0.24317130976865137,
         -0.43942236360685477, 2.7144660166018406, -1.4631736798004056, 0.4357325629861231},
        {-0.5312331855486973, -0.13408449243234943, 1.3642724524235599, 0.2327705351743837,
         1.1064458833206277, 0.8527354795918417, -0.6301719953003105, -0.6019872001874332,
         1.114919051942112, -0.5559898033655196, 0.5331687087503058, -0.18787474948239727,
         0.3993685890598303, -0.6760874461803966, 1.662168703751691, -0.6036839371988663},
        {0.0534191942795709, 0.013427502769270348, -1.5406177939388193, -0.4819087554044004,
         0.7623209434576803, 0.21679229361763427, -0.4789120616253859, -0.2578760451372613,
         0.2631423185835337, -0.4053288789025133, -0.0727123537950913, -0.13985617336865128,
         0.785995459350369, -2.0568571650881626, -0.6711099455038106, -0.40096823326862124},
    };
    const std::vector<double> want = {
        -34.13694772876998,  3.960818621424763,  -5.666025093951322,  35.630635705735216,
        -1.1809181925050358, -16.07516769414787, -19.576702260922158, -7.237997468766395,
        -34.45506024507692,  32.517371893793616, 22.921206532784574,  41.4270426998263,
        25.670781360574082,  -38.17887004828843, 4.130085385269155,   19.93889088304628};
    LcmScheduler s;
    s.set_timesteps(4);
    for (int i = 0; i < 4; ++i)
        s.step(eps[i], x, noise[i]);
    // diffusers keeps alphas_cumprod in float32; this header uses double.
    for (size_t i = 0; i < x.size(); ++i)
        CHECK(x[i] == doctest::Approx(want[i]).epsilon(1e-4));
}

TEST_CASE("LCM guidance embedding matches diffusers (w = 7)") {
    const auto e = LcmScheduler::guidance_embedding(8.0);
    REQUIRE(e.size() == 256);
    CHECK(std::fabs(e[0] - (0.5068854093551636)) < 2e-3);
    CHECK(std::fabs(e[1] - (0.8042871356010437)) < 2e-3);
    CHECK(std::fabs(e[10] - (0.2617175579071045)) < 2e-3);
    CHECK(std::fabs(e[64] - (-0.9993135333061218)) < 2e-3);
    CHECK(std::fabs(e[127] - (0.6442176103591919)) < 2e-3);
    CHECK(std::fabs(e[128] - (0.8620134592056274)) < 2e-3);
    CHECK(std::fabs(e[129] - (0.5942408442497253)) < 2e-3);
    CHECK(std::fabs(e[200] - (0.9958349466323853)) < 2e-3);
    CHECK(std::fabs(e[255] - (0.7648422122001648)) < 2e-3);
}

TEST_CASE("web command parsing") {
    std::string q;
    CHECK(xllama::parse_web_command("/web latest bootstrap version", &q));
    CHECK(q == "latest bootstrap version");
    CHECK(xllama::parse_web_command("  /SEARCH   css grid  ", &q));
    CHECK(q == "css grid");
    CHECK_FALSE(xllama::parse_web_command("/website ideas", &q));
    CHECK_FALSE(xllama::parse_web_command("/web   ", &q));
    CHECK_FALSE(xllama::parse_web_command("how do I use /web", &q));
    CHECK(xllama::url_encode("a b&c") == "a+b%26c");
}

TEST_CASE("Bing RSS parsing") {
    const std::string xml =
        "<rss><channel><title>Bing: q</title><item><title>Bootstrap &#183; Docs</title>"
        "<link>https://getbootstrap.com/</link><description>Build fast &amp; responsive "
        "sites.</description></item><item><title>Second</title><link>https://example.com/"
        "</link><description>&lt;b&gt;bold&lt;/b&gt; text</description></item></channel></rss>";
    const auto rs = xllama::parse_bing_rss(xml, 5);
    REQUIRE(rs.size() == 2);
    CHECK(rs[0].title == "Bootstrap \xC2\xB7 Docs");
    CHECK(rs[0].url == "https://getbootstrap.com/");
    CHECK(rs[0].snippet == "Build fast & responsive sites.");
    CHECK(rs[1].snippet == "<b>bold</b> text");
    const std::string p = xllama::build_web_prompt("q", rs);
    CHECK(p.find("[1] Bootstrap") != std::string::npos);
    CHECK(p.find("Question: q") != std::string::npos);
    CHECK(xllama::build_web_prompt("q", {}).find("no results") != std::string::npos);
}

TEST_CASE("web search key parsing and Tavily request body") {
    CHECK(xllama::parse_web_search_key("tvly-abc123\n").provider == "tavily");
    CHECK(xllama::parse_web_search_key("tvly-abc123\n").key == "tvly-abc123");
    CHECK(xllama::parse_web_search_key("brave: BSAkey ").provider == "brave");
    CHECK(xllama::parse_web_search_key("brave: BSAkey ").key == "BSAkey");
    CHECK(xllama::parse_web_search_key("").provider == "bing");
    CHECK(xllama::parse_web_search_key("tavily:").provider == "bing");
    CHECK(xllama::tavily_request_json("say \"hi\"\n", 5) ==
          "{\"query\":\"say \\\"hi\\\"\\n\",\"max_results\":5,\"search_depth\":\"basic\","
          "\"include_answer\":false}");
}

TEST_CASE("image command and data-URI image stripping") {
    std::string p;
    CHECK(xllama::parse_image_command("/image a red bus", &p));
    CHECK(p == "a red bus");
    CHECK(xllama::parse_image_command("/IMG cat", &p));
    CHECK_FALSE(xllama::parse_image_command("/imagery of cats", &p));
    CHECK_FALSE(xllama::parse_web_command("/image a red bus", &p));
    CHECK(xllama::strip_data_uri_images("before ![bus](data:image/png;base64,AAAA) after") ==
          "before [image] after");
    CHECK(xllama::strip_data_uri_images("keep ![x](https://e.com/a.png) and ![y](data:,1)") ==
          "keep ![x](https://e.com/a.png) and [image]");
    CHECK(xllama::strip_data_uri_images("no images here") == "no images here");
}

TEST_CASE("Qwen3 no-think history replays the empty think block (KV prefix reuse)") {
    const auto q = xllama::chat_format_for("qwen35-4b");
    REQUIRE(q.history_keeps_gen_suffix);
    std::vector<xllama::ChatTurn> hist;
    xllama::ChatTurn t;
    t.user = "hi";
    t.assistant = "Hello!";
    hist.push_back(t);
    // Turn 1 prompt + generated reply + stop must be a prefix of the turn-2 prompt.
    const std::string turn1 = q.render_prompt("sys", {}, "hi") + "Hello!<|im_end|>";
    const std::string turn2 = q.render_prompt("sys", hist, "more");
    CHECK(turn2.compare(0, turn1.size(), turn1) == 0);
    // Formats without a suffix and thinking models are unchanged.
    CHECK_FALSE(xllama::chat_format_for("phi4-mini").history_keeps_gen_suffix);
    CHECK_FALSE(xllama::chat_format_for("qwen25-coder-1.5b").history_keeps_gen_suffix);
    CHECK_FALSE(xllama::chat_format_for("lfm25-1.2b-thinking").history_keeps_gen_suffix);
}

#include "xllama/think_mode.h"
TEST_CASE("thinking prefill replaces the Qwen3 no-think suffix") {
    const std::string suffix = "<think>\n\n</think>\n\n";
    std::string p = "<|im_start|>assistant\n" + suffix;
    CHECK(xllama::apply_thinking_prefill(p, suffix));
    CHECK(p == "<|im_start|>assistant\n<think>\n");
    std::string q = "<|im_start|>assistant\n";
    CHECK_FALSE(xllama::apply_thinking_prefill(q, ""));        // not Qwen3
    CHECK_FALSE(xllama::apply_thinking_prefill(q, suffix));    // no suffix present
    CHECK(q == "<|im_start|>assistant\n");
}
