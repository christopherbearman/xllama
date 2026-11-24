// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

// Built only with the llama.cpp backend (it links ggml); the ORT-only UWP
// variant compiles this file to nothing.
#ifdef XLLAMA_USE_LLAMA

    #include "xllama/ggml_d3d12.h"

    #include "ggml-cpu.h"

    #include <algorithm>
    #include <cctype>
    #include <cmath>
    #include <cstdio>
    #include <cstring>
    #include <ctime>
    #include <limits>
    #include <optional>
    #include <string>
    #include <string_view>
    #include <vector>

namespace xllama {

// --- Pure rules (host-tested) ---

bool d3d12_weight_type_supported(ggml_type t) {
    return t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q6_K ||
           t == GGML_TYPE_IQ3_XXS || t == GGML_TYPE_IQ4_XS || // fork: Qwen3.5-9B IQ quants
           t == GGML_TYPE_Q5_K;                               // fork: Q4_K_M leftovers
}

D3d12Dispatch d3d12_mm_dispatch(std::int64_t n, std::int64_t ncols) {
    D3d12Dispatch d;
    if (n <= 0 || ncols < 0)
        return d;
    if (ncols == 0) { // no output rows in this ubatch: a no-op, still ours
        d.ok = true;
        return d;
    }
    const std::int64_t gx = (n + kD3d12MmvRows - 1) / kD3d12MmvRows;
    d.ok = gx <= kD3d12MaxGroups && ncols <= kD3d12MaxGroups;
    if (d.ok) {
        d.groups_x = static_cast<std::uint32_t>(gx);
        d.groups_y = static_cast<std::uint32_t>(ncols);
    }
    return d;
}

int d3d12_mm_threads(std::int64_t k) {
    return k / kD3d12Chunk >= kD3d12LongKChunks ? kD3d12MmvThreadsLong : kD3d12MmvThreadsShort;
}

bool d3d12_mm_supported(const D3d12MatmulDesc& d) {
    return d3d12_weight_type_supported(d.src0_type) && d.src0_in_weight_buffer &&
           d.src0_contiguous && d.src1_contiguous && d.src1_type == GGML_TYPE_F32 &&
           d.dst_type == GGML_TYPE_F32 && d.ne00 > 0 && d.ne00 % kD3d12Chunk == 0 &&
           d.ne00 == d.ne10 && d.ne02 == 1 && d.ne03 == 1 && d.ne12 == 1 && d.ne13 == 1 &&
           d3d12_mm_dispatch(d.ne01, d.ne11).ok;
}

bool d3d12_get_rows_type_supported(ggml_type t) {
    return t == GGML_TYPE_Q6_K;
}

bool d3d12_get_rows_supported(const D3d12GetRowsDesc& d) {
    return d3d12_get_rows_type_supported(d.src0_type) && d.src0_in_weight_buffer &&
           d.src0_contiguous && d.src1_contiguous && d.src1_type == GGML_TYPE_I32 &&
           d.dst_type == GGML_TYPE_F32 && d.ne00 > 0 && d.ne00 % kD3d12Chunk == 0 && d.ne02 == 1 &&
           d.ne03 == 1 && d.ne10 >= 0 && d.ne10 <= kD3d12MaxGroups && d.ne11 == 1 && d.ne12 == 1 &&
           d.ne00 / kD3d12Chunk <= kD3d12MaxGroups;
}

bool d3d12_place_tied_embedding(ggml_type embd_type, bool has_output_weight) {
    return !has_output_weight && d3d12_get_rows_type_supported(embd_type);
}

// --- Host emulation of shaders/ggml_d3d12_mmv_*.hlsl ---

namespace {

// Little-endian loads with the shaders' 2-byte-aligned dword trick.
std::uint32_t ld32(const std::uint8_t* w, std::uint32_t a) {
    auto dw = [&](std::uint32_t at) {
        std::uint32_t v;
        std::memcpy(&v, w + at, 4);
        return v;
    };
    const std::uint32_t b = a & ~3u;
    const std::uint32_t lo = dw(b);
    if ((a & 3u) == 0u)
        return lo;
    return (lo >> 16) | (dw(b + 4u) << 16);
}

std::uint32_t ld16(const std::uint8_t* w, std::uint32_t a) {
    std::uint32_t v;
    std::memcpy(&v, w + (a & ~3u), 4);
    return (a & 2u) != 0u ? (v >> 16) : (v & 0xffffu);
}

float h2f(std::uint32_t h) {
    return ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(h));
}

std::int32_t nib(std::uint32_t word, std::uint32_t i, std::uint32_t shift) {
    return static_cast<std::int32_t>((word >> (8u * i + shift)) & 0xFu);
}

std::int32_t sbyte(std::uint32_t word, std::uint32_t i) {
    return static_cast<std::int8_t>((word >> (8u * i)) & 0xffu);
}

// iq3xxs_grid from ggml-common.h (also in shaders/ggml_d3d12_mmv_iq3_xxs.hlsl).
const std::uint32_t kIq3xxsGrid[256] = {
    0x04040404u, 0x04040414u, 0x04040424u, 0x04040c0cu, 0x04040c1cu, 0x04040c3eu, 0x04041404u,
    0x04041414u, 0x04041c0cu, 0x04042414u, 0x04043e1cu, 0x04043e2cu, 0x040c040cu, 0x040c041cu,
    0x040c0c04u, 0x040c0c14u, 0x040c140cu, 0x040c142cu, 0x040c1c04u, 0x040c1c14u, 0x040c240cu,
    0x040c2c24u, 0x040c3e04u, 0x04140404u, 0x04140414u, 0x04140424u, 0x04140c0cu, 0x04141404u,
    0x04141414u, 0x04141c0cu, 0x04141c1cu, 0x04141c3eu, 0x04142c0cu, 0x04142c3eu, 0x04143e2cu,
    0x041c040cu, 0x041c043eu, 0x041c0c04u, 0x041c0c14u, 0x041c142cu, 0x041c3e04u, 0x04240c1cu,
    0x04241c3eu, 0x04242424u, 0x04242c3eu, 0x04243e1cu, 0x04243e2cu, 0x042c040cu, 0x042c043eu,
    0x042c1c14u, 0x042c2c14u, 0x04341c2cu, 0x04343424u, 0x043e0c04u, 0x043e0c24u, 0x043e0c34u,
    0x043e241cu, 0x043e340cu, 0x0c04040cu, 0x0c04041cu, 0x0c040c04u, 0x0c040c14u, 0x0c04140cu,
    0x0c04141cu, 0x0c041c04u, 0x0c041c14u, 0x0c041c24u, 0x0c04243eu, 0x0c042c04u, 0x0c0c0404u,
    0x0c0c0414u, 0x0c0c0c0cu, 0x0c0c1404u, 0x0c0c1414u, 0x0c14040cu, 0x0c14041cu, 0x0c140c04u,
    0x0c140c14u, 0x0c14140cu, 0x0c141c04u, 0x0c143e14u, 0x0c1c0404u, 0x0c1c0414u, 0x0c1c1404u,
    0x0c1c1c0cu, 0x0c1c2434u, 0x0c1c3434u, 0x0c24040cu, 0x0c24042cu, 0x0c242c04u, 0x0c2c1404u,
    0x0c2c1424u, 0x0c2c2434u, 0x0c2c3e0cu, 0x0c34042cu, 0x0c3e1414u, 0x0c3e2404u, 0x14040404u,
    0x14040414u, 0x14040c0cu, 0x14040c1cu, 0x14041404u, 0x14041414u, 0x14041434u, 0x14041c0cu,
    0x14042414u, 0x140c040cu, 0x140c041cu, 0x140c042cu, 0x140c0c04u, 0x140c0c14u, 0x140c140cu,
    0x140c1c04u, 0x140c341cu, 0x140c343eu, 0x140c3e04u, 0x14140404u, 0x14140414u, 0x14140c0cu,
    0x14140c3eu, 0x14141404u, 0x14141414u, 0x14141c3eu, 0x14142404u, 0x14142c2cu, 0x141c040cu,
    0x141c0c04u, 0x141c0c24u, 0x141c3e04u, 0x141c3e24u, 0x14241c2cu, 0x14242c1cu, 0x142c041cu,
    0x142c143eu, 0x142c240cu, 0x142c3e24u, 0x143e040cu, 0x143e041cu, 0x143e0c34u, 0x143e242cu,
    0x1c04040cu, 0x1c040c04u, 0x1c040c14u, 0x1c04140cu, 0x1c04141cu, 0x1c042c04u, 0x1c04342cu,
    0x1c043e14u, 0x1c0c0404u, 0x1c0c0414u, 0x1c0c1404u, 0x1c0c1c0cu, 0x1c0c2424u, 0x1c0c2434u,
    0x1c14040cu, 0x1c14041cu, 0x1c140c04u, 0x1c14142cu, 0x1c142c14u, 0x1c143e14u, 0x1c1c0c0cu,
    0x1c1c1c1cu, 0x1c241c04u, 0x1c24243eu, 0x1c243e14u, 0x1c2c0404u, 0x1c2c0434u, 0x1c2c1414u,
    0x1c2c2c2cu, 0x1c340c24u, 0x1c341c34u, 0x1c34341cu, 0x1c3e1c1cu, 0x1c3e3404u, 0x24040424u,
    0x24040c3eu, 0x24041c2cu, 0x24041c3eu, 0x24042c1cu, 0x24042c3eu, 0x240c3e24u, 0x24141404u,
    0x24141c3eu, 0x24142404u, 0x24143404u, 0x24143434u, 0x241c043eu, 0x241c242cu, 0x24240424u,
    0x24242c0cu, 0x24243424u, 0x242c142cu, 0x242c241cu, 0x242c3e04u, 0x243e042cu, 0x243e0c04u,
    0x243e0c14u, 0x243e1c04u, 0x2c040c14u, 0x2c04240cu, 0x2c043e04u, 0x2c0c0404u, 0x2c0c0434u,
    0x2c0c1434u, 0x2c0c2c2cu, 0x2c140c24u, 0x2c141c14u, 0x2c143e14u, 0x2c1c0414u, 0x2c1c2c1cu,
    0x2c240c04u, 0x2c24141cu, 0x2c24143eu, 0x2c243e14u, 0x2c2c0414u, 0x2c2c1c0cu, 0x2c342c04u,
    0x2c3e1424u, 0x2c3e2414u, 0x34041424u, 0x34042424u, 0x34042434u, 0x34043424u, 0x340c140cu,
    0x340c340cu, 0x34140c3eu, 0x34143424u, 0x341c1c04u, 0x341c1c34u, 0x34242424u, 0x342c042cu,
    0x342c2c14u, 0x34341c1cu, 0x343e041cu, 0x343e140cu, 0x3e04041cu, 0x3e04042cu, 0x3e04043eu,
    0x3e040c04u, 0x3e041c14u, 0x3e042c14u, 0x3e0c1434u, 0x3e0c2404u, 0x3e140c14u, 0x3e14242cu,
    0x3e142c14u, 0x3e1c0404u, 0x3e1c0c2cu, 0x3e1c1c1cu, 0x3e1c3404u, 0x3e24140cu, 0x3e24240cu,
    0x3e2c0404u, 0x3e2c0414u, 0x3e2c1424u, 0x3e341c04u,
};

std::uint32_t d3d12_iq3xxs_grid(std::uint32_t i) {
    return kIq3xxsGrid[i & 0xffu];
}

void q4k_scale_min(std::uint32_t j, const std::uint8_t* sc, std::int32_t* d, std::int32_t* m) {
    std::uint32_t dd, mm;
    if (j < 4u) {
        dd = sc[j] & 63u;
        mm = sc[j + 4u] & 63u;
    } else {
        dd = (sc[j + 4u] & 0xFu) | ((sc[j - 4u] >> 6) << 4);
        mm = (sc[j + 4u] >> 4) | ((sc[j] >> 6) << 4);
    }
    *d = static_cast<std::int32_t>(dd);
    *m = static_cast<std::int32_t>(mm);
}

// One thread's contribution for one row and one 256-element chunk. `xq` is
// one activation column in the vec_dot type's ggml blocks (block_q8_0 for
// Q4_0, block_q8_K for Q4_K / Q6_K). Integer sums are exact; the float terms
// follow ggml's vec_dot (#312).
float thread_chunk(ggml_type t, const std::uint8_t* w, std::uint32_t row_off, std::uint32_t blk,
                   std::uint32_t itid, const std::uint8_t* xq) {
    if (t == GGML_TYPE_Q4_0) {
        const std::uint32_t b = itid >> 1, h = itid & 1u;
        const std::uint8_t* xb = xq + (blk * 8u + b) * 34u; // block_q8_0
        const float dx = h2f(ld16(xb, 0));
        const auto* qx = reinterpret_cast<const std::int8_t*>(xb + 2);
        const std::uint32_t bb = row_off + (blk * 8u + b) * 18u;
        const float d = h2f(ld16(w, bb));
        const std::uint32_t q0 = ld32(w, bb + 2u + 8u * h), q1 = ld32(w, bb + 6u + 8u * h);
        std::int32_t sumi = 0;
        for (std::uint32_t i = 0; i < 4; ++i) {
            const std::uint32_t lo = 8u * h + i, hi = 16u + 8u * h + i;
            sumi += (nib(q0, i, 0) - 8) * qx[lo] + (nib(q1, i, 0) - 8) * qx[lo + 4] +
                    (nib(q0, i, 4) - 8) * qx[hi] + (nib(q1, i, 4) - 8) * qx[hi + 4];
        }
        return (d * dx) * static_cast<float>(sumi);
    }
    float dy;
    std::memcpy(&dy, xq + blk * 292u, 4); // block_q8_K: float d; int8 qs[256]; ...
    const auto* qx = reinterpret_cast<const std::int8_t*>(xq + blk * 292u + 4u);
    if (t == GGML_TYPE_Q4_K) {
        const std::uint32_t il = itid >> 2, ir = itid & 3u;
        const std::uint32_t e_lo = il * 64u + ir * 8u, e_hi = e_lo + 32u;
        const std::uint32_t bb = row_off + blk * 144u;
        const float d = h2f(ld16(w, bb)), dmin = h2f(ld16(w, bb + 2u));
        std::int32_t sc1, m1, sc2, m2;
        q4k_scale_min(2u * il, w + bb + 4u, &sc1, &m1);
        q4k_scale_min(2u * il + 1u, w + bb + 4u, &sc2, &m2);
        const std::uint32_t qa = ld32(w, bb + 16u + il * 32u + ir * 8u);
        const std::uint32_t qb = ld32(w, bb + 20u + il * 32u + ir * 8u);
        std::int32_t lo = 0, hi = 0, sl = 0, sh = 0;
        for (std::uint32_t i = 0; i < 4; ++i) {
            lo += nib(qa, i, 0) * qx[e_lo + i] + nib(qb, i, 0) * qx[e_lo + 4 + i];
            hi += nib(qa, i, 4) * qx[e_hi + i] + nib(qb, i, 4) * qx[e_hi + 4 + i];
            sl += qx[e_lo + i] + qx[e_lo + 4 + i];
            sh += qx[e_hi + i] + qx[e_hi + 4 + i];
        }
        return (d * dy) * static_cast<float>(sc1 * lo + sc2 * hi) -
               (dmin * dy) * static_cast<float>(m1 * sl + m2 * sh);
    }
    if (t == GGML_TYPE_Q5_K) {
        // Fork: shaders/ggml_d3d12_mmv_q5_k.hlsl (Q4_K lanes + qh bit 4).
        const std::uint32_t il = itid >> 2, ir = itid & 3u;
        const std::uint32_t e_lo = il * 64u + ir * 8u, e_hi = e_lo + 32u;
        const std::uint32_t bb = row_off + blk * 176u;
        const float d = h2f(ld16(w, bb)), dmin = h2f(ld16(w, bb + 2u));
        std::int32_t sc1, m1, sc2, m2;
        q4k_scale_min(2u * il, w + bb + 4u, &sc1, &m1);
        q4k_scale_min(2u * il + 1u, w + bb + 4u, &sc2, &m2);
        std::int32_t lo = 0, hi = 0, sl = 0, sh = 0;
        for (std::uint32_t i = 0; i < 8; ++i) {
            const std::uint32_t q = w[bb + 48u + il * 32u + ir * 8u + i];
            const std::uint32_t h = w[bb + 16u + ir * 8u + i];
            const std::int32_t vl =
                static_cast<std::int32_t>((q & 0xFu) | (((h >> (2u * il)) & 1u) << 4));
            const std::int32_t vh =
                static_cast<std::int32_t>((q >> 4) | (((h >> (2u * il + 1u)) & 1u) << 4));
            lo += vl * qx[e_lo + i];
            hi += vh * qx[e_hi + i];
            sl += qx[e_lo + i];
            sh += qx[e_hi + i];
        }
        return (d * dy) * static_cast<float>(sc1 * lo + sc2 * hi) -
               (dmin * dy) * static_cast<float>(m1 * sl + m2 * sh);
    }
    if (t == GGML_TYPE_IQ3_XXS) {
        // Fork: shaders/ggml_d3d12_mmv_iq3_xxs.hlsl. ib = itid / 2, h = itid % 2
        // own 16 values (groups l = 2h, 2h + 1) of sub-block ib.
        const std::uint32_t ib = itid >> 1, h = itid & 1u, e = 32u * ib + 16u * h;
        const std::uint32_t bb = row_off + blk * 98u;
        const float d = h2f(ld16(w, bb));
        const std::uint32_t idx = ld32(w, bb + 2u + 8u * ib + 4u * h);
        const std::uint32_t aux = ld32(w, bb + 66u + 4u * ib);
        const std::int32_t ls = static_cast<std::int32_t>(2u * (aux >> 28) + 1u);
        std::int32_t sumi = 0;
        for (std::uint32_t k = 0; k < 2; ++k) {
            const std::uint32_t l = 2u * h + k;
            const std::uint32_t s7 = (aux >> (7u * l)) & 127u;
            std::uint32_t par = s7;
            par ^= par >> 4;
            par ^= par >> 2;
            par ^= par >> 1;
            const std::uint32_t signs = s7 | ((par & 1u) << 7); // ksigns_iq2xs[s7]
            const std::uint32_t g1 = d3d12_iq3xxs_grid((idx >> (16u * k)) & 0xffu);
            const std::uint32_t g2 = d3d12_iq3xxs_grid((idx >> (16u * k + 8u)) & 0xffu);
            for (std::uint32_t j = 0; j < 4; ++j) {
                const std::int32_t v1 = static_cast<std::int32_t>((g1 >> (8u * j)) & 0xffu);
                const std::int32_t v2 = static_cast<std::int32_t>((g2 >> (8u * j)) & 0xffu);
                sumi += ((signs >> j) & 1u ? -v1 : v1) * qx[e + 8u * k + j];
                sumi += ((signs >> (j + 4u)) & 1u ? -v2 : v2) * qx[e + 8u * k + 4u + j];
            }
        }
        return 0.25f * (d * dy) * static_cast<float>(sumi * ls);
    }
    if (t == GGML_TYPE_IQ4_XS) {
        // Fork: shaders/ggml_d3d12_mmv_iq4_xs.hlsl. ib = itid / 2, h = itid % 2
        // own bytes 8h .. 8h + 7 of sub-block ib (values j and j + 16).
        static const std::int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                           1,    13,   25,  38,  53,  69,  89,  113};
        const std::uint32_t ib = itid >> 1, h = itid & 1u;
        const std::uint32_t e_lo = 32u * ib + 8u * h, e_hi = e_lo + 16u;
        const std::uint32_t bb = row_off + blk * 136u;
        const std::uint32_t hdr = ld32(w, bb);
        const float d = h2f(hdr & 0xffffu);
        const std::uint32_t sh = hdr >> 16;
        const std::uint32_t sl = (ld32(w, bb + 4u) >> (8u * (ib >> 1))) & 0xffu;
        const std::int32_t ls = static_cast<std::int32_t>(((sl >> (4u * (ib & 1u))) & 15u) |
                                                          (((sh >> (2u * ib)) & 3u) << 4));
        std::int32_t sumi = 0;
        for (std::uint32_t j = 0; j < 8; ++j) {
            const std::uint32_t q =
                (ld32(w, bb + 8u + 16u * ib + 8u * h + (j & ~3u)) >> (8u * (j & 3u))) & 0xffu;
            sumi += kv[q & 15u] * qx[e_lo + j] + kv[q >> 4] * qx[e_hi + j];
        }
        return (d * dy) * static_cast<float>((ls - 32) * sumi);
    }
    // Q6_K
    const std::uint32_t v = itid >> 3, l0 = 4u * (itid & 7u), is = l0 >> 4;
    const std::uint32_t e = 128u * v + l0;
    const std::uint32_t bb = row_off + blk * 210u;
    const std::uint32_t qla = ld32(w, bb + 64u * v + l0);
    const std::uint32_t qlb = ld32(w, bb + 64u * v + 32u + l0);
    const std::uint32_t qh = ld32(w, bb + 128u + 32u * v + l0);
    const std::uint32_t s0 = ld32(w, bb + 192u + 8u * v), s1 = ld32(w, bb + 196u + 8u * v);
    const float d = h2f(ld16(w, bb + 208u));
    auto q6 = [](std::uint32_t ql, std::uint32_t qhw, std::uint32_t i, std::uint32_t qs,
                 std::uint32_t hs) {
        const std::uint32_t lo4 = (ql >> (8u * i + qs)) & 0xFu;
        const std::uint32_t hi2 = (qhw >> (8u * i + hs)) & 3u;
        return static_cast<std::int32_t>(lo4 | (hi2 << 4)) - 32;
    };
    std::int32_t t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    for (std::uint32_t i = 0; i < 4; ++i) {
        t1 += q6(qla, qh, i, 0, 0) * qx[e + i];
        t2 += q6(qlb, qh, i, 0, 2) * qx[e + 32 + i];
        t3 += q6(qla, qh, i, 4, 4) * qx[e + 64 + i];
        t4 += q6(qlb, qh, i, 4, 6) * qx[e + 96 + i];
    }
    const std::int32_t tsum =
        sbyte(s0, is) * t1 + sbyte(s0, is + 2) * t2 + sbyte(s1, is) * t3 + sbyte(s1, is + 2) * t4;
    return (d * dy) * static_cast<float>(tsum);
}

} // namespace

ggml_type d3d12_activation_type(ggml_type weight) {
    return weight == GGML_TYPE_Q4_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q8_K;
}

void d3d12_quantize_activations(ggml_type weight, const float* x, std::size_t x_stride, int k,
                                int ncols, std::uint8_t* out) {
    const ggml_type at = d3d12_activation_type(weight);
    const auto from_float = ggml_get_type_traits_cpu(at)->from_float;
    const std::size_t row = ggml_row_size(at, k);
    for (int c = 0; c < ncols; ++c)
        from_float(x + static_cast<std::size_t>(c) * x_stride, out + c * row, k);
}

void d3d12_mmv_emulate(ggml_type t, const std::uint8_t* w, std::size_t w_row_bytes, const float* x,
                       std::size_t x_stride, float* y, std::size_t y_stride, int n, int k,
                       int ncols) {
    if (!w || !x || !y || n <= 0 || k <= 0 || k % kD3d12Chunk != 0 || ncols <= 0 ||
        !d3d12_weight_type_supported(t))
        return;
    const std::size_t xq_row = ggml_row_size(d3d12_activation_type(t), k);
    std::vector<std::uint8_t> xq(xq_row * static_cast<std::size_t>(ncols) + 4);
    d3d12_quantize_activations(t, x, x_stride, k, ncols, xq.data());
    const std::uint32_t nchunk = static_cast<std::uint32_t>(k / kD3d12Chunk);
    const std::uint32_t threads = static_cast<std::uint32_t>(d3d12_mm_threads(k));
    const std::uint32_t in_flight = threads / 16u;
    float acc[kD3d12MmvThreadsLong];
    for (int c = 0; c < ncols; ++c) {
        const std::uint8_t* xc = xq.data() + static_cast<std::size_t>(c) * xq_row;
        for (int row = 0; row < n; ++row) {
            const std::uint32_t row_off = static_cast<std::uint32_t>(row * w_row_bytes);
            for (std::uint32_t tid = 0; tid < threads; ++tid) {
                float a = 0.f;
                for (std::uint32_t blk = tid >> 4; blk < nchunk; blk += in_flight)
                    a += thread_chunk(t, w, row_off, blk, tid & 15u, xc);
                acc[tid] = a;
            }
            for (int stride = static_cast<int>(threads) / 2; stride > 0; stride >>= 1)
                for (int i = 0; i < stride; ++i)
                    acc[i] += acc[i + stride];
            y[static_cast<std::size_t>(c) * y_stride + row] = acc[0];
        }
    }
}

void d3d12_get_rows_emulate(ggml_type t, const std::uint8_t* w, std::size_t w_row_bytes,
                            const std::int32_t* ids, int n_ids, int k, float* y,
                            std::size_t y_stride) {
    if (!w || !ids || !y || n_ids <= 0 || k <= 0 || k % kD3d12Chunk != 0 ||
        !d3d12_get_rows_type_supported(t))
        return;
    auto ld8 = [&](std::uint32_t a) { return (ld32(w, a & ~3u) >> ((a & 3u) * 8u)) & 0xffu; };
    // One thread group per (256-element chunk, id); thread tid = 32v + l writes
    // elements 128v + l + {0, 32, 64, 96}, as dequantize_row_q6_K's inner loop.
    for (int i = 0; i < n_ids; ++i) {
        for (std::uint32_t blk = 0; blk < static_cast<std::uint32_t>(k / kD3d12Chunk); ++blk) {
            const std::uint32_t bb = static_cast<std::uint32_t>(ids[i] * w_row_bytes) + blk * 210u;
            const float d = h2f(ld16(w, bb + 208u));
            for (std::uint32_t tid = 0; tid < 64u; ++tid) {
                const std::uint32_t v = tid >> 5, l = tid & 31u, is = l >> 4;
                const std::uint32_t qla = ld8(bb + 64u * v + l), qlb = ld8(bb + 64u * v + l + 32u);
                const std::uint32_t qh = ld8(bb + 128u + 32u * v + l);
                auto sc = [&](std::uint32_t j) {
                    return static_cast<float>(
                        static_cast<std::int8_t>(ld8(bb + 192u + 8u * v + is + j)));
                };
                const int q1 = static_cast<int>((qla & 0xFu) | (((qh >> 0) & 3u) << 4)) - 32;
                const int q2 = static_cast<int>((qlb & 0xFu) | (((qh >> 2) & 3u) << 4)) - 32;
                const int q3 = static_cast<int>((qla >> 4) | (((qh >> 4) & 3u) << 4)) - 32;
                const int q4 = static_cast<int>((qlb >> 4) | (((qh >> 6) & 3u) << 4)) - 32;
                float* out = y + static_cast<std::size_t>(i) * y_stride + blk * 256u + 128u * v + l;
                out[0] = d * sc(0) * static_cast<float>(q1);
                out[32] = d * sc(2) * static_cast<float>(q2);
                out[64] = d * sc(4) * static_cast<float>(q3);
                out[96] = d * sc(6) * static_cast<float>(q4);
            }
        }
    }
}

// --- FLASH_ATTN_EXT (fork) ---

bool d3d12_fa_supported(const D3d12FlashAttnDesc& d) {
    const bool kv_ok =
        (d.k_type == GGML_TYPE_F16 || d.k_type == GGML_TYPE_Q8_0) && d.v_type == d.k_type;
    const bool mask_ok = d.mask_type == GGML_TYPE_COUNT ||
                         (d.mask_type == GGML_TYPE_F16 && d.mask_contiguous && d.mask_ne2 >= 1 &&
                          d.n_head % d.mask_ne2 == 0 && d.mask_ne1 >= d.n_q);
    return d.q_type == GGML_TYPE_F32 && d.dst_type == GGML_TYPE_F32 && kv_ok && mask_ok &&
           d.dk == d.dv && d.dk > 0 && d.dk % kD3d12FaThreads == 0 && d.dk <= kD3d12FaMaxD &&
           d.n_q > 0 && d.n_q <= kD3d12MaxGroups && d.n_head > 0 && d.n_head <= kD3d12MaxGroups &&
           d.n_head_kv > 0 && d.n_head % d.n_head_kv == 0 && d.n_kv > 0 && d.ne3 == 1 &&
           d.max_bias == 0.f && d.rows_contiguous && d.dst_contiguous && d.strides_even &&
           d.strides_fit_u32;
}

void d3d12_fa_scales(float scale, float logit_softcap, std::uint32_t* scale_bits,
                     std::uint32_t* softcap_bits) {
    if (logit_softcap != 0.f)
        scale /= logit_softcap;
    std::memcpy(scale_bits, &scale, 4);
    std::memcpy(softcap_bits, &logit_softcap, 4);
}

void d3d12_fa_emulate(const D3d12FaParams& p, const std::uint8_t* q, const std::uint8_t* k,
                      const std::uint8_t* v, const std::uint8_t* mask, const float* sinks,
                      float* out) {
    constexpr std::uint32_t T = kD3d12FaThreads;
    const float NEG = -std::numeric_limits<float>::infinity();
    auto rd16 = [](const std::uint8_t* b, std::uint32_t a) {
        std::uint16_t h;
        std::memcpy(&h, b + a, 2);
        return ggml_fp16_to_fp32(h);
    };
    float scale, softcap;
    std::memcpy(&scale, &p.scale_bits, 4);
    std::memcpy(&softcap, &p.softcap_bits, 4);
    const std::uint32_t d = p.d;
    std::vector<float> qs(d), acc(d), s(T), ps(T);
    for (std::uint32_t iq = 0; iq < p.n_q; ++iq) {
        for (std::uint32_t h = 0; h < p.n_head; ++h) {
            const std::uint32_t hk = h / p.gqa;
            for (std::uint32_t e = 0; e < d; ++e)
                std::memcpy(&qs[e], q + p.q_off + iq * p.nbq1 + h * p.nbq2 + e * 4u, 4);
            std::fill(acc.begin(), acc.end(), 0.f);
            float M = NEG, S = 0.f;
            const std::uint32_t mrow = p.m_off + iq * p.nbm1 + (h % p.mask_ne2) * p.nbm2;
            for (std::uint32_t kb = 0; kb < p.n_kv; kb += T) {
                for (std::uint32_t t = 0; t < T; ++t) {
                    const std::uint32_t ic = kb + t;
                    s[t] = NEG;
                    if (ic >= p.n_kv)
                        continue;
                    const float mv = p.has_mask ? rd16(mask, mrow + 2u * ic) : 0.f;
                    if (mv == NEG)
                        continue;
                    const std::uint32_t base = p.k_off + hk * p.nbk2 + ic * p.nbk1;
                    float dot = 0.f;
                    if (p.kv_q8) {
                        for (std::uint32_t b = 0; b < d / 32u; ++b) {
                            const std::uint32_t a = base + b * 34u;
                            float part = 0.f;
                            for (std::uint32_t e = 0; e < 32u; ++e)
                                part += qs[b * 32u + e] *
                                        static_cast<float>(static_cast<std::int8_t>(k[a + 2u + e]));
                            dot += rd16(k, a) * part;
                        }
                    } else {
                        for (std::uint32_t e = 0; e < d; ++e)
                            dot += qs[e] * rd16(k, base + 2u * e);
                    }
                    float x = dot * scale;
                    if (softcap != 0.f)
                        x = softcap * std::tanh(x);
                    s[t] = x + mv;
                }
                float bm = NEG;
                for (std::uint32_t t = 0; t < T; ++t)
                    bm = std::max(bm, s[t]);
                const float Mn = std::max(M, bm);
                if (Mn == NEG)
                    continue;
                float bs = 0.f;
                for (std::uint32_t t = 0; t < T; ++t) {
                    ps[t] = s[t] == NEG ? 0.f : std::exp(s[t] - Mn);
                    bs += ps[t];
                }
                const float ms = M == NEG ? 0.f : std::exp(M - Mn);
                S = S * ms + bs;
                M = Mn;
                for (float& a : acc)
                    a *= ms;
                const std::uint32_t nk = std::min(T, p.n_kv - kb);
                for (std::uint32_t t = 0; t < nk; ++t) {
                    if (ps[t] == 0.f)
                        continue;
                    const std::uint32_t vrow = p.v_off + hk * p.nbv2 + (kb + t) * p.nbv1;
                    for (std::uint32_t e = 0; e < d; ++e) {
                        float val;
                        if (p.kv_q8) {
                            const std::uint32_t a = vrow + (e >> 5) * 34u;
                            val = rd16(v, a) * static_cast<float>(
                                                   static_cast<std::int8_t>(v[a + 2u + (e & 31u)]));
                        } else {
                            val = rd16(v, vrow + 2u * e);
                        }
                        acc[e] += ps[t] * val;
                    }
                }
            }
            if (p.has_sinks) {
                const float sk = sinks[h];
                if (sk > M) {
                    const float ms = M == NEG ? 0.f : std::exp(M - sk);
                    for (float& a : acc)
                        a *= ms;
                    S = S * ms + 1.f;
                    M = sk;
                } else {
                    S += std::exp(sk - M);
                }
            }
            const float inv = S == 0.f ? 0.f : 1.f / S;
            float* o = out + (static_cast<std::size_t>(iq) * p.n_head + h) * d;
            for (std::uint32_t e = 0; e < d; ++e)
                o[e] = acc[e] * inv;
        }
    }
}

// --- Small ops between the matmuls (fork) ---

namespace {

bool ew_f32_rows(const ggml_tensor* t) {
    return t && t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float);
}

bool ew_grid_ok(const ggml_tensor* t) {
    return t->ne[0] > 0 && t->ne[1] <= kD3d12MaxGroups && t->ne[2] <= kD3d12MaxGroups &&
           t->ne[3] <= kD3d12MaxGroups && ggml_nbytes(t) < 0xF0000000ull;
}

} // namespace

D3d12EwOp d3d12_ew_op(const ggml_tensor* node) {
    const ggml_tensor* a = node->src[0];
    const ggml_tensor* b = node->src[1];
    if (!ew_f32_rows(node) || !ggml_is_contiguous(node) || !ew_f32_rows(a) || !ew_grid_ok(node) ||
        ggml_nbytes(a) >= 0xF0000000ull)
        return D3d12EwOp::None;
    switch (node->op) {
    case GGML_OP_ADD:
    case GGML_OP_MUL:
        if (!ew_f32_rows(b) || !ggml_can_repeat(b, a) || !ggml_are_same_shape(a, node) ||
            ggml_nbytes(b) >= 0xF0000000ull)
            return D3d12EwOp::None;
        return node->op == GGML_OP_ADD ? D3d12EwOp::Add : D3d12EwOp::Mul;
    case GGML_OP_RMS_NORM:
        return ggml_are_same_shape(a, node) ? D3d12EwOp::RmsNorm : D3d12EwOp::None;
    case GGML_OP_UNARY:
        return ggml_get_unary_op(node) == GGML_UNARY_OP_SILU && ggml_are_same_shape(a, node)
                   ? D3d12EwOp::Silu
                   : D3d12EwOp::None;
    case GGML_OP_GLU:
        if (ggml_get_glu_op(node) != GGML_GLU_OP_SWIGLU || !ggml_is_contiguous(a))
            return D3d12EwOp::None;
        if (b) {
            if (!ew_f32_rows(b) || !ggml_is_contiguous(b) || !ggml_are_same_shape(a, b) ||
                !ggml_are_same_shape(a, node))
                return D3d12EwOp::None;
        } else if (a->ne[0] != 2 * node->ne[0] || a->ne[1] != node->ne[1] ||
                   a->ne[2] != node->ne[2] || a->ne[3] != node->ne[3]) {
            return D3d12EwOp::None;
        }
        return D3d12EwOp::Swiglu;
    default:
        return D3d12EwOp::None;
    }
}

D3d12EwParams d3d12_ew_params(const ggml_tensor* node) {
    const ggml_tensor* a = node->src[0];
    const ggml_tensor* b = node->src[1];
    auto u = [](std::int64_t v) { return static_cast<std::uint32_t>(v); };
    D3d12EwParams p;
    p.ne0 = u(node->ne[0]);
    p.ne1 = u(node->ne[1]);
    p.ne2 = u(node->ne[2]);
    p.ne3 = u(node->ne[3]);
    p.s0_nb1 = u(a->nb[1]);
    p.s0_nb2 = u(a->nb[2]);
    p.s0_nb3 = u(a->nb[3]);
    p.d_nb1 = u(node->nb[1]);
    p.d_nb2 = u(node->nb[2]);
    p.d_nb3 = u(node->nb[3]);
    if (b) {
        p.s1_ne0 = u(b->ne[0]);
        p.s1_ne1 = u(b->ne[1]);
        p.s1_ne2 = u(b->ne[2]);
        p.s1_ne3 = u(b->ne[3]);
        p.s1_nb1 = u(b->nb[1]);
        p.s1_nb2 = u(b->nb[2]);
        p.s1_nb3 = u(b->nb[3]);
    }
    if (node->op == GGML_OP_RMS_NORM) {
        std::memcpy(&p.p0, node->op_params, 4); // eps
    } else if (node->op == GGML_OP_GLU) {
        if (b) {
            p.p0 = 1;
        } else {
            std::int32_t sw = 0;
            std::memcpy(&sw, reinterpret_cast<const std::int32_t*>(node->op_params) + 1, 4);
            const bool swapped = sw != 0;
            p.p1 = swapped ? p.ne0 : 0;
            p.p2 = swapped ? 0 : p.ne0;
        }
    }
    return p;
}

void d3d12_ew_emulate(D3d12EwOp op, const D3d12EwParams& p, const std::uint8_t* s0,
                      const std::uint8_t* s1, std::uint8_t* d) {
    auto ld = [](const std::uint8_t* b, std::uint32_t a) {
        float f;
        std::memcpy(&f, b + a, 4);
        return f;
    };
    auto st = [&](std::uint32_t a, float f) { std::memcpy(d + a, &f, 4); };
    auto silu = [](float x) { return x / (1.f + std::exp(-x)); };
    for (std::uint32_t z = 0; z < p.ne3; ++z)
        for (std::uint32_t y = 0; y < p.ne2; ++y)
            for (std::uint32_t x = 0; x < p.ne1; ++x) {
                const std::uint32_t a = p.s0_off + x * p.s0_nb1 + y * p.s0_nb2 + z * p.s0_nb3;
                const std::uint32_t o = p.d_off + x * p.d_nb1 + y * p.d_nb2 + z * p.d_nb3;
                const std::uint32_t b = p.s1_off + (x % p.s1_ne1) * p.s1_nb1 +
                                        (y % p.s1_ne2) * p.s1_nb2 + (z % p.s1_ne3) * p.s1_nb3;
                switch (op) {
                case D3d12EwOp::Add:
                    for (std::uint32_t i = 0; i < p.ne0; ++i)
                        st(o + i * 4, ld(s0, a + i * 4) + ld(s1, b + (i % p.s1_ne0) * 4));
                    break;
                case D3d12EwOp::Mul:
                    for (std::uint32_t i = 0; i < p.ne0; ++i)
                        st(o + i * 4, ld(s0, a + i * 4) * ld(s1, b + (i % p.s1_ne0) * 4));
                    break;
                case D3d12EwOp::Silu:
                    for (std::uint32_t i = 0; i < p.ne0; ++i)
                        st(o + i * 4, silu(ld(s0, a + i * 4)));
                    break;
                case D3d12EwOp::Swiglu:
                    for (std::uint32_t i = 0; i < p.ne0; ++i) {
                        const float gate = p.p0 ? ld(s0, a + i * 4) : ld(s0, a + (p.p1 + i) * 4);
                        const float up = p.p0 ? ld(s1, b + i * 4) : ld(s0, a + (p.p2 + i) * 4);
                        st(o + i * 4, silu(gate) * up);
                    }
                    break;
                case D3d12EwOp::RmsNorm: {
                    float sum = 0.f;
                    for (std::uint32_t i = 0; i < p.ne0; ++i)
                        sum += ld(s0, a + i * 4) * ld(s0, a + i * 4);
                    float eps;
                    std::memcpy(&eps, &p.p0, 4);
                    const float scale = 1.f / std::sqrt(sum / static_cast<float>(p.ne0) + eps);
                    for (std::uint32_t i = 0; i < p.ne0; ++i)
                        st(o + i * 4, ld(s0, a + i * 4) * scale);
                    break;
                }
                case D3d12EwOp::None:
                    break;
                }
            }
}

void d3d12_quant_q8k_emulate(const float* x, std::size_t x_stride, int k, int nrows,
                             std::uint8_t* out, std::size_t out_stride) {
    for (int r = 0; r < nrows; ++r) {
        const float* xr =
            reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(x) + r * x_stride);
        for (int blk = 0; blk < k / 256; ++blk) {
            const float* xb = xr + blk * 256;
            std::uint8_t* ob = out + r * out_stride + blk * 292;
            float amax = 0.f, mx = 0.f;
            for (int j = 0; j < 256; ++j)
                if (std::fabs(xb[j]) > amax) {
                    amax = std::fabs(xb[j]);
                    mx = xb[j];
                }
            float dval = 0.f;
            std::int8_t q[256] = {};
            if (amax != 0.f) {
                const float iscale = -127.f / mx;
                for (int j = 0; j < 256; ++j)
                    q[j] = static_cast<std::int8_t>(
                        std::min(127, static_cast<int>(std::nearbyint(iscale * xb[j]))));
                dval = 1.f / iscale;
            }
            std::memcpy(ob, &dval, 4);
            std::memcpy(ob + 4, q, 256);
            for (int s = 0; s < 16; ++s) {
                int sum = 0;
                for (int j = 0; j < 16; ++j)
                    sum += q[s * 16 + j];
                const std::int16_t v = static_cast<std::int16_t>(sum);
                std::memcpy(ob + 260 + 2 * s, &v, 2);
            }
        }
    }
}

// --- Selftest CSV ---

const char* d3d12_selftest_csv_header() {
    return "type,n,k,ncols,rel_err,gpu_ms,packed_gbs,ok,d3d12_ran,host,date,error\n";
}

bool d3d12_flag_enabled(std::string_view config) {
    const auto first = config.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos || config[first] != '1')
        return false;
    return config.find_first_not_of(" \t\r\n", first + 1) == std::string_view::npos;
}

const char* d3d12_cols_class(std::int64_t ncols) {
    if (ncols <= 1)
        return "1";
    if (ncols < 64)
        return "2-63";
    if (ncols < 512)
        return "64-511";
    return "512+";
}

void D3d12DispatchProfile::add(const D3d12DispatchKind& kind, double gpu_ms, double weight_bytes,
                               std::int64_t ncols) {
    auto it = std::find_if(rows_.begin(), rows_.end(),
                           [&](const D3d12DispatchRow& r) { return r.kind == kind; });
    if (it == rows_.end()) {
        rows_.push_back(D3d12DispatchRow{kind});
        it = rows_.end() - 1;
    }
    ++it->count;
    it->gpu_ms += gpu_ms;
    it->weight_bytes += weight_bytes;
    it->columns += static_cast<std::uint64_t>(ncols > 0 ? ncols : 0);
}

std::vector<D3d12DispatchRow> D3d12DispatchProfile::sorted() const {
    std::vector<D3d12DispatchRow> out = rows_;
    std::stable_sort(
        out.begin(), out.end(),
        [](const D3d12DispatchRow& a, const D3d12DispatchRow& b) { return a.gpu_ms > b.gpu_ms; });
    return out;
}

std::string format_d3d12_dispatch_profile(const char* phase, const D3d12DispatchProfile& p) {
    if (p.empty())
        return {};
    std::string out;
    double total = 0.0;
    std::uint64_t timed = 0;
    for (const D3d12DispatchRow& r : p.sorted()) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "[xllama] d3d12-dispatch phase=%s op=%s type=%s cpg=%d threads=%d cols=%s "
                      "count=%llu gpu_ms=%.3f ms_per=%.4f weight_mib=%.1f columns=%llu\n",
                      phase ? phase : "-", r.kind.op.c_str(), r.kind.type.c_str(),
                      r.kind.columns_per_group, r.kind.threads, r.kind.cols.c_str(),
                      static_cast<unsigned long long>(r.count), r.gpu_ms,
                      r.count ? r.gpu_ms / static_cast<double>(r.count) : 0.0,
                      r.weight_bytes / 1048576.0, static_cast<unsigned long long>(r.columns));
        out += buf;
        total += r.gpu_ms;
        timed += r.count;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "[xllama] d3d12-dispatch phase=%s summary kinds=%zu timed=%llu missing=%llu "
                  "gpu_ms=%.3f\n",
                  phase ? phase : "-", p.rows().size(), static_cast<unsigned long long>(timed),
                  static_cast<unsigned long long>(p.missing()), total);
    out += buf;
    return out;
}

namespace {

std::optional<int> parse_nonneg_int(std::string_view s, std::size_t i, std::size_t* end) {
    if (i >= s.size() || s[i] < '0' || s[i] > '9')
        return std::nullopt;
    int v = 0;
    std::size_t j = i;
    while (j < s.size() && s[j] >= '0' && s[j] <= '9') {
        const int d = s[j] - '0';
        if (v > (std::numeric_limits<int>::max() - d) / 10)
            return std::nullopt;
        v = v * 10 + d;
        ++j;
    }
    if (end)
        *end = j;
    return v;
}

bool layer_component_end(std::string_view s, std::size_t j) {
    return j == s.size() || s[j] == '.' || s[j] == '#' || s[j] == '/';
}

void write_ms_or_missing(char* buf, std::size_t n, std::uint64_t timed, double ms) {
    if (timed == 0)
        std::snprintf(buf, n, "missing");
    else
        std::snprintf(buf, n, "%.3f", ms);
}

} // namespace

std::optional<int> d3d12_layer_from_name(std::string_view name) {
    std::string_view inner = name;
    const auto hash1 = name.find('#');
    const auto hash2 = name.rfind('#');
    if (hash1 != std::string_view::npos && hash2 != hash1)
        inner = name.substr(hash1 + 1, hash2 - hash1 - 1);

    for (std::size_t i = 0; i + 4 <= inner.size(); ++i) {
        if ((i == 0 || inner[i - 1] == '.' || inner[i - 1] == '/' || inner[i - 1] == '#') &&
            inner.compare(i, 4, "blk.") == 0) {
            std::size_t end = 0;
            const auto v = parse_nonneg_int(inner, i + 4, &end);
            if (v && layer_component_end(inner, end))
                return v;
        }
    }

    const auto dash = inner.rfind('-');
    if (dash != std::string_view::npos && dash + 1 < inner.size()) {
        std::size_t end = 0;
        const auto v = parse_nonneg_int(inner, dash + 1, &end);
        if (v && end == inner.size())
            return v;
    }
    return std::nullopt;
}

std::optional<int> d3d12_layer_from_names(std::string_view tensor_name,
                                          std::string_view node_name) {
    if (const auto layer = d3d12_layer_from_name(tensor_name))
        return layer;
    return d3d12_layer_from_name(node_name);
}

std::string d3d12_layer_label(std::optional<int> layer) {
    if (!layer)
        return kD3d12UnattributedLayer;
    return std::to_string(*layer);
}

void D3d12LayerProfile::add(const D3d12LayerKey& key, const std::optional<double>& gpu_ms,
                            const std::optional<double>& wait_ms) {
    auto it = std::find_if(rows_.begin(), rows_.end(),
                           [&](const D3d12LayerRow& r) { return r.key == key; });
    if (it == rows_.end()) {
        rows_.push_back(D3d12LayerRow{key});
        it = rows_.end() - 1;
    }
    ++it->count;
    if (gpu_ms) {
        ++it->gpu_timed;
        it->gpu_ms += *gpu_ms;
    } else {
        ++it->gpu_missing;
    }
    if (wait_ms) {
        ++it->wait_timed;
        it->wait_ms += *wait_ms;
    } else {
        ++it->wait_missing;
    }
}

std::vector<D3d12LayerRow> D3d12LayerProfile::sorted() const {
    std::vector<D3d12LayerRow> out = rows_;
    std::stable_sort(out.begin(), out.end(), [](const D3d12LayerRow& a, const D3d12LayerRow& b) {
        const bool a_u = !a.key.layer.has_value();
        const bool b_u = !b.key.layer.has_value();
        if (a_u != b_u)
            return b_u; // unattributed last
        if (a.key.layer && b.key.layer && *a.key.layer != *b.key.layer)
            return *a.key.layer < *b.key.layer;
        return a.key.op < b.key.op;
    });
    return out;
}

std::string format_d3d12_layer_profile(const char* phase, const D3d12LayerProfile& p) {
    if (p.empty())
        return {};
    std::string out;
    std::uint64_t timed = 0, missing = 0, wait_timed = 0, wait_missing = 0;
    double gpu_total = 0.0, wait_total = 0.0;
    for (const D3d12LayerRow& r : p.sorted()) {
        char gpu[32], wait[32];
        write_ms_or_missing(gpu, sizeof(gpu), r.gpu_timed, r.gpu_ms);
        write_ms_or_missing(wait, sizeof(wait), r.wait_timed, r.wait_ms);
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "[xllama] d3d12-layer phase=%s layer=%s op=%s count=%llu gpu_ms=%s "
                      "wait_ms=%s gpu_missing=%llu wait_missing=%llu\n",
                      phase ? phase : "-", d3d12_layer_label(r.key.layer).c_str(), r.key.op.c_str(),
                      static_cast<unsigned long long>(r.count), gpu, wait,
                      static_cast<unsigned long long>(r.gpu_missing),
                      static_cast<unsigned long long>(r.wait_missing));
        out += buf;
        timed += r.gpu_timed;
        missing += r.gpu_missing;
        wait_timed += r.wait_timed;
        wait_missing += r.wait_missing;
        gpu_total += r.gpu_ms;
        wait_total += r.wait_ms;
    }
    char gpu[32], wait[32];
    write_ms_or_missing(gpu, sizeof(gpu), timed, gpu_total);
    write_ms_or_missing(wait, sizeof(wait), wait_timed, wait_total);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "[xllama] d3d12-layer phase=%s summary rows=%zu timed=%llu missing=%llu "
                  "gpu_ms=%s wait_ms=%s wait_missing=%llu\n",
                  phase ? phase : "-", p.rows().size(), static_cast<unsigned long long>(timed),
                  static_cast<unsigned long long>(missing), gpu, wait,
                  static_cast<unsigned long long>(wait_missing));
    out += buf;
    return out;
}

std::string format_d3d12_phase(const char* phase, const D3d12PhaseStats& s) {
    char gpu[32];
    if (s.gpu_time_valid())
        std::snprintf(gpu, sizeof(gpu), "%.3f", s.gpu_ms);
    else
        std::snprintf(gpu, sizeof(gpu), "unavailable");
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "[xllama] d3d12-phase phase=%s calls=%llu submissions=%llu dispatches=%llu "
                  "wall_ms=%.3f quant_ms=%.3f record_ms=%.3f exec_wait_ms=%.3f gpu_ms=%s "
                  "ts_valid=%llu ts_missing=%llu\n",
                  phase ? phase : "-", static_cast<unsigned long long>(s.calls),
                  static_cast<unsigned long long>(s.submissions),
                  static_cast<unsigned long long>(s.dispatches), s.wall_ms, s.quant_ms, s.record_ms,
                  s.exec_wait_ms, gpu, static_cast<unsigned long long>(s.ts_valid),
                  static_cast<unsigned long long>(s.ts_missing));
    return buf;
}

std::string format_d3d12_selftest_row(const D3d12SelftestRow& r, const char* host_label) {
    char date_buf[32];
    std::time_t now = std::time(nullptr);
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    std::string err = r.error.empty() ? "-" : r.error;
    for (char& ch : err)
        if (ch == ',' || ch == '\n' || ch == '\r')
            ch = ' ';
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s,%d,%d,%d,%.3g,%.4f,%.2f,%d,%d,%s,%s,%s\n", r.type.c_str(),
                  r.n, r.k, r.ncols, r.rel_err, r.gpu_ms, r.packed_gbs, r.ok ? 1 : 0,
                  r.d3d12_ran ? 1 : 0, host_label ? host_label : "unknown", date_buf, err.c_str());
    return buf;
}

} // namespace xllama

    #if !defined(_WIN32)

namespace xllama {

bool ggml_d3d12_register() {
    return false;
}

ggml_backend_buffer_type_t ggml_d3d12_weights_buft() {
    return nullptr;
}

bool ggml_d3d12_take_stats(D3d12PhaseStats* out) {
    if (out)
        *out = D3d12PhaseStats{};
    return false;
}

void run_d3d12_selftest(std::vector<D3d12SelftestRow>* out) {
    if (!out)
        return;
    D3d12SelftestRow r;
    r.type = "-";
    r.error = "d3d12 unavailable on this platform";
    out->push_back(std::move(r));
}

} // namespace xllama

    #else // _WIN32 — the backend itself

        #include <atomic>
        #include <chrono>
        #include <cstdint>
        #include <mutex>
        #include <random>

        #include <dxgi1_4.h>

        #include "d3d12_compute.h"
        #include "ggml-alloc.h"
        #include "ggml-backend-impl.h"
        #include "ggml-backend.h"
        #include "ggml_d3d12_ew_add_dxil.h"
        #include "ggml_d3d12_ew_mul_dxil.h"
        #include "ggml_d3d12_ew_quantq8k_dxil.h"
        #include "ggml_d3d12_ew_rmsnorm_dxil.h"
        #include "ggml_d3d12_ew_silu_dxil.h"
        #include "ggml_d3d12_ew_swiglu_dxil.h"
        #include "ggml_d3d12_fa_dxil.h"
        #include "ggml_d3d12_get_rows_q6_k_dxil.h"
        #include "ggml_d3d12_mmt_iq3_xxs_t128_dxil.h"
        #include "ggml_d3d12_mmt_iq3_xxs_t64_dxil.h"
        #include "ggml_d3d12_mmt_iq4_xs_t128_dxil.h"
        #include "ggml_d3d12_mmt_iq4_xs_t64_dxil.h"
        #include "ggml_d3d12_mmt_q4_k_t128_dxil.h"
        #include "ggml_d3d12_mmt_q4_k_t64_dxil.h"
        #include "ggml_d3d12_mmt_q5_k_t128_dxil.h"
        #include "ggml_d3d12_mmt_q5_k_t64_dxil.h"
        #include "ggml_d3d12_mmt_q6_k_t128_dxil.h"
        #include "ggml_d3d12_mmt_q6_k_t64_dxil.h"
        #include "ggml_d3d12_mmv_iq3_xxs_t128_dxil.h"
        #include "ggml_d3d12_mmv_iq3_xxs_t64_dxil.h"
        #include "ggml_d3d12_mmv_iq4_xs_t128_dxil.h"
        #include "ggml_d3d12_mmv_iq4_xs_t64_dxil.h"
        #include "ggml_d3d12_mmv_q4_0_t128_dxil.h"
        #include "ggml_d3d12_mmv_q4_0_t64_dxil.h"
        #include "ggml_d3d12_mmv_q4_k_t128_dxil.h"
        #include "ggml_d3d12_mmv_q4_k_t64_dxil.h"
        #include "ggml_d3d12_mmv_q5_k_t128_dxil.h"
        #include "ggml_d3d12_mmv_q5_k_t64_dxil.h"
        #include "ggml_d3d12_mmv_q6_k_t128_dxil.h"
        #include "ggml_d3d12_mmv_q6_k_t64_dxil.h"
        #include "xllama/d3d12_dyn.h"
        #include "xllama/path_utils.h"
        #include "xllama/platform.h"
        #include "xllama/utf8_utils.h"

namespace xllama {
namespace {

using d3d12c::ComPtr;

enum Pso {
    kPsoQ40 = 0,
    kPsoQ4K = 1,
    kPsoQ6K = 2,
    kPsoIq3Xxs = 3,
    kPsoIq4Xs = 4,
    kPsoQ5K = 5,
    kPsoCount = 6
};

int pso_for(ggml_type t) {
    switch (t) {
    case GGML_TYPE_Q4_0:
        return kPsoQ40;
    case GGML_TYPE_Q4_K:
        return kPsoQ4K;
    case GGML_TYPE_Q6_K:
        return kPsoQ6K;
    case GGML_TYPE_IQ3_XXS:
        return kPsoIq3Xxs;
    case GGML_TYPE_IQ4_XS:
        return kPsoIq4Xs;
    case GGML_TYPE_Q5_K:
        return kPsoQ5K;
    default:
        return -1;
    }
}

// Process-wide D3D12 state behind the ggml device. One queue, serialised.
struct Gpu {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cl;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso[kPsoCount][2]; // [type][0 = 64 threads, 1 = 128]
    ComPtr<ID3D12PipelineState> pso_get_rows;      // Q6_K
    // Fork: tiled prefill (ggml_d3d12_mmt_*.hlsl, kD3d12Tile columns per group);
    // null where a type has none. d3d12-tile-off.txt disables them.
    ComPtr<ID3D12PipelineState> pso_tile[kPsoCount][2];
    bool tile_on = true;
    // Fork: FLASH_ATTN_EXT (ggml_d3d12_fa.hlsl) and its own root signature
    // (24 constants, six root UAVs). Null when the PSO failed to build.
    ComPtr<ID3D12RootSignature> root_fa;
    ComPtr<ID3D12PipelineState> pso_fa;
    // Fork: small ops (ggml_d3d12_ew.hlsl) on root_fa, indexed by D3d12EwOp;
    // pso_q8k quantizes activations produced on the GPU.
    ComPtr<ID3D12PipelineState> pso_ew[5];
    ComPtr<ID3D12PipelineState> pso_q8k;
    ComPtr<ID3D12QueryHeap> ts;
    ComPtr<ID3D12Resource> ts_rb;
    // #421 per-dispatch timing (d3d12-dispatch-timing.txt): one timestamp
    // before the first dispatch and one after each, grown on demand.
    bool dispatch_timing = false;
    ComPtr<ID3D12QueryHeap> dts;
    ComPtr<ID3D12Resource> dts_rb;
    UINT dts_cap = 0;
    ComPtr<ID3D12Resource> staging;  // weight upload ring (kStagingBytes)
    ComPtr<ID3D12Resource> readback; // get_tensor on weights (kStagingBytes)
    std::uint8_t* staging_ptr = nullptr;
    std::uint8_t* readback_ptr = nullptr;
    // q8 activations for the next submission (CPU-written, GPU-read; grows on
    // demand to the largest single input, ~6 MiB for a 512-token ubatch).
    ComPtr<ID3D12Resource> xq;
    std::uint8_t* xq_ptr = nullptr;
    std::size_t xq_bytes = 0;
    d3d12c::QueueFence fence;
    UINT64 ts_freq = 0;
    LUID luid = {};
    double last_gpu_ms = 0.0;
    // Per-backend-lifetime counters, logged when the backend is freed.
    std::uint64_t n_calls = 0;
    std::uint64_t n_dispatches = 0; // MUL_MAT + GET_ROWS
    double wall_ms = 0.0;
    double gpu_ms = 0.0;
    // #421: activity since the last ggml_d3d12_take_stats(), guarded by mu.
    D3d12PhaseStats phase;
    std::mutex mu;
    bool ok = false;
    std::string error;
};

// Weight upload / readback ring. Mapped for the process lifetime and counted in
// its working set, so it stays small: 64 MiB each put the first GPU-layer
// smoke at 640 MB peak vs 311 MB on the CPU (D2b); 8 MiB costs a few more
// round trips at load only.
constexpr UINT64 kStagingBytes = 8ull << 20;

ComPtr<ID3D12RootSignature> make_root_sig(ID3D12Device* device, std::string* err) {
    D3D12_ROOT_PARAMETER p[4] = {};
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p[0].Constants.ShaderRegister = 0;
    p[0].Constants.Num32BitValues = 8;
    p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; // t0 weights
    p[1].Descriptor.ShaderRegister = 0;
    p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; // u0 output
    p[2].Descriptor.ShaderRegister = 0;
    p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; // u1 activations
    p[3].Descriptor.ShaderRegister = 1;
    for (auto& q : p)
        q.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = 4;
    desc.pParameters = p;
    auto serialize = d3d12_dyn::SerializeRootSignature();
    if (!serialize) {
        *err = "D3D12SerializeRootSignature not available";
        return {};
    }
    ComPtr<ID3DBlob> sig, blob_err;
    HRESULT hr = serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &blob_err);
    if (FAILED(hr)) {
        *err = d3d12c::hr_message("D3D12SerializeRootSignature", hr);
        return {};
    }
    ComPtr<ID3D12RootSignature> root;
    hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                     IID_PPV_ARGS(&root));
    if (FAILED(hr)) {
        *err = d3d12c::hr_message("CreateRootSignature", hr);
        return {};
    }
    return root;
}

// Fork: FLASH_ATTN_EXT — 24 root constants (D3d12FaParams) at b0 and root
// UAVs u0 (output), u1 Q, u2 K, u3 V, u4 mask, u5 sinks.
ComPtr<ID3D12RootSignature> make_root_sig_fa(ID3D12Device* device, std::string* err) {
    D3D12_ROOT_PARAMETER p[7] = {};
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p[0].Constants.ShaderRegister = 0;
    p[0].Constants.Num32BitValues = sizeof(D3d12FaParams) / 4;
    for (UINT i = 1; i < 7; ++i) {
        p[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        p[i].Descriptor.ShaderRegister = i - 1;
    }
    for (auto& q : p)
        q.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = 7;
    desc.pParameters = p;
    auto serialize = d3d12_dyn::SerializeRootSignature();
    if (!serialize) {
        *err = "D3D12SerializeRootSignature not available";
        return {};
    }
    ComPtr<ID3DBlob> sig, blob_err;
    HRESULT hr = serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &blob_err);
    if (FAILED(hr)) {
        *err = d3d12c::hr_message("D3D12SerializeRootSignature", hr);
        return {};
    }
    ComPtr<ID3D12RootSignature> root;
    hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                     IID_PPV_ARGS(&root));
    if (FAILED(hr)) {
        *err = d3d12c::hr_message("CreateRootSignature", hr);
        return {};
    }
    return root;
}

bool init_gpu(Gpu& g) {
    std::string& err = g.error;
    g.device = d3d12c::create_device(&err);
    if (!g.device)
        return false;
    D3D12_FEATURE_DATA_ARCHITECTURE1 arch = {};
    if (FAILED(g.device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &arch, sizeof(arch))) ||
        !arch.CacheCoherentUMA) {
        // D3D12_Host relies on a CPU-cached GPU-visible heap (D1: cc_uma=1 on Series S).
        err = "adapter is not cache-coherent UMA";
        return false;
    }
    g.luid = g.device->GetAdapterLuid();
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr = g.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&g.queue));
    if (FAILED(hr)) {
        err = d3d12c::hr_message("CreateCommandQueue", hr);
        return false;
    }
    hr = g.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g.alloc));
    if (SUCCEEDED(hr))
        hr = g.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g.alloc.Get(), nullptr,
                                         IID_PPV_ARGS(&g.cl));
    if (FAILED(hr)) {
        err = d3d12c::hr_message("CreateCommandList", hr);
        return false;
    }
    g.cl->Close();
    if (!g.fence.init(g.device.Get(), &err))
        return false;
    g.root = make_root_sig(g.device.Get(), &err);
    if (!g.root)
        return false;
    const void* blobs[kPsoCount][2] = {{kGgmlD3d12MmvQ40T64Dxil, kGgmlD3d12MmvQ40T128Dxil},
                                       {kGgmlD3d12MmvQ4KT64Dxil, kGgmlD3d12MmvQ4KT128Dxil},
                                       {kGgmlD3d12MmvQ6KT64Dxil, kGgmlD3d12MmvQ6KT128Dxil},
                                       {kGgmlD3d12MmvIq3XxsT64Dxil, kGgmlD3d12MmvIq3XxsT128Dxil},
                                       {kGgmlD3d12MmvIq4XsT64Dxil, kGgmlD3d12MmvIq4XsT128Dxil},
                                       {kGgmlD3d12MmvQ5KT64Dxil, kGgmlD3d12MmvQ5KT128Dxil}};
    const size_t sizes[kPsoCount][2] = {
        {kGgmlD3d12MmvQ40T64DxilSize, kGgmlD3d12MmvQ40T128DxilSize},
        {kGgmlD3d12MmvQ4KT64DxilSize, kGgmlD3d12MmvQ4KT128DxilSize},
        {kGgmlD3d12MmvQ6KT64DxilSize, kGgmlD3d12MmvQ6KT128DxilSize},
        {kGgmlD3d12MmvIq3XxsT64DxilSize, kGgmlD3d12MmvIq3XxsT128DxilSize},
        {kGgmlD3d12MmvIq4XsT64DxilSize, kGgmlD3d12MmvIq4XsT128DxilSize},
        {kGgmlD3d12MmvQ5KT64DxilSize, kGgmlD3d12MmvQ5KT128DxilSize}};
    for (int i = 0; i < kPsoCount; ++i) {
        for (int wd = 0; wd < 2; ++wd) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature = g.root.Get();
            pd.CS.pShaderBytecode = blobs[i][wd];
            pd.CS.BytecodeLength = sizes[i][wd];
            hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso[i][wd]));
            if (FAILED(hr)) {
                err = d3d12c::hr_message("CreateComputePipelineState", hr);
                return false;
            }
        }
    }
    {
        // Fork: tiled prefill kernels (Q4_0 has none and keeps per-column).
        struct TileBlob {
            int pso;
            const void* b64;
            size_t s64;
            const void* b128;
            size_t s128;
        };
        const TileBlob tb[] = {
            {kPsoQ4K, kGgmlD3d12MmtQ4KT64Dxil, kGgmlD3d12MmtQ4KT64DxilSize,
             kGgmlD3d12MmtQ4KT128Dxil, kGgmlD3d12MmtQ4KT128DxilSize},
            {kPsoQ5K, kGgmlD3d12MmtQ5KT64Dxil, kGgmlD3d12MmtQ5KT64DxilSize,
             kGgmlD3d12MmtQ5KT128Dxil, kGgmlD3d12MmtQ5KT128DxilSize},
            {kPsoQ6K, kGgmlD3d12MmtQ6KT64Dxil, kGgmlD3d12MmtQ6KT64DxilSize,
             kGgmlD3d12MmtQ6KT128Dxil, kGgmlD3d12MmtQ6KT128DxilSize},
            {kPsoIq3Xxs, kGgmlD3d12MmtIq3XxsT64Dxil, kGgmlD3d12MmtIq3XxsT64DxilSize,
             kGgmlD3d12MmtIq3XxsT128Dxil, kGgmlD3d12MmtIq3XxsT128DxilSize},
            {kPsoIq4Xs, kGgmlD3d12MmtIq4XsT64Dxil, kGgmlD3d12MmtIq4XsT64DxilSize,
             kGgmlD3d12MmtIq4XsT128Dxil, kGgmlD3d12MmtIq4XsT128DxilSize},
        };
        for (const TileBlob& t : tb) {
            for (int wd = 0; wd < 2; ++wd) {
                D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
                pd.pRootSignature = g.root.Get();
                pd.CS.pShaderBytecode = wd ? t.b128 : t.b64;
                pd.CS.BytecodeLength = wd ? t.s128 : t.s64;
                if (FAILED(g.device->CreateComputePipelineState(
                        &pd, IID_PPV_ARGS(&g.pso_tile[t.pso][wd]))))
                    g.pso_tile[t.pso][wd].Reset(); // per-column path still works
            }
        }
        if (FILE* f =
                _wfopen(utf8_to_wstring(resolve_local_path("d3d12-tile-off.txt")).c_str(), L"rb")) {
            std::fclose(f);
            g.tile_on = false;
        }
        log_output(std::string("[xllama] d3d12: tiled prefill ") + (g.tile_on ? "on" : "off") +
                   "\n");
    }
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = g.root.Get();
        pd.CS.pShaderBytecode = kGgmlD3d12GetRowsQ6KDxil;
        pd.CS.BytecodeLength = kGgmlD3d12GetRowsQ6KDxilSize;
        hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso_get_rows));
        if (FAILED(hr)) {
            err = d3d12c::hr_message("CreateComputePipelineState", hr);
            return false;
        }
    }
    {
        // Fork: flash attention. A failure here only leaves attention on the CPU.
        std::string fa_err;
        g.root_fa = make_root_sig_fa(g.device.Get(), &fa_err);
        if (g.root_fa) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature = g.root_fa.Get();
            pd.CS.pShaderBytecode = kGgmlD3d12FaDxil;
            pd.CS.BytecodeLength = kGgmlD3d12FaDxilSize;
            hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso_fa));
            if (FAILED(hr)) {
                fa_err = d3d12c::hr_message("CreateComputePipelineState(fa)", hr);
                g.pso_fa.Reset();
            }
        }
        if (!g.pso_fa)
            log_output("[xllama] d3d12: flash attention kernel unavailable: " + fa_err + "\n");
        // Fork: small ops share the flash-attention root signature.
        struct Blob {
            const void* b;
            size_t n;
            ComPtr<ID3D12PipelineState>* out;
        };
        const Blob blobs_ew[] = {
            {kGgmlD3d12EwAddDxil, kGgmlD3d12EwAddDxilSize, &g.pso_ew[0]},
            {kGgmlD3d12EwMulDxil, kGgmlD3d12EwMulDxilSize, &g.pso_ew[1]},
            {kGgmlD3d12EwSiluDxil, kGgmlD3d12EwSiluDxilSize, &g.pso_ew[2]},
            {kGgmlD3d12EwSwigluDxil, kGgmlD3d12EwSwigluDxilSize, &g.pso_ew[3]},
            {kGgmlD3d12EwRmsnormDxil, kGgmlD3d12EwRmsnormDxilSize, &g.pso_ew[4]},
            {kGgmlD3d12EwQuantq8kDxil, kGgmlD3d12EwQuantq8kDxilSize, &g.pso_q8k},
        };
        for (const Blob& bl : blobs_ew) {
            if (!g.root_fa)
                break;
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature = g.root_fa.Get();
            pd.CS.pShaderBytecode = bl.b;
            pd.CS.BytecodeLength = bl.n;
            if (FAILED(g.device->CreateComputePipelineState(&pd,
                                                            IID_PPV_ARGS(bl.out->GetAddressOf()))))
                bl.out->Reset();
        }
    }
    if (SUCCEEDED(g.queue->GetTimestampFrequency(&g.ts_freq)) && g.ts_freq > 0) {
        D3D12_QUERY_HEAP_DESC qhd = {};
        qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhd.Count = 2;
        g.device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&g.ts));
        g.ts_rb = d3d12c::create_buffer(g.device.Get(), 256, D3D12_HEAP_TYPE_READBACK,
                                        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                        "ts_rb", &err);
        if (!g.ts || !g.ts_rb) {
            g.ts.Reset();
            g.ts_rb.Reset();
            err.clear();
        }
    }
    if (g.ts) {
        if (FILE* f = _wfopen(
                utf8_to_wstring(resolve_local_path("d3d12-dispatch-timing.txt")).c_str(), L"rb")) {
            char config[16] = {};
            const std::size_t bytes = std::fread(config, 1, sizeof(config), f);
            std::fclose(f);
            g.dispatch_timing = d3d12_flag_enabled(std::string_view(config, bytes));
        }
    }
    log_output(std::string("[xllama] d3d12: dispatch_timing=") + (g.dispatch_timing ? "1" : "0") +
               "\n");
    g.staging = d3d12c::create_buffer(g.device.Get(), kStagingBytes, D3D12_HEAP_TYPE_UPLOAD,
                                      D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                      "staging", &err);
    g.readback = d3d12c::create_buffer(g.device.Get(), kStagingBytes, D3D12_HEAP_TYPE_READBACK,
                                       D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                       "readback", &err);
    if (!g.staging || !g.readback)
        return false;
    void* p = nullptr;
    if (FAILED(g.staging->Map(0, nullptr, &p)))
        return false;
    g.staging_ptr = static_cast<std::uint8_t*>(p);
    if (FAILED(g.readback->Map(0, nullptr, &p)))
        return false;
    g.readback_ptr = static_cast<std::uint8_t*>(p);
    return true;
}

Gpu& gpu() {
    static Gpu g;
    static std::once_flag once;
    std::call_once(once, [] { g.ok = init_gpu(g); });
    return g;
}

// Record into the shared list and run it to completion. Caller holds g.mu.
// `stats` (graph submissions only) receives the record and execute+wait times.
template <typename F> bool run_now(Gpu& g, F&& record, D3d12PhaseStats* stats = nullptr) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    g.alloc->Reset();
    g.cl->Reset(g.alloc.Get(), nullptr);
    record(g.cl.Get());
    g.cl->Close();
    const auto t1 = clock::now();
    ID3D12CommandList* lists[] = {g.cl.Get()};
    g.queue->ExecuteCommandLists(1, lists);
    // Fork: 60 s is far past any real submission (a 512-token ubatch is ~2 s).
    const bool ok = g.fence.signal_and_wait(g.queue.Get(), /*spin=*/true, 60000);
    if (stats) {
        stats->record_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->exec_wait_ms += std::chrono::duration<double, std::milli>(clock::now() - t1).count();
        ++stats->submissions;
    }
    return ok;
}

// --- Buffers ---

struct BufCtx {
    ComPtr<ID3D12Resource> res;
    D3D12_GPU_VIRTUAL_ADDRESS va = 0;
    std::uint8_t* host = nullptr; // mapped pointer (D3D12_Host only)
    bool is_host = false;
};

void* buf_base(ggml_backend_buffer_t b) {
    auto* c = static_cast<BufCtx*>(b->context);
    return c->is_host ? static_cast<void*>(c->host)
                      : reinterpret_cast<void*>(static_cast<std::uintptr_t>(c->va));
}

void buf_free(ggml_backend_buffer_t b) {
    delete static_cast<BufCtx*>(b->context);
}

std::size_t tensor_offset(const ggml_tensor* t) {
    return static_cast<std::size_t>(static_cast<const std::uint8_t*>(t->data) -
                                    static_cast<const std::uint8_t*>(buf_base(t->buffer)));
}

D3D12_GPU_VIRTUAL_ADDRESS tensor_va(const ggml_tensor* t) {
    return static_cast<BufCtx*>(t->buffer->context)->va + tensor_offset(t);
}

// D3D12_Host: plain memory.
void host_memset(ggml_backend_buffer_t, ggml_tensor* t, uint8_t v, size_t off, size_t size) {
    std::memset(static_cast<std::uint8_t*>(t->data) + off, v, size);
}
void host_set(ggml_backend_buffer_t, ggml_tensor* t, const void* data, size_t off, size_t size) {
    std::memcpy(static_cast<std::uint8_t*>(t->data) + off, data, size);
}
void host_get(ggml_backend_buffer_t, const ggml_tensor* t, void* data, size_t off, size_t size) {
    std::memcpy(data, static_cast<const std::uint8_t*>(t->data) + off, size);
}
bool host_cpy(ggml_backend_buffer_t, const ggml_tensor* src, ggml_tensor* dst) {
    if (!ggml_backend_buffer_is_host(src->buffer))
        return false;
    std::memcpy(dst->data, src->data, ggml_nbytes(src));
    return true;
}
void host_clear(ggml_backend_buffer_t b, uint8_t v) {
    std::memset(static_cast<BufCtx*>(b->context)->host, v, b->size);
}

const ggml_backend_buffer_i kHostBufIface = {
    /* .free_buffer   = */ buf_free,
    /* .get_base      = */ buf_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ host_memset,
    /* .set_tensor    = */ host_set,
    /* .get_tensor    = */ host_get,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ host_cpy,
    /* .clear         = */ host_clear,
    /* .reset         = */ nullptr,
};

// D3D12_Weights: DEFAULT heap, filled through the staging ring. The buffer
// stays in COMMON; buffers promote implicitly to COPY_DEST / shader read and
// decay back at the end of each submission.
void weights_write(ggml_backend_buffer_t b, std::size_t dst_off, const void* data, std::size_t size,
                   int fill) {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    auto* c = static_cast<BufCtx*>(b->context);
    for (std::size_t done = 0; done < size;) {
        const std::size_t chunk = std::min<std::size_t>(size - done, kStagingBytes);
        if (data)
            std::memcpy(g.staging_ptr, static_cast<const std::uint8_t*>(data) + done, chunk);
        else
            std::memset(g.staging_ptr, fill, chunk);
        run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            cl->CopyBufferRegion(c->res.Get(), dst_off + done, g.staging.Get(), 0, chunk);
        });
        done += chunk;
    }
}
void weights_memset(ggml_backend_buffer_t b, ggml_tensor* t, uint8_t v, size_t off, size_t size) {
    weights_write(b, tensor_offset(t) + off, nullptr, size, v);
}
void weights_set(ggml_backend_buffer_t b, ggml_tensor* t, const void* data, size_t off,
                 size_t size) {
    weights_write(b, tensor_offset(t) + off, data, size, 0);
}
void weights_get(ggml_backend_buffer_t b, const ggml_tensor* t, void* data, size_t off,
                 size_t size) {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    auto* c = static_cast<BufCtx*>(b->context);
    const std::size_t src_off = tensor_offset(t) + off;
    for (std::size_t done = 0; done < size;) {
        const std::size_t chunk = std::min<std::size_t>(size - done, kStagingBytes);
        run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            cl->CopyBufferRegion(g.readback.Get(), 0, c->res.Get(), src_off + done, chunk);
        });
        std::memcpy(static_cast<std::uint8_t*>(data) + done, g.readback_ptr, chunk);
        done += chunk;
    }
}
void weights_clear(ggml_backend_buffer_t b, uint8_t v) {
    weights_write(b, 0, nullptr, b->size, v);
}

const ggml_backend_buffer_i kWeightsBufIface = {
    /* .free_buffer   = */ buf_free,
    /* .get_base      = */ buf_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ weights_memset,
    /* .set_tensor    = */ weights_set,
    /* .get_tensor    = */ weights_get,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ weights_clear,
    /* .reset         = */ nullptr,
};

// --- Buffer types ---

ggml_backend_dev_t device_ptr();

const char* host_buft_name(ggml_backend_buffer_type_t) {
    return "D3D12_Host";
}
const char* weights_buft_name(ggml_backend_buffer_type_t) {
    return "D3D12_Weights";
}
size_t buft_alignment(ggml_backend_buffer_type_t) {
    return 256;
}
bool buft_is_host_true(ggml_backend_buffer_type_t) {
    return true;
}
// The Q4_0/Q6_K kernels read whole aligned dwords, up to 2 bytes past a
// tensor's last block; root descriptors have no bounds check, so pad.
size_t weights_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor* t) {
    return ggml_nbytes(t) + 16;
}

ggml_backend_buffer_t alloc_buf(ggml_backend_buffer_type_t buft, size_t size, bool host) {
    Gpu& g = gpu();
    if (!g.ok)
        return nullptr;
    // Fork: the flash-attention kernel reads Q8_0 rows by aligned dwords, up to
    // 4 bytes past a view's last block; root UAVs have no bounds check, so pad.
    size = std::max<size_t>(size + (host ? 16 : 0), 256);
    auto* c = new BufCtx;
    c->is_host = host;
    std::string err;
    if (host) {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_CUSTOM;
        hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
        hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        c->res = d3d12c::create_buffer_props(
            g.device.Get(), size, hp, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "D3D12_Host", &err);
        void* p = nullptr;
        if (c->res && SUCCEEDED(c->res->Map(0, nullptr, &p)))
            c->host = static_cast<std::uint8_t*>(p);
    } else {
        c->res = d3d12c::create_buffer(g.device.Get(), size, D3D12_HEAP_TYPE_DEFAULT,
                                       D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON,
                                       "D3D12_Weights", &err);
    }
    if (!c->res || (host && !c->host)) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "[xllama] d3d12: allocation of %zu bytes failed: %s\n",
                      size, err.c_str());
        log_output(msg);
        delete c;
        return nullptr;
    }
    c->va = c->res->GetGPUVirtualAddress();
    if (size >= (1u << 20)) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "[xllama] d3d12: %s buffer %.1f MiB\n",
                      host ? "D3D12_Host" : "D3D12_Weights", static_cast<double>(size) / 1048576.0);
        log_output(msg);
    }
    return ggml_backend_buffer_init(buft, host ? kHostBufIface : kWeightsBufIface, c, size);
}

ggml_backend_buffer_t host_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    return alloc_buf(buft, size, true);
}
ggml_backend_buffer_t weights_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    return alloc_buf(buft, size, false);
}

ggml_backend_buffer_type kHostBuft = {
    /* .iface = */
    {
        /* .get_name         = */ host_buft_name,
        /* .alloc_buffer     = */ host_alloc,
        /* .alloc_buffer_n   = */ nullptr,
        /* .get_alignment    = */ buft_alignment,
        /* .get_max_size     = */ nullptr,
        /* .get_alloc_size   = */ nullptr,
        /* .get_alloc_size_n = */ nullptr,
        /* .is_host          = */ buft_is_host_true,
    },
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};
ggml_backend_buffer_type kWeightsBuft = {
    /* .iface = */
    {
        /* .get_name         = */ weights_buft_name,
        /* .alloc_buffer     = */ weights_alloc,
        /* .alloc_buffer_n   = */ nullptr,
        /* .get_alignment    = */ buft_alignment,
        /* .get_max_size     = */ nullptr,
        /* .get_alloc_size   = */ weights_alloc_size,
        /* .get_alloc_size_n = */ nullptr,
        /* .is_host          = */ nullptr,
    },
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};

bool is_ours(ggml_backend_buffer_type_t buft) {
    return buft == &kHostBuft || buft == &kWeightsBuft;
}

// --- Backend ---

const char* backend_name(ggml_backend_t) {
    return "D3D12";
}
void backend_free(ggml_backend_t b) {
    Gpu& g = gpu();
    {
        std::lock_guard<std::mutex> lock(g.mu);
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "[xllama] d3d12: %llu graph_compute calls, %llu dispatches, %.1f ms wall "
                      "(%.1f ms GPU)\n",
                      static_cast<unsigned long long>(g.n_calls),
                      static_cast<unsigned long long>(g.n_dispatches), g.wall_ms, g.gpu_ms);
        log_output(msg);
        g.n_calls = g.n_dispatches = 0;
        g.wall_ms = g.gpu_ms = 0.0;
    }
    delete b;
}

// Grow the q8 activation scratch. Caller holds g.mu and has nothing in flight.
bool ensure_xq(Gpu& g, std::size_t bytes) {
    if (bytes <= g.xq_bytes)
        return true;
    const std::size_t want = std::max<std::size_t>(bytes, std::size_t(1) << 20);
    g.xq.Reset();
    g.xq_ptr = nullptr;
    g.xq_bytes = 0;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_CUSTOM;
    hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    std::string err;
    g.xq = d3d12c::create_buffer_props(g.device.Get(), want, hp,
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "D3D12_Xq", &err);
    void* p = nullptr;
    if (!g.xq || FAILED(g.xq->Map(0, nullptr, &p))) {
        log_output("[xllama] d3d12: q8 activation scratch allocation failed: " + err + "\n");
        g.xq.Reset();
        return false;
    }
    g.xq_ptr = static_cast<std::uint8_t*>(p);
    g.xq_bytes = want;
    return true;
}

// The tensor that owns a node's memory (follows view chains).
const ggml_tensor* base_of(const ggml_tensor* t) {
    while (t->view_src)
        t = t->view_src;
    return t;
}

// --- Fork: FLASH_ATTN_EXT ---

D3d12FlashAttnDesc fa_desc_of(const ggml_tensor* node) {
    const ggml_tensor* q = node->src[0];
    const ggml_tensor* k = node->src[1];
    const ggml_tensor* v = node->src[2];
    const ggml_tensor* m = node->src[3];
    D3d12FlashAttnDesc d;
    d.q_type = q->type;
    d.k_type = k->type;
    d.v_type = v->type;
    d.mask_type = m ? m->type : GGML_TYPE_COUNT;
    d.dst_type = node->type;
    d.dk = k->ne[0];
    d.dv = v->ne[0];
    d.n_q = q->ne[1];
    d.n_head = q->ne[2];
    d.n_head_kv = k->ne[2];
    d.n_kv = k->ne[1];
    d.ne3 = std::max({q->ne[3], k->ne[3], v->ne[3], node->ne[3], m ? m->ne[3] : 1});
    d.mask_ne2 = m ? m->ne[2] : 1;
    d.mask_ne1 = m ? m->ne[1] : d.n_q;
    std::memcpy(&d.max_bias, reinterpret_cast<const float*>(node->op_params) + 1, sizeof(float));
    d.rows_contiguous = q->nb[0] == ggml_type_size(q->type) &&
                        k->nb[0] == ggml_type_size(k->type) && v->nb[0] == ggml_type_size(v->type);
    d.dst_contiguous = ggml_is_contiguous(node);
    d.mask_contiguous = !m || ggml_is_contiguous(m);
    d.strides_even = k->nb[1] % 2 == 0 && k->nb[2] % 2 == 0 && v->nb[1] % 2 == 0 &&
                     v->nb[2] % 2 == 0 && q->nb[1] % 4 == 0 && q->nb[2] % 4 == 0 &&
                     (!m || (m->nb[1] % 2 == 0 && m->nb[2] % 2 == 0));
    // Largest byte offset the kernel forms in each view (+3 for alignment).
    const double lim = 4294967295.0 - 4096.0;
    auto span = [](const ggml_tensor* t) {
        double s = static_cast<double>(ggml_row_size(t->type, t->ne[0]));
        for (int i = 1; i < 4; ++i)
            s += static_cast<double>(t->ne[i] - 1) * static_cast<double>(t->nb[i]);
        return s;
    };
    d.strides_fit_u32 = span(q) < lim && span(k) < lim && span(v) < lim && (!m || span(m) < lim) &&
                        static_cast<double>(ggml_nbytes(node)) < lim;
    return d;
}

// Every source and the output in D3D12_Host (CPU-visible, UAV state), so the
// kernel reads them in place. Before allocation a buffer is still unknown.
bool fa_buffers_ok(const ggml_tensor* node) {
    for (int i = 0; i < 5; ++i) {
        const ggml_tensor* t = node->src[i];
        if (t && t->buffer && t->buffer->buft != &kHostBuft)
            return false;
    }
    return !node->buffer || node->buffer->buft == &kHostBuft;
}

// Root UAV address (rounded down to 4 bytes) and the remainder the kernel adds.
D3D12_GPU_VIRTUAL_ADDRESS fa_va(const ggml_tensor* t, std::uint32_t* off) {
    const D3D12_GPU_VIRTUAL_ADDRESS va = tensor_va(t);
    *off = static_cast<std::uint32_t>(va & 3u);
    return va - *off;
}

D3d12FaParams fa_params_of(const ggml_tensor* node) {
    const ggml_tensor* q = node->src[0];
    const ggml_tensor* k = node->src[1];
    const ggml_tensor* v = node->src[2];
    const ggml_tensor* m = node->src[3];
    D3d12FaParams p;
    p.n_kv = static_cast<std::uint32_t>(k->ne[1]);
    p.d = static_cast<std::uint32_t>(k->ne[0]);
    p.n_q = static_cast<std::uint32_t>(q->ne[1]);
    p.n_head = static_cast<std::uint32_t>(q->ne[2]);
    p.gqa = static_cast<std::uint32_t>(q->ne[2] / k->ne[2]);
    p.nbq1 = static_cast<std::uint32_t>(q->nb[1]);
    p.nbq2 = static_cast<std::uint32_t>(q->nb[2]);
    p.nbk1 = static_cast<std::uint32_t>(k->nb[1]);
    p.nbk2 = static_cast<std::uint32_t>(k->nb[2]);
    p.nbv1 = static_cast<std::uint32_t>(v->nb[1]);
    p.nbv2 = static_cast<std::uint32_t>(v->nb[2]);
    if (m) {
        p.has_mask = 1;
        p.nbm1 = static_cast<std::uint32_t>(m->nb[1]);
        p.nbm2 = static_cast<std::uint32_t>(m->nb[2]);
        p.mask_ne2 = static_cast<std::uint32_t>(m->ne[2]);
    }
    p.has_sinks = node->src[4] ? 1u : 0u;
    p.kv_q8 = k->type == GGML_TYPE_Q8_0 ? 1u : 0u;
    float scale = 1.f, softcap = 0.f;
    std::memcpy(&scale, reinterpret_cast<const float*>(node->op_params) + 0, sizeof(float));
    std::memcpy(&softcap, reinterpret_cast<const float*>(node->op_params) + 2, sizeof(float));
    d3d12_fa_scales(scale, softcap, &p.scale_bits, &p.softcap_bits);
    return p;
}

void record_fa(ID3D12GraphicsCommandList* cl, Gpu& g, const D3d12FaParams& p,
               D3D12_GPU_VIRTUAL_ADDRESS o, D3D12_GPU_VIRTUAL_ADDRESS q,
               D3D12_GPU_VIRTUAL_ADDRESS k, D3D12_GPU_VIRTUAL_ADDRESS v,
               D3D12_GPU_VIRTUAL_ADDRESS m, D3D12_GPU_VIRTUAL_ADDRESS s) {
    cl->SetComputeRootSignature(g.root_fa.Get());
    cl->SetPipelineState(g.pso_fa.Get());
    cl->SetComputeRoot32BitConstants(0, sizeof(D3d12FaParams) / 4, &p, 0);
    cl->SetComputeRootUnorderedAccessView(1, o);
    cl->SetComputeRootUnorderedAccessView(2, q);
    cl->SetComputeRootUnorderedAccessView(3, k);
    cl->SetComputeRootUnorderedAccessView(4, v);
    cl->SetComputeRootUnorderedAccessView(5, m);
    cl->SetComputeRootUnorderedAccessView(6, s);
    cl->Dispatch(p.n_q, p.n_head, 1);
}

// One FLASH_ATTN_EXT case on the GPU against d3d12_fa_emulate. Caller holds g.mu.
bool fa_check_case(Gpu& g, ggml_type kv, int d, int n_q, int n_head, int n_head_kv, int n_kv,
                   bool sinks, std::string* why) {
    std::mt19937 rng(99u + static_cast<unsigned>(n_kv));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    const std::size_t row = ggml_row_size(kv, d);
    const std::size_t q_bytes = static_cast<std::size_t>(d) * n_head * n_q * 4;
    const std::size_t kv_bytes = row * n_head_kv * n_kv;
    const std::size_t m_bytes = static_cast<std::size_t>(n_kv) * n_q * 2;
    const std::size_t s_bytes = static_cast<std::size_t>(n_head) * 4;
    auto up = [](std::size_t b) { return (b + 255) & ~std::size_t(255); };
    const std::size_t o_off = 0, q_off = up(q_bytes), k_off = q_off + up(q_bytes),
                      v_off = k_off + up(kv_bytes), m_off = v_off + up(kv_bytes),
                      s_off = m_off + up(m_bytes), total = s_off + up(s_bytes) + 256;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_CUSTOM;
    hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    std::string err;
    ComPtr<ID3D12Resource> res = d3d12c::create_buffer_props(
        g.device.Get(), total, hp, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "fa_check", &err);
    void* mp = nullptr;
    if (!res || FAILED(res->Map(0, nullptr, &mp))) {
        *why = "buffer: " + err;
        return false;
    }
    auto* b = static_cast<std::uint8_t*>(mp);
    std::memset(b, 0, total);
    auto* qf = reinterpret_cast<float*>(b + q_off);
    for (std::size_t i = 0; i < q_bytes / 4; ++i)
        qf[i] = uni(rng);
    for (std::size_t off : {k_off, v_off}) {
        std::vector<float> f(static_cast<std::size_t>(d) * n_head_kv * n_kv);
        for (float& x : f)
            x = uni(rng);
        ggml_quantize_chunk(kv, f.data(), b + off, 0, n_head_kv * n_kv, d, nullptr);
    }
    auto* mh = reinterpret_cast<ggml_fp16_t*>(b + m_off);
    for (int r = 0; r < n_q; ++r)
        for (int i = 0; i < n_kv; ++i)
            mh[r * n_kv + i] =
                ggml_fp32_to_fp16((i > n_kv - n_q + r || rng() % 9u == 0u) ? -INFINITY : 0.f);
    auto* sf = reinterpret_cast<float*>(b + s_off);
    for (int i = 0; i < n_head; ++i)
        sf[i] = uni(rng);
    // Layout as llama's permuted cache views: [d, n_kv, n_head_kv], rows
    // n_head_kv apart.
    D3d12FaParams p;
    p.n_kv = static_cast<std::uint32_t>(n_kv);
    p.d = static_cast<std::uint32_t>(d);
    p.n_q = static_cast<std::uint32_t>(n_q);
    p.n_head = static_cast<std::uint32_t>(n_head);
    p.gqa = static_cast<std::uint32_t>(n_head / n_head_kv);
    p.nbq1 = static_cast<std::uint32_t>(d * n_head * 4);
    p.nbq2 = static_cast<std::uint32_t>(d * 4);
    p.nbk1 = p.nbv1 = static_cast<std::uint32_t>(row * n_head_kv);
    p.nbk2 = p.nbv2 = static_cast<std::uint32_t>(row);
    p.has_mask = 1;
    p.nbm1 = static_cast<std::uint32_t>(n_kv * 2);
    p.nbm2 = static_cast<std::uint32_t>(n_kv * 2 * n_q);
    p.has_sinks = sinks ? 1u : 0u;
    p.kv_q8 = kv == GGML_TYPE_Q8_0 ? 1u : 0u;
    d3d12_fa_scales(1.f / std::sqrt(static_cast<float>(d)), 0.f, &p.scale_bits, &p.softcap_bits);
    const D3D12_GPU_VIRTUAL_ADDRESS va = res->GetGPUVirtualAddress();
    const bool ran = run_now(g, [&](ID3D12GraphicsCommandList* cl) {
        record_fa(cl, g, p, va + o_off, va + q_off, va + k_off, va + v_off, va + m_off, va + s_off);
    });
    if (!ran) {
        *why = "dispatch failed";
        return false;
    }
    std::vector<float> want(static_cast<std::size_t>(d) * n_head * n_q);
    d3d12_fa_emulate(p, b + q_off, b + k_off, b + v_off, b + m_off, sf, want.data());
    const auto* got = reinterpret_cast<const float*>(b + o_off);
    double max_ref = 0.0, max_diff = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (!std::isfinite(got[i])) {
            *why = "non-finite output";
            return false;
        }
        max_ref = std::max(max_ref, std::fabs(static_cast<double>(want[i])));
        max_diff = std::max(max_diff, std::fabs(static_cast<double>(want[i]) - got[i]));
    }
    const double rel = max_ref > 0 ? max_diff / max_ref : max_diff;
    char msg[160];
    std::snprintf(msg, sizeof(msg), "%s d=%d n_q=%d n_kv=%d rel_err=%.2e", ggml_type_name(kv), d,
                  n_q, n_kv, rel);
    *why = msg;
    return rel < 1e-3;
}

// Flash attention on the GPU: kernel built, no d3d12-fa-off.txt, and the
// first use's check against the emulation passed. Decided once.
bool fa_enabled() {
    static bool on = false;
    static std::once_flag once;
    std::call_once(once, [] {
        Gpu& g = gpu();
        if (!g.ok || !g.pso_fa)
            return;
        if (FILE* f =
                _wfopen(utf8_to_wstring(resolve_local_path("d3d12-fa-off.txt")).c_str(), L"rb")) {
            std::fclose(f);
            log_output("[xllama] d3d12: flash attention off (d3d12-fa-off.txt)\n");
            return;
        }
        std::lock_guard<std::mutex> lock(g.mu);
        std::string why1, why2;
        const bool ok1 = fa_check_case(g, GGML_TYPE_Q8_0, 256, 5, 4, 2, 150, false, &why1);
        const bool ok2 = ok1 && fa_check_case(g, GGML_TYPE_F16, 128, 3, 4, 4, 70, true, &why2);
        on = ok1 && ok2;
        log_output(std::string("[xllama] d3d12: flash attention ") + (on ? "on" : "OFF") +
                   " (check " + why1 + (why2.empty() ? "" : "; " + why2) + ")\n");
    });
    return on;
}

// --- Fork: small ops (ADD, MUL, SILU, SWIGLU, RMS_NORM) and the q8_K quantizer ---

bool ew_buffers_ok(const ggml_tensor* node) {
    for (int i = 0; i < 2; ++i) {
        const ggml_tensor* t = node->src[i];
        if (t && t->buffer && t->buffer->buft != &kHostBuft)
            return false;
    }
    return !node->buffer || node->buffer->buft == &kHostBuft;
}

void record_ew(ID3D12GraphicsCommandList* cl, Gpu& g, ID3D12PipelineState* pso,
               const D3d12EwParams& p, D3D12_GPU_VIRTUAL_ADDRESS d, D3D12_GPU_VIRTUAL_ADDRESS s0,
               D3D12_GPU_VIRTUAL_ADDRESS s1, UINT gx, UINT gy, UINT gz) {
    cl->SetComputeRootSignature(g.root_fa.Get());
    cl->SetPipelineState(pso);
    cl->SetComputeRoot32BitConstants(0, sizeof(D3d12EwParams) / 4, &p, 0);
    cl->SetComputeRootUnorderedAccessView(1, d);
    cl->SetComputeRootUnorderedAccessView(2, s0);
    cl->SetComputeRootUnorderedAccessView(3, s1);
    cl->SetComputeRootUnorderedAccessView(4, d);
    cl->SetComputeRootUnorderedAccessView(5, d);
    cl->SetComputeRootUnorderedAccessView(6, d);
    cl->Dispatch(gx, gy, gz);
}

// A CPU-visible scratch buffer for the first-use checks. Caller holds g.mu.
bool check_buffer(Gpu& g, std::size_t bytes, ComPtr<ID3D12Resource>* res, std::uint8_t** ptr) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_CUSTOM;
    hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    std::string err;
    *res = d3d12c::create_buffer_props(g.device.Get(), bytes, hp,
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "ew_check", &err);
    void* mp = nullptr;
    if (!*res || FAILED((*res)->Map(0, nullptr, &mp)))
        return false;
    *ptr = static_cast<std::uint8_t*>(mp);
    std::memset(*ptr, 0, bytes);
    return true;
}

// Every small op once on the GPU against d3d12_ew_emulate, and the quantizer
// against d3d12_quant_q8k_emulate. Caller holds g.mu.
bool ew_check(Gpu& g, std::string* why) {
    for (const auto& pso : g.pso_ew)
        if (!pso) {
            *why = "kernel missing";
            return false;
        }
    if (!g.pso_q8k) {
        *why = "quantizer missing";
        return false;
    }
    constexpr std::uint32_t kNe0 = 300, kRows = 7, kPlanes = 2;
    const std::size_t plane = std::size_t(kNe0) * 2 * kRows * 4; // room for a 2x-wide src
    const std::size_t s0_off = 0, s1_off = plane * kPlanes, d_off = 2 * plane * kPlanes,
                      total = 3 * plane * kPlanes + 4096;
    ComPtr<ID3D12Resource> res;
    std::uint8_t* b = nullptr;
    if (!check_buffer(g, total, &res, &b)) {
        *why = "buffer";
        return false;
    }
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> uni(-2.f, 2.f);
    for (std::size_t i = 0; i < d_off / 4; ++i)
        reinterpret_cast<float*>(b)[i] = uni(rng);
    const D3D12_GPU_VIRTUAL_ADDRESS va = res->GetGPUVirtualAddress();
    double worst = 0.0;
    for (int k = 0; k < 5; ++k) {
        const auto op = static_cast<D3d12EwOp>(k);
        D3d12EwParams p;
        p.ne0 = kNe0;
        p.ne1 = kRows;
        p.ne2 = kPlanes;
        p.s0_nb1 = op == D3d12EwOp::Swiglu ? 2 * kNe0 * 4 : kNe0 * 4;
        p.s0_nb2 = p.s0_nb1 * kRows;
        p.d_nb1 = kNe0 * 4;
        p.d_nb2 = p.d_nb1 * kRows;
        p.d_nb3 = p.d_nb2 * kPlanes;
        p.s0_nb3 = p.s0_nb2 * kPlanes;
        if (op == D3d12EwOp::Mul) { // a norm weight: one row, broadcast
            p.s1_ne0 = kNe0;
        } else {
            p.s1_ne0 = kNe0;
            p.s1_ne1 = kRows;
            p.s1_ne2 = kPlanes;
            p.s1_nb1 = kNe0 * 4;
            p.s1_nb2 = p.s1_nb1 * kRows;
        }
        if (op == D3d12EwOp::RmsNorm) {
            const float eps = 1e-6f;
            std::memcpy(&p.p0, &eps, 4);
        }
        if (op == D3d12EwOp::Swiglu) { // fused gate|up rows
            p.p1 = 0;
            p.p2 = kNe0;
        }
        const bool ran = run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            record_ew(cl, g, g.pso_ew[k].Get(), p, va + d_off, va + s0_off, va + s1_off, kRows,
                      kPlanes, 1);
        });
        if (!ran) {
            *why = "dispatch failed";
            return false;
        }
        std::vector<std::uint8_t> want(plane * kPlanes);
        d3d12_ew_emulate(op, p, b + s0_off, b + s1_off, want.data());
        const auto* wf = reinterpret_cast<const float*>(want.data());
        const auto* gf = reinterpret_cast<const float*>(b + d_off);
        double mr = 0, md = 0;
        for (std::size_t i = 0; i < std::size_t(kNe0) * kRows * kPlanes; ++i) {
            if (!std::isfinite(gf[i])) {
                *why = "non-finite output";
                return false;
            }
            mr = std::max(mr, std::fabs(static_cast<double>(wf[i])));
            md = std::max(md, std::fabs(static_cast<double>(wf[i]) - gf[i]));
        }
        worst = std::max(worst, mr > 0 ? md / mr : md);
    }
    // Quantizer: 3 rows of 512 from s0, into d.
    {
        D3d12EwParams p;
        p.ne0 = 512;
        p.s0_nb1 = 512 * 4;
        p.d_nb1 = 2 * 292;
        const bool ran = run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            record_ew(cl, g, g.pso_q8k.Get(), p, va + d_off, va + s0_off, va + s0_off, 2, 3, 1);
        });
        if (!ran) {
            *why = "quantize dispatch failed";
            return false;
        }
        std::vector<std::uint8_t> want(3 * 2 * 292);
        d3d12_quant_q8k_emulate(reinterpret_cast<const float*>(b + s0_off), 512 * 4, 512, 3,
                                want.data(), 2 * 292);
        // Dequantized values within one step (GPU division may differ by an ulp).
        for (int blk = 0; blk < 6; ++blk) {
            const std::uint8_t* w = want.data() + blk * 292;
            const std::uint8_t* q = b + d_off + blk * 292;
            float dw, dg;
            std::memcpy(&dw, w, 4);
            std::memcpy(&dg, q, 4);
            if (std::fabs(dw - dg) > 1e-5f * std::fabs(dw)) {
                *why = "q8_K scale mismatch";
                return false;
            }
            for (int j = 0; j < 256; ++j) {
                const int a = static_cast<std::int8_t>(w[4 + j]);
                const int c = static_cast<std::int8_t>(q[4 + j]);
                if (a - c > 1 || c - a > 1) {
                    *why = "q8_K value mismatch";
                    return false;
                }
            }
        }
    }
    char msg[96];
    std::snprintf(msg, sizeof(msg), "rel_err=%.2e", worst);
    *why = msg;
    return worst < 1e-4;
}

// Set once ew_enabled() has decided; graph_compute (which holds g.mu and so
// cannot run the check) reads it to quantize GPU-produced inputs on the GPU.
std::atomic<int> g_ew_state{0}; // 0 undecided, 1 on, 2 off
// Which small ops may run (bit per D3d12EwOp, bit 5 = the q8_K quantizer).
// LocalState\d3d12-ew-ops.txt narrows it: any of "add mul silu swiglu rms_norm
// quant", e.g. to find which one misbehaves without a new build.
std::atomic<unsigned> g_ew_mask{0x3fu};
constexpr unsigned kEwQuantBit = 1u << 5;

// Small ops on the GPU: kernels built, no d3d12-ew-off.txt, first-use check passed.
bool ew_enabled() {
    static bool on = false;
    static std::once_flag once;
    std::call_once(once, [] {
        Gpu& g = gpu();
        if (!g.ok || !g.root_fa)
            return;
        if (FILE* f =
                _wfopen(utf8_to_wstring(resolve_local_path("d3d12-ew-off.txt")).c_str(), L"rb")) {
            std::fclose(f);
            log_output("[xllama] d3d12: small ops off (d3d12-ew-off.txt)\n");
            return;
        }
        if (FILE* f =
                _wfopen(utf8_to_wstring(resolve_local_path("d3d12-ew-ops.txt")).c_str(), L"rb")) {
            char buf[256] = {};
            const std::size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
            std::fclose(f);
            const std::string t(buf, n);
            unsigned m = 0;
            const char* names[] = {"add", "mul", "silu", "swiglu", "rms_norm", "quant"};
            for (unsigned i = 0; i < 6; ++i) {
                // whole words only: "mul" must not match inside another name
                for (std::size_t at = t.find(names[i]); at != std::string::npos;
                     at = t.find(names[i], at + 1)) {
                    const std::size_t end = at + std::strlen(names[i]);
                    const bool l =
                        at == 0 ||
                        (!std::isalnum(static_cast<unsigned char>(t[at - 1])) && t[at - 1] != '_');
                    const bool r =
                        end >= t.size() ||
                        (!std::isalnum(static_cast<unsigned char>(t[end])) && t[end] != '_');
                    if (l && r) {
                        m |= 1u << i;
                        break;
                    }
                }
            }
            g_ew_mask = m;
            log_output("[xllama] d3d12: small ops limited to '" + t + "'\n");
        }
        std::lock_guard<std::mutex> lock(g.mu);
        std::string why;
        on = ew_check(g, &why);
        g_ew_state = on ? 1 : 2;
        log_output(std::string("[xllama] d3d12: small ops on GPU ") + (on ? "on" : "OFF") +
                   " (check " + why + ")\n");
    });
    return on;
}

// One dispatch: a MUL_MAT (u1 = its quantized input) or a GET_ROWS (u1 = its
// row ids), both staged in g.xq.
struct Op {
    const ggml_tensor* node;
    std::size_t xq_off;   // staged input in g.xq
    std::uint32_t xq_row; // bytes per quantized column (MUL_MAT)
    // Fork: quant_x set = quantize that GPU-produced f32 input into g.xq at
    // xq_off (rows xq_row apart) for the MUL_MAT `node` that follows.
    const ggml_tensor* quant_x = nullptr;
    bool barrier = false; // wait for earlier dispatches (memory dependency)
};

// Room for `count` per-dispatch timestamps. Caller holds g.mu, nothing in flight.
bool ensure_dts(Gpu& g, UINT count) {
    if (count <= g.dts_cap)
        return true;
    const UINT want = std::max<UINT>(count, 256);
    g.dts.Reset();
    g.dts_rb.Reset();
    g.dts_cap = 0;
    D3D12_QUERY_HEAP_DESC qhd = {};
    qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qhd.Count = want;
    std::string err;
    if (FAILED(g.device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&g.dts))))
        return false;
    g.dts_rb =
        d3d12c::create_buffer(g.device.Get(), ((std::size_t(want) * 8) + 255) & ~std::size_t(255),
                              D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                              D3D12_RESOURCE_STATE_COPY_DEST, "dts_rb", &err);
    if (!g.dts_rb) {
        g.dts.Reset();
        return false;
    }
    g.dts_cap = want;
    return true;
}

// What a dispatch is, for the per-dispatch profile.
D3d12DispatchKind dispatch_kind(const ggml_tensor* node) {
    const ggml_tensor* w = node->src[0];
    D3d12DispatchKind k;
    k.type = ggml_type_name(w->type);
    k.cols = d3d12_cols_class(node->ne[1]);
    if (node->op != GGML_OP_FLASH_ATTN_EXT && node->op != GGML_OP_GET_ROWS &&
        node->op != GGML_OP_MUL_MAT) {
        k.op = ggml_op_desc(node);
        k.type = "f32";
    } else if (node->op == GGML_OP_FLASH_ATTN_EXT) {
        k.op = "flash_attn";
        k.type = ggml_type_name(node->src[1]->type);
        k.cols = d3d12_cols_class(node->ne[2]);
    } else if (node->op == GGML_OP_GET_ROWS) {
        k.op = "get_rows";
    } else {
        k.op = "mul_mat";
        k.threads = d3d12_mm_threads(w->ne[0]);
    }
    return k;
}

D3d12LayerKey dispatch_layer_key(const ggml_tensor* node) {
    D3d12LayerKey k;
    k.op = node->op == GGML_OP_FLASH_ATTN_EXT ? "flash_attn"
           : node->op == GGML_OP_GET_ROWS     ? "get_rows"
           : node->op == GGML_OP_MUL_MAT      ? "mul_mat"
                                              : ggml_op_desc(node);
    const char* weight = node->src[0] ? node->src[0]->name : "";
    k.layer = d3d12_layer_from_names(weight, node->name);
    return k;
}

// Record and run one submission. Caller holds g.mu.
bool submit_ops(Gpu& g, const std::vector<Op>& ops) {
    const bool ts = g.ts && g.ts_rb;
    const bool dt = ts && g.dispatch_timing && ensure_dts(g, static_cast<UINT>(ops.size() + 1));
    // A UAV barrier before each timestamp waits for the dispatch to finish, so
    // the interval between two timestamps belongs to one dispatch.
    D3D12_RESOURCE_BARRIER uav = {};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav.UAV.pResource = nullptr;
    double exec_before = 0.0;
    if (dt)
        exec_before = g.phase.exec_wait_ms;
    const bool ran = run_now(
        g,
        [&](ID3D12GraphicsCommandList* cl) {
            cl->SetComputeRootSignature(g.root.Get());
            if (ts)
                cl->EndQuery(g.ts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            if (dt)
                cl->EndQuery(g.dts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            const D3D12_GPU_VIRTUAL_ADDRESS xq_va = g.xq->GetGPUVirtualAddress();
            UINT stamp = 0;
            const auto after_dispatch = [&]() {
                if (!dt)
                    return;
                cl->ResourceBarrier(1, &uav);
                cl->EndQuery(g.dts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, ++stamp);
            };
            for (const Op& op : ops) {
                const ggml_tensor* node = op.node;
                const ggml_tensor* w = node->src[0];
                if (op.barrier && !dt) // with dt every dispatch is already followed by one
                    cl->ResourceBarrier(1, &uav);
                if (op.quant_x) {
                    // Fork: quantize a GPU-produced activation into g.xq.
                    const ggml_tensor* x = op.quant_x;
                    D3d12EwParams p;
                    p.ne0 = static_cast<std::uint32_t>(x->ne[0]);
                    p.s0_nb1 = static_cast<std::uint32_t>(x->nb[1]);
                    p.d_nb1 = op.xq_row;
                    const D3D12_GPU_VIRTUAL_ADDRESS xs = fa_va(x, &p.s0_off);
                    record_ew(cl, g, g.pso_q8k.Get(), p, xq_va + op.xq_off, xs, xs,
                              static_cast<UINT>(x->ne[0] / 256), static_cast<UINT>(x->ne[1]), 1);
                    cl->SetComputeRootSignature(g.root.Get());
                    after_dispatch();
                    continue;
                }
                if (const D3d12EwOp eo = d3d12_ew_op(node); eo != D3d12EwOp::None &&
                                                            node->op != GGML_OP_MUL_MAT &&
                                                            node->op != GGML_OP_GET_ROWS) {
                    D3d12EwParams p = d3d12_ew_params(node);
                    const D3D12_GPU_VIRTUAL_ADDRESS d = fa_va(node, &p.d_off);
                    const D3D12_GPU_VIRTUAL_ADDRESS a = fa_va(node->src[0], &p.s0_off);
                    const D3D12_GPU_VIRTUAL_ADDRESS b =
                        node->src[1] ? fa_va(node->src[1], &p.s1_off) : a;
                    record_ew(cl, g, g.pso_ew[static_cast<int>(eo)].Get(), p, d, a, b, p.ne1, p.ne2,
                              p.ne3);
                    cl->SetComputeRootSignature(g.root.Get());
                    after_dispatch();
                    continue;
                }
                if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                    D3d12FaParams p = fa_params_of(node);
                    std::uint32_t o_rem = 0;
                    const D3D12_GPU_VIRTUAL_ADDRESS o = fa_va(node, &o_rem);
                    const D3D12_GPU_VIRTUAL_ADDRESS q = fa_va(node->src[0], &p.q_off);
                    const D3D12_GPU_VIRTUAL_ADDRESS k = fa_va(node->src[1], &p.k_off);
                    const D3D12_GPU_VIRTUAL_ADDRESS v = fa_va(node->src[2], &p.v_off);
                    D3D12_GPU_VIRTUAL_ADDRESS m = o, s = o;
                    if (node->src[3])
                        m = fa_va(node->src[3], &p.m_off);
                    if (node->src[4]) {
                        std::uint32_t s_rem = 0;
                        s = fa_va(node->src[4], &s_rem);
                        GGML_ASSERT(s_rem == 0);
                    }
                    GGML_ASSERT(o_rem == 0 && p.q_off % 4 == 0); // f32 views
                    record_fa(cl, g, p, o, q, k, v, m, s);
                    cl->SetComputeRootSignature(g.root.Get());
                    after_dispatch();
                    continue;
                }
                if (node->op == GGML_OP_GET_ROWS) {
                    const std::uint32_t c[8] = {
                        static_cast<std::uint32_t>(node->ne[1]),
                        static_cast<std::uint32_t>(w->ne[0]),
                        static_cast<std::uint32_t>(w->ne[0] / kD3d12Chunk),
                        static_cast<std::uint32_t>(w->nb[1]),
                        0,
                        static_cast<std::uint32_t>(node->nb[1] / sizeof(float)),
                        0,
                        0};
                    cl->SetPipelineState(g.pso_get_rows.Get());
                    cl->SetComputeRoot32BitConstants(0, 8, c, 0);
                    cl->SetComputeRootShaderResourceView(1, tensor_va(w));
                    cl->SetComputeRootUnorderedAccessView(2, tensor_va(node));
                    cl->SetComputeRootUnorderedAccessView(3, xq_va + op.xq_off);
                    cl->Dispatch(static_cast<UINT>(w->ne[0] / kD3d12Chunk),
                                 static_cast<UINT>(node->ne[1]), 1);
                    after_dispatch();
                    continue;
                }
                D3d12Dispatch d = d3d12_mm_dispatch(node->ne[0], node->ne[1]);
                const int wide = d3d12_mm_threads(w->ne[0]) == kD3d12MmvThreadsLong ? 1 : 0;
                ID3D12PipelineState* pso = g.pso[pso_for(w->type)][wide].Get();
                // Fork: several columns (prompt reading) use the tiled kernel,
                // which loads each weight block once per kD3d12Tile columns.
                if (g.tile_on && node->ne[1] > 1 && g.pso_tile[pso_for(w->type)][wide]) {
                    pso = g.pso_tile[pso_for(w->type)][wide].Get();
                    d.groups_y =
                        static_cast<std::uint32_t>((node->ne[1] + kD3d12Tile - 1) / kD3d12Tile);
                }
                const std::uint32_t c[8] = {
                    static_cast<std::uint32_t>(w->ne[1]),
                    static_cast<std::uint32_t>(w->ne[0]),
                    static_cast<std::uint32_t>(w->ne[0] / kD3d12Chunk),
                    static_cast<std::uint32_t>(w->nb[1]),
                    op.xq_row,
                    static_cast<std::uint32_t>(node->nb[1] / sizeof(float)),
                    static_cast<std::uint32_t>(node->ne[1]), // ncols (tiled)
                    0};
                cl->SetPipelineState(pso);
                cl->SetComputeRoot32BitConstants(0, 8, c, 0);
                cl->SetComputeRootShaderResourceView(1, tensor_va(w));
                cl->SetComputeRootUnorderedAccessView(2, tensor_va(node));
                cl->SetComputeRootUnorderedAccessView(3, xq_va + op.xq_off);
                cl->Dispatch(d.groups_x, d.groups_y, 1);
                after_dispatch();
            }
            if (dt)
                cl->ResolveQueryData(g.dts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0,
                                     static_cast<UINT>(ops.size() + 1), g.dts_rb.Get(), 0);
            if (ts) {
                cl->EndQuery(g.ts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
                cl->ResolveQueryData(g.ts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, g.ts_rb.Get(),
                                     0);
            }
        },
        &g.phase);
    if (!ran) {
        // Fork: name what was in the submission that failed or hung.
        std::string list;
        for (std::size_t i = 0; i < ops.size() && i < 48; ++i) {
            const ggml_tensor* n = ops[i].node;
            char one[96];
            std::snprintf(one, sizeof(one), "%s%s[%lld,%lld,%lld]%s ",
                          ops[i].quant_x ? "quant>" : "", ggml_op_desc(n),
                          static_cast<long long>(n->ne[0]), static_cast<long long>(n->ne[1]),
                          static_cast<long long>(n->ne[2]), ops[i].barrier ? "|" : "");
            list += one;
        }
        log_output("[xllama] d3d12: submission of " + std::to_string(ops.size()) +
                   " dispatches failed or timed out: " + list + "\n");
        return false;
    }
    bool ts_ok = false;
    double submission_gpu_ms = 0.0;
    if (ts && g.ts_freq > 0) {
        void* p = nullptr;
        if (SUCCEEDED(g.ts_rb->Map(0, nullptr, &p))) {
            const auto* t = static_cast<const std::uint64_t*>(p);
            const double ms = t[1] > t[0] ? 1000.0 * static_cast<double>(t[1] - t[0]) /
                                                static_cast<double>(g.ts_freq)
                                          : 0.0;
            g.last_gpu_ms += ms;
            ts_ok = t[1] > t[0];
            if (ts_ok) {
                g.phase.gpu_ms += ms;
                submission_gpu_ms = ms;
            }
            g.ts_rb->Unmap(0, nullptr);
        }
    }
    ++(ts_ok ? g.phase.ts_valid : g.phase.ts_missing);
    if (dt) {
        // wait = exec_wait − submission GPU time. Missing GPU timestamps leave
        // wait as missing; it is not filled in from wall time.
        const std::optional<double> wait_share =
            ts_ok && !ops.empty()
                ? std::optional<double>((g.phase.exec_wait_ms - exec_before - submission_gpu_ms) /
                                        static_cast<double>(ops.size()))
                : std::nullopt;
        void* p = nullptr;
        const D3D12_RANGE range = {0, (ops.size() + 1) * sizeof(std::uint64_t)};
        if (SUCCEEDED(g.dts_rb->Map(0, &range, &p))) {
            const auto* t = static_cast<const std::uint64_t*>(p);
            for (std::size_t i = 0; i < ops.size(); ++i) {
                const ggml_tensor* node = ops[i].node;
                const ggml_tensor* w = node->src[0];
                if (t[i + 1] <= t[i]) {
                    g.phase.dispatch.note_missing();
                    g.phase.layer.add(dispatch_layer_key(node), std::nullopt, wait_share);
                    continue;
                }
                const double ms =
                    1000.0 * static_cast<double>(t[i + 1] - t[i]) / static_cast<double>(g.ts_freq);
                const bool small_op = ops[i].quant_x || (node->op != GGML_OP_MUL_MAT &&
                                                         node->op != GGML_OP_GET_ROWS &&
                                                         node->op != GGML_OP_FLASH_ATTN_EXT);
                const double bytes =
                    ops[i].quant_x ? static_cast<double>(ggml_nbytes(ops[i].quant_x))
                    : small_op     ? static_cast<double>(ggml_nbytes(node))
                    : node->op == GGML_OP_FLASH_ATTN_EXT
                        ? 2.0 * static_cast<double>(ggml_nbytes(node->src[1]))
                    : node->op == GGML_OP_GET_ROWS ? static_cast<double>(w->nb[1]) * node->ne[1]
                                                   : static_cast<double>(w->nb[1]) * w->ne[1];
                D3d12DispatchKind kind = dispatch_kind(node);
                if (ops[i].quant_x) {
                    kind.op = "quant_q8k";
                    kind.type = "f32";
                }
                g.phase.dispatch.add(kind, ms, bytes, node->ne[1]);
                g.phase.layer.add(dispatch_layer_key(node), ms, wait_share);
            }
            const D3D12_RANGE none = {0, 0};
            g.dts_rb->Unmap(0, &none);
        } else {
            for (std::size_t i = 0; i < ops.size(); ++i) {
                g.phase.dispatch.note_missing();
                g.phase.layer.add(dispatch_layer_key(ops[i].node), std::nullopt, wait_share);
            }
        }
    }
    g.n_dispatches += ops.size();
    g.phase.dispatches += ops.size();
    return true;
}

ggml_status backend_graph_compute(ggml_backend_t, ggml_cgraph* cgraph) {
    Gpu& g = gpu();
    std::vector<const ggml_tensor*> mm;
    for (int i = 0; i < ggml_graph_n_nodes(cgraph); ++i) {
        const ggml_tensor* node = ggml_graph_node(cgraph, i);
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)
            continue;
        switch (node->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_GET_ROWS:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_RMS_NORM:
        case GGML_OP_UNARY:
        case GGML_OP_GLU:
            mm.push_back(node);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;
        default:
            GGML_ABORT("d3d12: unsupported op %s", ggml_op_desc(node));
        }
    }
    if (mm.empty())
        return GGML_STATUS_SUCCESS;

    std::lock_guard<std::mutex> lock(g.mu);
    const auto t0 = std::chrono::steady_clock::now();
    g.last_gpu_ms = 0.0;
    // Activations are quantized on the CPU before the GPU reads them, so a
    // matmul whose input another matmul of the same batch produces (a LoRA
    // chain) waits for that batch. Matmuls sharing an input (q/k/v, gate/up)
    // share one quantized copy.
    std::vector<Op> ops;
    std::vector<const ggml_tensor*> produced; // outputs of `ops`
    struct Quantized {
        const void* data;
        ggml_type type;
        std::size_t off;
        std::uint32_t row;
        int ncols;
    };
    std::vector<Quantized> done;
    std::size_t used = 0;
    // Fork: dispatches in one submission may now feed each other (and the
    // graph allocator reuses memory between them), so each op carries a UAV
    // barrier when its reads or writes overlap an earlier op's writes, or its
    // writes overlap an earlier op's reads, since the last barrier.
    struct Span {
        const std::uint8_t* a;
        const std::uint8_t* b;
    };
    std::vector<Span> pend_r, pend_w;
    const auto span_of = [](const ggml_tensor* t) {
        const auto* a = static_cast<const std::uint8_t*>(t->data);
        return Span{a, a + ggml_nbytes(t)};
    };
    const auto xq_span = [&](std::size_t off, std::size_t bytes) {
        return Span{g.xq_ptr + off, g.xq_ptr + off + bytes};
    };
    const auto hits = [](const std::vector<Span>& v, const Span& x) {
        for (const Span& s : v)
            if (x.a < s.b && s.a < x.b)
                return true;
        return false;
    };
    const auto place = [&](Op op, std::initializer_list<Span> reads,
                           std::initializer_list<Span> writes) {
        bool need = false;
        for (const Span& w : writes)
            need = need || hits(pend_r, w) || hits(pend_w, w);
        for (const Span& r : reads)
            need = need || hits(pend_w, r);
        if (need) {
            op.barrier = true;
            pend_r.clear();
            pend_w.clear();
        }
        pend_r.insert(pend_r.end(), reads.begin(), reads.end());
        pend_w.insert(pend_w.end(), writes.begin(), writes.end());
        ops.push_back(op);
    };
    auto flush = [&]() -> bool {
        if (ops.empty())
            return true;
        const bool ok = submit_ops(g, ops);
        ops.clear();
        produced.clear();
        done.clear();
        pend_r.clear();
        pend_w.clear();
        used = 0;
        return ok;
    };
    for (const ggml_tensor* node : mm) {
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            // Fork: reads its sources in place; one produced earlier in this
            // submission must be finished first.
            GGML_ASSERT(fa_buffers_ok(node));
            if (ggml_nelements(node) == 0)
                continue;
            const Span none{nullptr, nullptr};
            place({node, 0, 0},
                  {span_of(node->src[0]), span_of(node->src[1]), span_of(node->src[2]),
                   node->src[3] ? span_of(node->src[3]) : none,
                   node->src[4] ? span_of(node->src[4]) : none},
                  {span_of(node)});
            produced.push_back(base_of(node));
            continue;
        }
        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_GET_ROWS) {
            // Fork: a small op, reading and writing D3D12_Host in place.
            GGML_ASSERT(ew_buffers_ok(node) && d3d12_ew_op(node) != D3d12EwOp::None);
            if (ggml_nelements(node) == 0)
                continue;
            const Span none{nullptr, nullptr};
            place({node, 0, 0},
                  {span_of(node->src[0]), node->src[1] ? span_of(node->src[1]) : none},
                  {span_of(node)});
            produced.push_back(base_of(node));
            continue;
        }
        const ggml_tensor* w = node->src[0];
        const ggml_tensor* x = node->src[1];
        GGML_ASSERT(w->buffer && w->buffer->buft == &kWeightsBuft);
        GGML_ASSERT(x->buffer && ggml_backend_buffer_is_host(x->buffer));
        GGML_ASSERT(node->buffer && is_ours(node->buffer->buft));
        if (ggml_nelements(node) == 0)
            continue; // a ubatch without output rows: nothing to compute
        if (node->op == GGML_OP_GET_ROWS) {
            // Row ids are tiny: stage them next to the quantized activations.
            const std::size_t bytes = static_cast<std::size_t>(x->ne[0]) * sizeof(std::int32_t);
            if (used + bytes > g.xq_bytes && !flush())
                return GGML_STATUS_FAILED;
            if (!ensure_xq(g, used + bytes))
                return GGML_STATUS_FAILED;
            std::memcpy(g.xq_ptr + used, x->data, bytes);
            // The kernel does not bound-check rows; ggml-cpu asserts the same.
            const auto* ids = reinterpret_cast<const std::int32_t*>(g.xq_ptr + used);
            for (std::int64_t i = 0; i < x->ne[0]; ++i)
                GGML_ASSERT(ids[i] >= 0 && ids[i] < w->ne[1]);
            place({node, used, 0}, {xq_span(used, bytes)}, {span_of(node)});
            produced.push_back(base_of(node));
            used += (bytes + 255) & ~std::size_t(255);
            continue;
        }
        const ggml_type at = d3d12_activation_type(w->type);
        const ggml_tensor* xb = base_of(x);
        bool x_on_gpu = false; // produced earlier in this submission
        for (const ggml_tensor* p : produced)
            x_on_gpu = x_on_gpu || p == xb;
        // Fork: a q8_K input produced on the GPU is quantized there too;
        // anything else waits for the submission, as before.
        const bool gpu_quant = x_on_gpu && at == GGML_TYPE_Q8_K && g.pso_q8k &&
                               g_ew_state.load() == 1 && (g_ew_mask.load() & kEwQuantBit) &&
                               x->nb[0] == sizeof(float) && x->ne[0] % 256 == 0;
        if (x_on_gpu && !gpu_quant && !flush())
            return GGML_STATUS_FAILED;
        const int k = static_cast<int>(x->ne[0]);
        const int ncols = static_cast<int>(x->ne[1]);
        const std::uint32_t row = static_cast<std::uint32_t>(ggml_row_size(at, k));
        const Quantized* q = nullptr;
        for (const Quantized& d : done)
            if (d.data == x->data && d.type == at && d.row == row && d.ncols >= ncols)
                q = &d;
        if (!q && gpu_quant) {
            const std::size_t bytes = static_cast<std::size_t>(row) * ncols;
            if (used + bytes > g.xq_bytes) {
                // No room left: finish this submission; the input is then
                // ready for the CPU quantizer below.
                if (!flush())
                    return GGML_STATUS_FAILED;
            } else {
                Op qop{node, used, row};
                qop.quant_x = x;
                place(qop, {span_of(x)}, {xq_span(used, bytes)});
                done.push_back({x->data, at, used, row, ncols});
                q = &done.back();
                used += (bytes + 255) & ~std::size_t(255);
            }
        }
        if (!q) {
            const std::size_t bytes = static_cast<std::size_t>(row) * ncols;
            if (used + bytes > g.xq_bytes && !flush())
                return GGML_STATUS_FAILED;
            if (!ensure_xq(g, used + bytes))
                return GGML_STATUS_FAILED;
            const auto tq = std::chrono::steady_clock::now();
            d3d12_quantize_activations(w->type, static_cast<const float*>(x->data),
                                       x->nb[1] / sizeof(float), k, ncols, g.xq_ptr + used);
            g.phase.quant_ms +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tq)
                    .count();
            done.push_back({x->data, at, used, row, ncols});
            q = &done.back();
            used += (bytes + 255) & ~std::size_t(255); // root UAV addresses stay aligned
        }
        place({node, q->off, q->row}, {xq_span(q->off, static_cast<std::size_t>(q->row) * ncols)},
              {span_of(node)});
        produced.push_back(base_of(node));
    }
    if (!flush())
        return GGML_STATUS_FAILED;
    const double call_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++g.n_calls;
    g.gpu_ms += g.last_gpu_ms;
    g.wall_ms += call_ms;
    ++g.phase.calls;
    g.phase.wall_ms += call_ms;
    return GGML_STATUS_SUCCESS;
}

} // namespace

bool ggml_d3d12_take_stats(D3d12PhaseStats* out) {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    if (out)
        *out = g.ok ? g.phase : D3d12PhaseStats{};
    g.phase = D3d12PhaseStats{};
    return g.ok;
}

namespace {

ggml_guid_t backend_guid() {
    static ggml_guid guid = {0x78, 0x6c, 0x6c, 0x61, 0x6d, 0x61, 0x2d, 0x64,
                             0x33, 0x64, 0x31, 0x32, 0x2d, 0x76, 0x30, 0x31};
    return &guid;
}

const ggml_backend_i kBackendIface = {
    /* .get_name            = */ backend_name,
    /* .free                = */ backend_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ nullptr, // graph_compute completes before returning
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ backend_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

// --- Device ---

const char* dev_name(ggml_backend_dev_t) {
    return "D3D12";
}
const char* dev_description(ggml_backend_dev_t) {
    return "xllama D3D12 compute (Q4_0/Q4_K/Q6_K matmul, Q6_K get_rows)";
}
void dev_memory(ggml_backend_dev_t, size_t* free, size_t* total) {
    *free = *total = 0;
    Gpu& g = gpu();
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter3> adapter;
    if (!g.ok || FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumAdapterByLuid(g.luid, IID_PPV_ARGS(&adapter))))
        return;
    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        *total = static_cast<size_t>(info.Budget);
        *free = info.Budget > info.CurrentUsage
                    ? static_cast<size_t>(info.Budget - info.CurrentUsage)
                    : 0;
    }
}
enum ggml_backend_dev_type dev_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}
void dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props* props) {
    props->name = dev_name(dev);
    props->description = dev_description(dev);
    props->type = dev_type(dev);
    props->device_id = nullptr;
    dev_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {/* async */ false, /* host_buffer */ false, /* buffer_from_host_ptr */ false,
                   /* events */ false, /* mmap_support */ false};
}
ggml_backend_t dev_init_backend(ggml_backend_dev_t dev, const char*) {
    if (!gpu().ok)
        return nullptr;
    return new ggml_backend{backend_guid(), kBackendIface, dev, nullptr};
}
ggml_backend_buffer_type_t dev_buffer_type(ggml_backend_dev_t) {
    return &kHostBuft;
}

bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor* op) {
    switch (op->op) {
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
        return true;
    case GGML_OP_MUL_MAT: {
        const ggml_tensor* w = op->src[0];
        const ggml_tensor* x = op->src[1];
        D3d12MatmulDesc d;
        d.src0_type = w->type;
        d.ne00 = w->ne[0];
        d.ne01 = w->ne[1];
        d.ne02 = w->ne[2];
        d.ne03 = w->ne[3];
        d.ne10 = x->ne[0];
        d.ne11 = x->ne[1];
        d.ne12 = x->ne[2];
        d.ne13 = x->ne[3];
        d.src1_type = x->type;
        d.dst_type = op->type;
        d.src0_contiguous = ggml_is_contiguous(w);
        d.src1_contiguous = ggml_is_contiguous(x);
        d.src0_in_weight_buffer = w->buffer && w->buffer->buft == &kWeightsBuft;
        const bool ok = d3d12_mm_supported(d);
        // A refused matmul whose weight already sits in D3D12_Weights means a
        // placement the backend then cannot run — log the first few (D2b #309).
        static int logged = 0;
        if (!ok && w->buffer && w->buffer->buft == &kWeightsBuft && logged < 8) {
            ++logged;
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "[xllama] d3d12: MUL_MAT refused: %s %lldx%lld x [%lld,%lld,%lld,%lld] "
                          "src1=%s dst=%s\n",
                          ggml_type_name(w->type), static_cast<long long>(w->ne[0]),
                          static_cast<long long>(w->ne[1]), static_cast<long long>(x->ne[0]),
                          static_cast<long long>(x->ne[1]), static_cast<long long>(x->ne[2]),
                          static_cast<long long>(x->ne[3]), ggml_type_name(x->type),
                          ggml_type_name(op->type));
            log_output(msg);
        }
        return ok;
    }
    case GGML_OP_GET_ROWS: {
        const ggml_tensor* w = op->src[0];
        const ggml_tensor* ids = op->src[1];
        D3d12GetRowsDesc d;
        d.src0_type = w->type;
        d.ne00 = w->ne[0];
        d.ne01 = w->ne[1];
        d.ne02 = w->ne[2];
        d.ne03 = w->ne[3];
        d.src1_type = ids->type;
        d.ne10 = ids->ne[0];
        d.ne11 = ids->ne[1];
        d.ne12 = ids->ne[2];
        d.dst_type = op->type;
        d.src0_contiguous = ggml_is_contiguous(w);
        d.src1_contiguous = ggml_is_contiguous(ids);
        d.src0_in_weight_buffer = w->buffer && w->buffer->buft == &kWeightsBuft;
        return d3d12_get_rows_supported(d);
    }
    case GGML_OP_FLASH_ATTN_EXT:
        return fa_buffers_ok(op) && d3d12_fa_supported(fa_desc_of(op)) && fa_enabled();
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_RMS_NORM:
    case GGML_OP_UNARY:
    case GGML_OP_GLU:
        if (!ew_buffers_ok(op))
            return false;
        if (const D3d12EwOp eo = d3d12_ew_op(op); eo != D3d12EwOp::None)
            return ew_enabled() && (g_ew_mask.load() & (1u << static_cast<unsigned>(eo))) != 0;
        return false;
    default:
        return false;
    }
}
bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return is_ours(buft);
}

const ggml_backend_device_i kDeviceIface = {
    /* .get_name             = */ dev_name,
    /* .get_description      = */ dev_description,
    /* .get_memory           = */ dev_memory,
    /* .get_type             = */ dev_type,
    /* .get_props            = */ dev_props,
    /* .init_backend         = */ dev_init_backend,
    /* .get_buffer_type      = */ dev_buffer_type,
    // The default buft doubles as the host buft: llama.cpp then allocates the
    // CPU backend's compute buffer in D3D12_Host, which supports_buft accepts,
    // so the scheduler stops copying every matmul input into a second buffer.
    /* .get_host_buffer_type = */ dev_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ dev_supports_op,
    /* .supports_buft        = */ dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// --- Registry ---

const char* reg_name(ggml_backend_reg_t) {
    return "D3D12";
}
size_t reg_device_count(ggml_backend_reg_t) {
    return 1;
}
ggml_backend_dev_t reg_device(ggml_backend_reg_t, size_t index) {
    GGML_ASSERT(index == 0);
    return device_ptr();
}

// Weights live in the extra buft; llama.cpp lists it after the default one.
ggml_backend_buffer_type_t* dev_extra_bufts(ggml_backend_dev_t) {
    static ggml_backend_buffer_type_t list[] = {&kWeightsBuft, nullptr};
    return list;
}

void* reg_proc_address(ggml_backend_reg_t, const char* name) {
    if (std::strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0)
        return reinterpret_cast<void*>(dev_extra_bufts);
    return nullptr;
}

const ggml_backend_reg_i kRegIface = {
    /* .get_name         = */ reg_name,
    /* .get_device_count = */ reg_device_count,
    /* .get_device       = */ reg_device,
    /* .get_proc_address = */ reg_proc_address,
};

ggml_backend_reg kReg = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ kRegIface,
    /* .context     = */ nullptr,
};

ggml_backend_device kDevice = {
    /* .iface   = */ kDeviceIface,
    /* .reg     = */ &kReg,
    /* .context = */ nullptr,
};

ggml_backend_dev_t device_ptr() {
    return &kDevice;
}

} // namespace

bool ggml_d3d12_register() {
    static std::once_flag once;
    static bool registered = false;
    std::call_once(once, [] {
        Gpu& g = gpu();
        if (!g.ok) {
            log_output(("[xllama] d3d12: backend unavailable: " + g.error + "\n").c_str());
            return;
        }
        kHostBuft.device = &kDevice;
        kWeightsBuft.device = &kDevice;
        ggml_backend_register(&kReg);
        registered = true;
    });
    return registered;
}

ggml_backend_buffer_type_t ggml_d3d12_weights_buft() {
    return ggml_d3d12_register() ? &kWeightsBuft : nullptr;
}

// --- Selftest ---

namespace {

constexpr int kSelftestTimedRuns = 5;

struct SelftestCase {
    ggml_type type;
    const char* name;
    int n, k, ncols;
};

// Max |gpu - cpu| / max |cpu| against the CPU backend's own path: the
// vec_dot type's from_float, then ggml's vec_dot (#312, gate A).
double reference_rel_err(ggml_type t, const std::vector<std::uint8_t>& q, std::size_t row_bytes,
                         const std::vector<float>& x, const std::vector<float>& y, int n, int k,
                         int ncols) {
    ggml_cpu_init();
    const auto* wt = ggml_get_type_traits_cpu(t);
    std::vector<std::uint8_t> xq(ggml_row_size(wt->vec_dot_type, k));
    double max_ref = 0.0, max_diff = 0.0;
    for (int c = 0; c < ncols; ++c) {
        ggml_get_type_traits_cpu(wt->vec_dot_type)
            ->from_float(x.data() + static_cast<std::size_t>(c) * k, xq.data(), k);
        for (int r = 0; r < n; ++r) {
            float ref = 0.f;
            wt->vec_dot(k, &ref, 0, q.data() + static_cast<std::size_t>(r) * row_bytes, 0,
                        xq.data(), 0, 1);
            const double got = y[static_cast<std::size_t>(c) * n + r];
            // std::max ignores a NaN operand, which would hide corrupt output.
            if (!std::isfinite(ref) || !std::isfinite(got))
                return std::numeric_limits<double>::infinity();
            max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref)));
            max_diff = std::max(max_diff, std::fabs(static_cast<double>(ref) - got));
        }
    }
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

D3d12SelftestRow run_case(ggml_backend_t backend, const SelftestCase& sc) {
    D3d12SelftestRow row;
    row.type = sc.name;
    row.n = sc.n;
    row.k = sc.k;
    row.ncols = sc.ncols;

    ggml_init_params ip = {};
    ip.mem_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx_w = ggml_init(ip);
    ggml_context* ctx_x = ggml_init(ip);
    ggml_context* ctx_g = ggml_init(ip);
    ggml_tensor* w = ggml_new_tensor_2d(ctx_w, sc.type, sc.k, sc.n);
    ggml_tensor* x = ggml_new_tensor_2d(ctx_x, GGML_TYPE_F32, sc.k, sc.ncols);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, &kWeightsBuft);
    ggml_backend_buffer_t xbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_x, &kHostBuft);
    ggml_gallocr_t galloc = nullptr;

    auto cleanup = [&] {
        if (galloc)
            ggml_gallocr_free(galloc);
        if (wbuf)
            ggml_backend_buffer_free(wbuf);
        if (xbuf)
            ggml_backend_buffer_free(xbuf);
        ggml_free(ctx_g);
        ggml_free(ctx_x);
        ggml_free(ctx_w);
    };
    if (!wbuf || !xbuf) {
        row.error = "buffer allocation failed";
        cleanup();
        return row;
    }

    std::mt19937 rng(1234u + static_cast<unsigned>(sc.n + sc.k + sc.ncols));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> wf(static_cast<std::size_t>(sc.n) * sc.k);
    for (float& v : wf)
        v = uni(rng);
    std::vector<std::uint8_t> q(ggml_nbytes(w));
    ggml_quantize_chunk(sc.type, wf.data(), q.data(), 0, sc.n, sc.k, nullptr);
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
    std::vector<float> xf(static_cast<std::size_t>(sc.k) * sc.ncols);
    for (float& v : xf)
        v = uni(rng);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));

    ggml_tensor* y = ggml_mul_mat(ctx_g, w, x);
    ggml_cgraph* gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, y);
    galloc = ggml_gallocr_new(&kHostBuft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        row.error = "graph allocation failed";
        cleanup();
        return row;
    }
    // Warm-up, then the median of kSelftestTimedRuns: one sample of a ~9 MB
    // matmul swung from 78 to 109 GB/s across back-to-back selftests (CI
    // 1.6.0.1154, same blob).
    ggml_status st = ggml_backend_graph_compute(backend, gf);
    std::vector<double> ms;
    for (int i = 0; i < kSelftestTimedRuns && st == GGML_STATUS_SUCCESS; ++i) {
        st = ggml_backend_graph_compute(backend, gf);
        ms.push_back(gpu().last_gpu_ms);
    }
    row.d3d12_ran = st == GGML_STATUS_SUCCESS;
    if (!row.d3d12_ran) {
        row.error = "graph_compute failed";
        cleanup();
        return row;
    }
    std::nth_element(ms.begin(), ms.begin() + ms.size() / 2, ms.end());
    row.gpu_ms = ms[ms.size() / 2];
    row.packed_gbs = row.gpu_ms > 0.0 ? static_cast<double>(q.size()) / 1e6 / row.gpu_ms : 0.0;
    std::vector<float> yf(static_cast<std::size_t>(sc.n) * sc.ncols);
    ggml_backend_tensor_get(y, yf.data(), 0, yf.size() * sizeof(float));
    row.rel_err = reference_rel_err(sc.type, q, w->nb[1], xf, yf, sc.n, sc.k, sc.ncols);
    row.ok = row.rel_err <= kD3d12SelftestRelTol;
    if (!row.ok)
        row.error = "mismatch vs the CPU backend (q8 vec_dot)";
    cleanup();
    return row;
}

// GET_ROWS from a Q6_K weight (the tied token_embd path, #309): the GPU must
// equal ggml's dequantizer bit for bit, as the host emulation does.
D3d12SelftestRow run_get_rows_case(ggml_backend_t backend, int n, int k) {
    D3d12SelftestRow row;
    row.type = "get_rows_q6_k";
    row.n = n;
    row.k = k;
    const std::int32_t ids_v[] = {0, 1, n / 2 + 1, n - 1, 7, n / 3};
    const int n_ids = static_cast<int>(sizeof(ids_v) / sizeof(ids_v[0]));
    row.ncols = n_ids;

    ggml_init_params ip = {};
    ip.mem_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx_w = ggml_init(ip);
    ggml_context* ctx_i = ggml_init(ip);
    ggml_context* ctx_g = ggml_init(ip);
    ggml_tensor* w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q6_K, k, n);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx_i, GGML_TYPE_I32, n_ids);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, &kWeightsBuft);
    ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_i, &kHostBuft);
    ggml_gallocr_t galloc = nullptr;
    auto cleanup = [&] {
        if (galloc)
            ggml_gallocr_free(galloc);
        if (wbuf)
            ggml_backend_buffer_free(wbuf);
        if (ibuf)
            ggml_backend_buffer_free(ibuf);
        ggml_free(ctx_g);
        ggml_free(ctx_i);
        ggml_free(ctx_w);
    };
    if (!wbuf || !ibuf) {
        row.error = "buffer allocation failed";
        cleanup();
        return row;
    }
    std::mt19937 rng(99u);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> wf(static_cast<std::size_t>(n) * k);
    for (float& v : wf)
        v = uni(rng);
    std::vector<std::uint8_t> q(ggml_nbytes(w));
    ggml_quantize_chunk(GGML_TYPE_Q6_K, wf.data(), q.data(), 0, n, k, nullptr);
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
    ggml_backend_tensor_set(ids, ids_v, 0, sizeof(ids_v));

    ggml_tensor* y = ggml_get_rows(ctx_g, w, ids);
    ggml_cgraph* gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, y);
    galloc = ggml_gallocr_new(&kHostBuft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        row.error = "graph allocation failed";
        cleanup();
        return row;
    }
    row.d3d12_ran = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    if (!row.d3d12_ran) {
        row.error = "graph_compute failed";
        cleanup();
        return row;
    }
    row.gpu_ms = gpu().last_gpu_ms;
    std::vector<float> got(static_cast<std::size_t>(k) * n_ids), ref(static_cast<std::size_t>(k));
    ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));
    double max_ref = 0.0, max_diff = 0.0;
    for (int i = 0; i < n_ids; ++i) {
        ggml_get_type_traits(GGML_TYPE_Q6_K)
            ->to_float(q.data() + static_cast<std::size_t>(ids_v[i]) * w->nb[1], ref.data(), k);
        for (int j = 0; j < k; ++j) {
            const double actual = got[static_cast<std::size_t>(i) * k + j];
            if (!std::isfinite(ref[j]) || !std::isfinite(actual)) {
                row.rel_err = std::numeric_limits<double>::infinity();
                row.error = "non-finite get_rows reference or output";
                cleanup();
                return row;
            }
            max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref[j])));
            max_diff = std::max(max_diff, std::fabs(static_cast<double>(ref[j]) - actual));
        }
    }
    row.rel_err = max_ref > 0.0 ? max_diff / max_ref : max_diff;
    row.ok = max_diff == 0.0; // bit-exact, like the host emulation
    if (!row.ok)
        row.error = "mismatch vs ggml dequantize_row_q6_K";
    cleanup();
    return row;
}

} // namespace

void run_d3d12_selftest(std::vector<D3d12SelftestRow>* out) {
    if (!out)
        return;
    if (!ggml_d3d12_register()) {
        D3d12SelftestRow r;
        r.type = "-";
        r.error = gpu().error.empty() ? "d3d12 backend unavailable" : gpu().error;
        out->push_back(std::move(r));
        return;
    }
    ggml_backend_t backend = dev_init_backend(&kDevice, nullptr);
    // Shapes of the D2 target models; q6_k 2048x11008 has 2-byte-aligned rows.
    const SelftestCase cases[] = {
        {GGML_TYPE_Q4_0, "q4_0", 8192, 2048, 1},
        {GGML_TYPE_Q4_0, "q4_0", 2048, 8192, 1},
        {GGML_TYPE_Q4_K, "q4_k", 11008, 2048, 1},
        {GGML_TYPE_Q4_K, "q4_k", 2048, 11008, 1},
        {GGML_TYPE_Q6_K, "q6_k", 2048, 11008, 1},
        {GGML_TYPE_Q6_K, "q6_k", 65536, 1024, 1},
        {GGML_TYPE_Q4_0, "q4_0", 1024, 1024, 7},
        {GGML_TYPE_Q4_K, "q4_k", 1024, 1024, 7},
        {GGML_TYPE_Q6_K, "q6_k", 1024, 1024, 7},
        {GGML_TYPE_Q4_0, "q4_0", 1024, 1024, 512},
        {GGML_TYPE_Q4_K, "q4_k", 1024, 1024, 512},
        {GGML_TYPE_Q6_K, "q6_k", 1024, 1024, 512},
        // Fork: Qwen3.5-9B shapes (hidden 4096, ffn 12288).
        {GGML_TYPE_IQ3_XXS, "iq3_xxs", 12288, 4096, 1},
        {GGML_TYPE_IQ3_XXS, "iq3_xxs", 4096, 12288, 1},
        {GGML_TYPE_IQ4_XS, "iq4_xs", 12288, 4096, 1},
        {GGML_TYPE_IQ4_XS, "iq4_xs", 4096, 12288, 1},
        {GGML_TYPE_IQ3_XXS, "iq3_xxs", 1024, 1024, 7},
        {GGML_TYPE_IQ4_XS, "iq4_xs", 1024, 1024, 7},
        {GGML_TYPE_IQ3_XXS, "iq3_xxs", 1024, 1024, 512},
        {GGML_TYPE_IQ4_XS, "iq4_xs", 1024, 1024, 512},
        // Fork: Q5_K (Qwen3.5-4B Q4_K_M leftovers: 4096x2560, 2560x8192).
        {GGML_TYPE_Q5_K, "q5_k", 2560, 4096, 1},
        {GGML_TYPE_Q5_K, "q5_k", 2560, 8192, 1},
        {GGML_TYPE_Q5_K, "q5_k", 1024, 1024, 7},
        {GGML_TYPE_Q5_K, "q5_k", 1024, 1024, 512},
    };
    for (const auto& sc : cases)
        out->push_back(run_case(backend, sc));
    out->push_back(run_get_rows_case(backend, 151936, 2048)); // Coder-3B tied token_embd
    out->push_back(run_get_rows_case(backend, 65536, 1024));  // LFM2.5-350M
    ggml_backend_free(backend);
}

} // namespace xllama

    #endif // _WIN32

#endif // XLLAMA_USE_LLAMA
