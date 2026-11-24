// ggml backend d3d12 (fork): FLASH_ATTN_EXT — the "look-back" of the
// full-attention layers, which ggml-cpu ran on the CPU (61% of a 6k-token
// prefill, 52% of each decoded token at that length, op profile Nov 2026).
//
// One group per (query row, head): gid.x = query, gid.y = head. The group
// walks the keys in blocks of FA_BLOCK (= threads): each thread scores one key
// (q . k * scale, softcap, + mask), the group takes the block max and sum, and
// rescales its running accumulator — the same online softmax as
// ggml_compute_forward_flash_attn_ext_f16_one_chunk, one block at a time.
// Thread t owns output dims t, t + 64, t + 128, t + 192.
//
// K/V are F16 or Q8_0 (block = { half d; int8 qs[32]; } = 34 B) rows; Q is
// f32; mask is f16; sinks f32 per head. All buffers are CPU-visible D3D12_Host
// memory in UNORDERED_ACCESS, read through root UAVs (like X in the matmuls).
// Host emulation, block for block: d3d12_fa_emulate (src/bridge/ggml_d3d12.cpp).

#define FA_THREADS 64
#define FA_BLOCK 64
#define FA_MAX_D 256
#define FA_MAX_DPT (FA_MAX_D / FA_THREADS)

cbuffer FaParams : register(b0) {
    uint n_kv;         // keys
    uint d;            // head size (DK == DV), multiple of 64, <= 256
    uint n_q;          // query rows
    uint n_head;       // query heads
    uint gqa;          // query heads per K/V head
    uint nbq1;         // Q bytes between query rows
    uint nbq2;         // Q bytes between heads
    uint nbk1;         // K bytes between keys
    uint nbk2;         // K bytes between K/V heads
    uint nbv1;         // V bytes between keys
    uint nbv2;         // V bytes between K/V heads
    uint nbm1;         // mask bytes between query rows
    uint nbm2;         // mask bytes between mask planes
    uint mask_ne2;     // mask planes (heads % mask_ne2), >= 1
    uint scale_bits;   // asfloat: KQ scale (already / softcap when softcap != 0)
    uint softcap_bits; // asfloat: logit softcap, 0 = none
    uint has_mask;
    uint has_sinks;
    uint kv_q8;        // 1 = Q8_0 K/V, 0 = F16
    uint q_off;        // byte offset of each view in its root UAV: the backend
    uint k_off;        // rounds the address down to 4 bytes and passes the rest
    uint v_off;
    uint m_off;
    uint pad0;
};

RWByteAddressBuffer O : register(u0);
RWByteAddressBuffer Qb : register(u1);
RWByteAddressBuffer Kb : register(u2);
RWByteAddressBuffer Vb : register(u3);
RWByteAddressBuffer Mb : register(u4);
RWByteAddressBuffer Sb : register(u5);

groupshared float qs[FA_MAX_D];
groupshared float ps[FA_BLOCK];
groupshared float red[FA_THREADS];

// Q8_0 blocks are 34 B, so rows sit on 2-byte boundaries; loads need 4-byte
// alignment. Every address here is even (the backend checks the strides).
uint ldk32(uint a) {
    const uint b = a & ~3u;
    const uint lo = Kb.Load(b);
    if ((a & 3u) == 0u)
        return lo;
    return (lo >> 16) | (Kb.Load(b + 4u) << 16);
}

uint ldk16(uint a) {
    const uint v = Kb.Load(a & ~3u);
    return (a & 2u) != 0u ? (v >> 16) : (v & 0xffffu);
}

uint ldv16(uint a) {
    const uint v = Vb.Load(a & ~3u);
    return (a & 2u) != 0u ? (v >> 16) : (v & 0xffffu);
}

uint ldv8(uint a) {
    const uint v = Vb.Load(a & ~3u);
    return (v >> ((a & 3u) * 8u)) & 0xffu;
}

uint ldm16(uint a) {
    const uint v = Mb.Load(a & ~3u);
    return (a & 2u) != 0u ? (v >> 16) : (v & 0xffffu);
}

// Byte i (0..3) of w as a signed int8.
int s8(uint w, uint i) {
    return ((int)(w << (24u - 8u * i))) >> 24;
}

float neg_inf() {
    return asfloat(0xff800000u);
}

// q . k for the key row at byte `base` of Kb.
float k_dot(uint base) {
    float acc = 0.0;
    if (kv_q8 != 0u) {
        const uint nb = d >> 5;
        for (uint b = 0u; b < nb; ++b) {
            const uint a = base + b * 34u;
            const float sc = f16tof32(ldk16(a));
            float part = 0.0;
            [unroll]
            for (uint w = 0u; w < 8u; ++w) {
                const uint v = ldk32(a + 2u + 4u * w);
                const uint e = b * 32u + w * 4u;
                part += qs[e] * (float)s8(v, 0u) + qs[e + 1u] * (float)s8(v, 1u) +
                        qs[e + 2u] * (float)s8(v, 2u) + qs[e + 3u] * (float)s8(v, 3u);
            }
            acc += sc * part;
        }
    } else {
        for (uint e = 0u; e < d; e += 2u) {
            const uint v = ldk32(base + 2u * e);
            acc += qs[e] * f16tof32(v & 0xffffu) + qs[e + 1u] * f16tof32(v >> 16);
        }
    }
    return acc;
}

// Element e of the V row at byte `base` of Vb.
float v_at(uint base, uint e) {
    if (kv_q8 != 0u) {
        const uint a = base + (e >> 5) * 34u;
        const int q = ((int)(ldv8(a + 2u + (e & 31u)) << 24)) >> 24;
        return f16tof32(ldv16(a)) * (float)q;
    }
    return f16tof32(ldv16(base + 2u * e));
}

void reduce_max(uint tid) {
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint st = FA_THREADS / 2u; st > 0u; st >>= 1u) {
        if (tid < st)
            red[tid] = max(red[tid], red[tid + st]);
        GroupMemoryBarrierWithGroupSync();
    }
}

void reduce_sum(uint tid) {
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint st = FA_THREADS / 2u; st > 0u; st >>= 1u) {
        if (tid < st)
            red[tid] = red[tid] + red[tid + st];
        GroupMemoryBarrierWithGroupSync();
    }
}

[numthreads(FA_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint iq = gid.x;
    const uint h = gid.y;
    const uint hk = h / gqa;
    const float scale = asfloat(scale_bits);
    const float softcap = asfloat(softcap_bits);
    const float NEG = neg_inf();
    const uint dpt = d / FA_THREADS;

    for (uint e = tid; e < d; e += FA_THREADS)
        qs[e] = asfloat(Qb.Load(q_off + iq * nbq1 + h * nbq2 + e * 4u));

    float acc[FA_MAX_DPT];
    [unroll]
    for (uint j = 0u; j < FA_MAX_DPT; ++j)
        acc[j] = 0.0;
    float M = NEG;
    float S = 0.0;
    const uint mrow = m_off + iq * nbm1 + (h % mask_ne2) * nbm2;
    const uint kbase = k_off + hk * nbk2;
    const uint vbase = v_off + hk * nbv2;

    GroupMemoryBarrierWithGroupSync(); // qs ready

    for (uint kb = 0u; kb < n_kv; kb += FA_BLOCK) {
        const uint ic = kb + tid;
        float s = NEG;
        if (ic < n_kv) {
            float mv = 0.0;
            if (has_mask != 0u)
                mv = f16tof32(ldm16(mrow + 2u * ic));
            if (mv != NEG) {
                float x = k_dot(kbase + ic * nbk1) * scale;
                if (softcap != 0.0)
                    x = softcap * tanh(x);
                s = x + mv;
            }
        }
        red[tid] = s;
        reduce_max(tid);
        const float bm = red[0];
        GroupMemoryBarrierWithGroupSync(); // everyone has red[0] before it is reused
        const float Mn = max(M, bm);
        if (Mn == NEG)
            continue; // the whole block is masked (uniform across the group)
        const float p = (s == NEG) ? 0.0 : exp(s - Mn);
        ps[tid] = p;
        red[tid] = p;
        reduce_sum(tid);
        const float bs = red[0];
        const float ms = (M == NEG) ? 0.0 : exp(M - Mn);
        S = S * ms + bs;
        M = Mn;
        [unroll]
        for (uint j2 = 0u; j2 < FA_MAX_DPT; ++j2)
            acc[j2] *= ms;
        const uint nk = min((uint)FA_BLOCK, n_kv - kb);
        for (uint t = 0u; t < nk; ++t) {
            const float pt = ps[t];
            if (pt == 0.0)
                continue;
            const uint vrow = vbase + (kb + t) * nbv1;
            [unroll]
            for (uint j3 = 0u; j3 < FA_MAX_DPT; ++j3) {
                if (j3 < dpt)
                    acc[j3] += pt * v_at(vrow, tid + FA_THREADS * j3);
            }
        }
        GroupMemoryBarrierWithGroupSync(); // ps / red free for the next block
    }

    if (has_sinks != 0u) {
        const float sk = asfloat(Sb.Load(h * 4u));
        if (sk > M) {
            const float ms = (M == NEG) ? 0.0 : exp(M - sk);
            [unroll]
            for (uint j4 = 0u; j4 < FA_MAX_DPT; ++j4)
                acc[j4] *= ms;
            S = S * ms + 1.0;
            M = sk;
        } else {
            S = S + exp(sk - M);
        }
    }

    const float inv = S == 0.0 ? 0.0 : 1.0 / S;
    const uint row = iq * n_head + h; // dst = permute(0, 2, 1, 3): [d, n_head, n_q]
    [unroll]
    for (uint j5 = 0u; j5 < FA_MAX_DPT; ++j5) {
        if (j5 < dpt)
            O.Store((row * d + tid + FA_THREADS * j5) * 4u, asuint(acc[j5] * inv));
    }
}
