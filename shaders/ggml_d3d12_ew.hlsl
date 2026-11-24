// ggml backend d3d12 (fork): the small ops between the matmuls — ADD, MUL,
// RMS_NORM, SILU, SWIGLU — and the activation quantizer (f32 -> q8_K), so a
// layer's matmul chain stays on the GPU instead of handing over to the CPU for
// each one (53% GPU busy at decode, Nov 2026). One entry point per op
// (-E CSAdd, ...), all on the flash-attention root signature: 24 root
// constants at b0, u0 = output, u1 = src0, u2 = src1.
//
// f32 only. Rows are contiguous (nb0 = 4); src0 and dst rows may be strided;
// src1 broadcasts by modulo over each dimension (ggml_can_repeat). Host
// emulations: d3d12_ew_emulate / d3d12_quant_q8k_emulate (ggml_d3d12.cpp).

#define EW_THREADS 64

cbuffer EwParams : register(b0) {
    uint ne0;      // dst row length (elements)
    uint ne1;
    uint ne2;
    uint ne3;
    uint s0_nb1;   // src0 bytes between rows / planes / volumes
    uint s0_nb2;
    uint s0_nb3;
    uint s1_ne0;   // src1 shape (broadcast by modulo)
    uint s1_ne1;
    uint s1_ne2;
    uint s1_ne3;
    uint s1_nb1;
    uint s1_nb2;
    uint s1_nb3;
    uint d_nb1;    // dst bytes between rows / planes / volumes
    uint d_nb2;
    uint d_nb3;
    uint s0_off;   // byte offset of each view in its root UAV
    uint s1_off;
    uint d_off;
    uint p0;       // RMS_NORM: asfloat eps. SWIGLU: 1 = two sources.
    uint p1;       // SWIGLU single source: element offset of the gate half
    uint p2;       // SWIGLU single source: element offset of the up half
    uint p3;
};

RWByteAddressBuffer D : register(u0);
RWByteAddressBuffer S0 : register(u1);
RWByteAddressBuffer S1 : register(u2);

groupshared float red[EW_THREADS];

uint row0(uint3 g) { return s0_off + g.x * s0_nb1 + g.y * s0_nb2 + g.z * s0_nb3; }
uint rowd(uint3 g) { return d_off + g.x * d_nb1 + g.y * d_nb2 + g.z * d_nb3; }
uint row1(uint3 g) {
    return s1_off + (g.x % s1_ne1) * s1_nb1 + (g.y % s1_ne2) * s1_nb2 + (g.z % s1_ne3) * s1_nb3;
}

float silu(float x) {
    return x / (1.0 + exp(-x));
}

[numthreads(EW_THREADS, 1, 1)]
void CSAdd(uint tid : SV_GroupIndex, uint3 g : SV_GroupID) {
    const uint a = row0(g), b = row1(g), o = rowd(g);
    for (uint i = tid; i < ne0; i += EW_THREADS)
        D.Store(o + i * 4u, asuint(asfloat(S0.Load(a + i * 4u)) +
                                    asfloat(S1.Load(b + (i % s1_ne0) * 4u))));
}

[numthreads(EW_THREADS, 1, 1)]
void CSMul(uint tid : SV_GroupIndex, uint3 g : SV_GroupID) {
    const uint a = row0(g), b = row1(g), o = rowd(g);
    for (uint i = tid; i < ne0; i += EW_THREADS)
        D.Store(o + i * 4u, asuint(asfloat(S0.Load(a + i * 4u)) *
                                    asfloat(S1.Load(b + (i % s1_ne0) * 4u))));
}

[numthreads(EW_THREADS, 1, 1)]
void CSSilu(uint tid : SV_GroupIndex, uint3 g : SV_GroupID) {
    const uint a = row0(g), o = rowd(g);
    for (uint i = tid; i < ne0; i += EW_THREADS)
        D.Store(o + i * 4u, asuint(silu(asfloat(S0.Load(a + i * 4u)))));
}

// silu(gate) * up. Two sources: gate = src0, up = src1 (same shape). One
// source: both halves of each src0 row, at element offsets p1 (gate), p2 (up).
[numthreads(EW_THREADS, 1, 1)]
void CSSwiglu(uint tid : SV_GroupIndex, uint3 g : SV_GroupID) {
    const uint a = row0(g), o = rowd(g);
    const uint b = p0 != 0u ? row1(g) : a + p2 * 4u;
    const uint ga = p0 != 0u ? a : a + p1 * 4u;
    for (uint i = tid; i < ne0; i += EW_THREADS)
        D.Store(o + i * 4u, asuint(silu(asfloat(S0.Load(ga + i * 4u))) *
                                    asfloat((p0 != 0u ? S1.Load(b + i * 4u) : S0.Load(b + i * 4u)))));
}

[numthreads(EW_THREADS, 1, 1)]
void CSRmsNorm(uint tid : SV_GroupIndex, uint3 g : SV_GroupID) {
    const uint a = row0(g), o = rowd(g);
    float sum = 0.0;
    for (uint i = tid; i < ne0; i += EW_THREADS) {
        const float x = asfloat(S0.Load(a + i * 4u));
        sum += x * x;
    }
    red[tid] = sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint st = EW_THREADS / 2u; st > 0u; st >>= 1u) {
        if (tid < st)
            red[tid] += red[tid + st];
        GroupMemoryBarrierWithGroupSync();
    }
    const float scale = 1.0 / sqrt(red[0] / (float)ne0 + asfloat(p0));
    for (uint j = tid; j < ne0; j += EW_THREADS)
        D.Store(o + j * 4u, asuint(asfloat(S0.Load(a + j * 4u)) * scale));
}

// --- f32 -> block_q8_K (quantize_row_q8_K_ref) ---
// block_q8_K = { float d; int8 qs[256]; int16 bsums[16]; } = 292 B.
// Group = one block of one row: gid.x = block, gid.y = row. Thread t owns
// elements 4t..4t+3. Constants reuse EwParams: ne0 = K, s0_nb1 = x row
// stride (bytes), s0_off, d_nb1 = output row stride (bytes), d_off.
groupshared float qa[EW_THREADS]; // |x| max per thread
groupshared uint qi[EW_THREADS];  // its element index (first wins on ties)
groupshared int qsum[EW_THREADS];

[numthreads(EW_THREADS, 1, 1)]
void CSQuantQ8K(uint tid : SV_GroupIndex, uint3 g : SV_GroupID) {
    const uint blk = g.x, row = g.y;
    const uint xa = s0_off + row * s0_nb1 + (blk * 256u + tid * 4u) * 4u;
    float x[4];
    float m = 0.0;
    uint mi = 0xffffffffu;
    [unroll]
    for (uint j = 0u; j < 4u; ++j) {
        x[j] = asfloat(S0.Load(xa + j * 4u));
        const float ax = abs(x[j]);
        if (ax > m) {
            m = ax;
            mi = tid * 4u + j;
        }
    }
    qa[tid] = m;
    qi[tid] = mi;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint st = EW_THREADS / 2u; st > 0u; st >>= 1u) {
        if (tid < st) {
            const float o = qa[tid + st];
            const uint oi = qi[tid + st];
            if (o > qa[tid] || (o == qa[tid] && oi < qi[tid])) {
                qa[tid] = o;
                qi[tid] = oi;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float amax = qa[0];
    const uint ob = d_off + row * d_nb1 + blk * 292u;
    int q[4] = {0, 0, 0, 0};
    float dval = 0.0;
    if (amax != 0.0) {
        const float mx = asfloat(S0.Load(s0_off + row * s0_nb1 + (blk * 256u + qi[0]) * 4u));
        const float iscale = -127.0 / mx;
        [unroll]
        for (uint j2 = 0u; j2 < 4u; ++j2)
            q[j2] = min(127, (int)round(iscale * x[j2])); // round = nearest, ties to even
        dval = 1.0 / iscale;
    }
    D.Store(ob + 4u + tid * 4u, (uint)(q[0] & 0xff) | ((uint)(q[1] & 0xff) << 8) |
                                    ((uint)(q[2] & 0xff) << 16) | ((uint)(q[3] & 0xff) << 24));
    // bsums: 16 sums of 16 elements = 4 threads each.
    qsum[tid] = q[0] + q[1] + q[2] + q[3];
    GroupMemoryBarrierWithGroupSync();
    if (tid < 8u) {
        const uint t0 = tid * 8u; // bsums 2*tid and 2*tid+1
        const int lo = qsum[t0] + qsum[t0 + 1u] + qsum[t0 + 2u] + qsum[t0 + 3u];
        const int hi = qsum[t0 + 4u] + qsum[t0 + 5u] + qsum[t0 + 6u] + qsum[t0 + 7u];
        D.Store(ob + 260u + tid * 4u, ((uint)lo & 0xffffu) | (((uint)hi & 0xffffu) << 16));
    }
    if (tid == 0u)
        D.Store(ob, asuint(dval));
}
