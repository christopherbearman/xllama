// Fork: tiled prefill Q6_K (see ggml_d3d12_tile.hlsli, ggml_d3d12_mmv_q6_k.hlsl).
#include "ggml_d3d12_tile.hlsli"
uint q6pack(uint ql, uint qh, uint ql_shift, uint qh_shift) {
    return ((ql >> ql_shift) & 0x0F0F0F0Fu) | (((qh >> qh_shift) & 0x03030303u) << 4);
}
int sbyte(uint word, uint i) {
    return asint(((word >> (8u * i)) & 0xffu) << 24) >> 24;
}
[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
    const uint col0 = gid.y * TILE;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint v = itid >> 3;
    const uint l0 = 4u * (itid & 7u);
    const uint is = l0 >> 4;
    const uint e = 128u * v + l0;
    float acc[NUM_ROWS][TILE];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r) {
        [unroll]
        for (uint c = 0; c < TILE; ++c)
            acc[r][c] = 0.0;
    }
    for (uint blk = ix; blk < nchunk; blk += IN_FLIGHT) {
        uint p1[NUM_ROWS], p2[NUM_ROWS], p3[NUM_ROWS], p4[NUM_ROWS];
        int c1[NUM_ROWS], c2[NUM_ROWS], c3[NUM_ROWS], c4[NUM_ROWS];
        float wd[NUM_ROWS];
        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + blk * 210u;
            const uint ql_a = ld32(bb + 64u * v + l0);
            const uint ql_b = ld32(bb + 64u * v + 32u + l0);
            const uint qh = ld32(bb + 128u + 32u * v + l0);
            const uint sc0 = ld32(bb + 192u + 8u * v);
            const uint sc1 = ld32(bb + 192u + 8u * v + 4u);
            wd[r] = f16tof32(ld16(bb + 208u));
            p1[r] = q6pack(ql_a, qh, 0u, 0u);
            p2[r] = q6pack(ql_b, qh, 0u, 2u);
            p3[r] = q6pack(ql_a, qh, 4u, 4u);
            p4[r] = q6pack(ql_b, qh, 4u, 6u);
            c1[r] = sbyte(sc0, is);
            c2[r] = sbyte(sc0, is + 2u);
            c3[r] = sbyte(sc1, is);
            c4[r] = sbyte(sc1, is + 2u);
        }
        [unroll]
        for (uint c = 0; c < TILE; ++c) {
            const uint col = min(col0 + c, ncols - 1u);
            const uint xb = col * x_stride + blk * 292u;
            const float dy = asfloat(X.Load(xb));
            const uint x1 = X.Load(xb + 4u + e);
            const uint x2 = X.Load(xb + 4u + e + 32u);
            const uint x3 = X.Load(xb + 4u + e + 64u);
            const uint x4 = X.Load(xb + 4u + e + 96u);
            const int s1 = sum4(x1), s2 = sum4(x2), s3 = sum4(x3), s4 = sum4(x4);
            [unroll]
            for (uint r = 0; r < NUM_ROWS; ++r) {
                const int t = c1[r] * (dot4(p1[r], x1) - 32 * s1) + c2[r] * (dot4(p2[r], x2) - 32 * s2) +
                              c3[r] * (dot4(p3[r], x3) - 32 * s3) + c4[r] * (dot4(p4[r], x4) - 32 * s4);
                acc[r][c] += (wd[r] * dy) * (float)t;
            }
        }
    }
    reduce_store_tile(acc, tid, row0, col0);
}
