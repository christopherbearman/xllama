// Fork: tiled prefill IQ4_XS (see ggml_d3d12_tile.hlsli, ggml_d3d12_mmv_iq4_xs.hlsl).
#include "ggml_d3d12_tile.hlsli"
static const int kValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
uint lut4(uint q, uint shift) {
    uint outw = 0u;
    [unroll]
    for (uint j = 0; j < 4u; ++j)
        outw |= ((uint)kValues[(q >> (8u * j + shift)) & 15u] & 0xffu) << (8u * j);
    return outw;
}
[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
    const uint col0 = gid.y * TILE;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint ib = itid >> 1;
    const uint h = itid & 1u;
    const uint e_lo = 32u * ib + 8u * h;
    const uint e_hi = e_lo + 16u;
    float acc[NUM_ROWS][TILE];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r) {
        [unroll]
        for (uint c = 0; c < TILE; ++c)
            acc[r][c] = 0.0;
    }
    for (uint blk = ix; blk < nchunk; blk += IN_FLIGHT) {
        uint a0[NUM_ROWS], a1[NUM_ROWS], b0[NUM_ROWS], b1[NUM_ROWS];
        float wd[NUM_ROWS];
        int wls[NUM_ROWS];
        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + blk * 136u;
            const uint hdr = W.Load(bb);
            const uint sh = hdr >> 16;
            const uint sl = (W.Load(bb + 4u) >> (8u * (ib >> 1))) & 0xffu;
            wls[r] = (int)(((sl >> (4u * (ib & 1u))) & 15u) | (((sh >> (2u * ib)) & 3u) << 4)) - 32;
            wd[r] = f16tof32(hdr & 0xffffu);
            const uint2 q = W.Load2(bb + 8u + 16u * ib + 8u * h);
            a0[r] = lut4(q.x, 0u);
            a1[r] = lut4(q.y, 0u);
            b0[r] = lut4(q.x, 4u);
            b1[r] = lut4(q.y, 4u);
        }
        [unroll]
        for (uint c = 0; c < TILE; ++c) {
            const uint col = min(col0 + c, ncols - 1u);
            const uint xb = col * x_stride + blk * 292u;
            const float dy = asfloat(X.Load(xb));
            const uint2 xl = X.Load2(xb + 4u + e_lo);
            const uint2 xh = X.Load2(xb + 4u + e_hi);
            [unroll]
            for (uint r = 0; r < NUM_ROWS; ++r) {
                const int sumi = dot4(a0[r], xl.x) + dot4(a1[r], xl.y) + dot4(b0[r], xh.x) + dot4(b1[r], xh.y);
                acc[r][c] += (wd[r] * dy) * (float)(wls[r] * sumi);
            }
        }
    }
    reduce_store_tile(acc, tid, row0, col0);
}
