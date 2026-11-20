// Fork: tiled prefill Q4_K (see ggml_d3d12_tile.hlsli, ggml_d3d12_mmv_q4_k.hlsl).
#include "ggml_d3d12_tile.hlsli"
uint scale_byte(uint i, uint4 hdr) {
    uint w = (i < 4u) ? hdr.y : ((i < 8u) ? hdr.z : hdr.w);
    return (w >> ((i & 3u) * 8u)) & 0xffu;
}
void get_scale_min_k4(uint j, uint4 hdr, out uint d, out uint m) {
    if (j < 4u) {
        d = scale_byte(j, hdr) & 63u;
        m = scale_byte(j + 4u, hdr) & 63u;
    } else {
        uint a = scale_byte(j + 4u, hdr);
        uint b = scale_byte(j - 4u, hdr);
        uint cc = scale_byte(j, hdr);
        d = (a & 0xFu) | ((b >> 6) << 4);
        m = (a >> 4) | ((cc >> 6) << 4);
    }
}
[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint row0 = gid.x * NUM_ROWS;
    const uint col0 = gid.y * TILE;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint il = itid >> 2, ir = itid & 3u;
    const uint e_lo = il * 64u + ir * 8u, e_hi = e_lo + 32u;
    float acc[NUM_ROWS][TILE];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r) {
        [unroll]
        for (uint c = 0; c < TILE; ++c)
            acc[r][c] = 0.0;
    }
    for (uint blk = ix; blk < nchunk; blk += IN_FLIGHT) {
        float wd[NUM_ROWS], wm[NUM_ROWS];
        uint lx[NUM_ROWS], ly[NUM_ROWS], hx[NUM_ROWS], hy[NUM_ROWS];
        int s1[NUM_ROWS], m1[NUM_ROWS], s2[NUM_ROWS], m2[NUM_ROWS];
        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + blk * 144u;
            const uint4 hdr = W.Load4(bb);
            const uint2 q = W.Load2(bb + 16u + il * 32u + ir * 8u);
            wd[r] = f16tof32(hdr.x & 0xffffu);
            wm[r] = f16tof32(hdr.x >> 16);
            uint a, b, a2, b2;
            get_scale_min_k4(2u * il, hdr, a, b);
            get_scale_min_k4(2u * il + 1u, hdr, a2, b2);
            s1[r] = (int)a; m1[r] = (int)b; s2[r] = (int)a2; m2[r] = (int)b2;
            lx[r] = q.x & 0x0F0F0F0Fu; ly[r] = q.y & 0x0F0F0F0Fu;
            hx[r] = (q.x >> 4) & 0x0F0F0F0Fu; hy[r] = (q.y >> 4) & 0x0F0F0F0Fu;
        }
        [unroll]
        for (uint c = 0; c < TILE; ++c) {
            const uint col = min(col0 + c, ncols - 1u);
            const uint xb = col * x_stride + blk * 292u;
            const float dy = asfloat(X.Load(xb));
            const uint2 xl = X.Load2(xb + 4u + e_lo);
            const uint2 xh = X.Load2(xb + 4u + e_hi);
            const int sl = sum4(xl.x) + sum4(xl.y);
            const int sh = sum4(xh.x) + sum4(xh.y);
            [unroll]
            for (uint r = 0; r < NUM_ROWS; ++r) {
                const int lo = dot4(lx[r], xl.x) + dot4(ly[r], xl.y);
                const int hi = dot4(hx[r], xh.x) + dot4(hy[r], xh.y);
                acc[r][c] += (wd[r] * dy) * (float)(s1[r] * lo + s2[r] * hi) -
                             (wm[r] * dy) * (float)(m1[r] * sl + m2[r] * sh);
            }
        }
    }
    reduce_store_tile(acc, tid, row0, col0);
}
