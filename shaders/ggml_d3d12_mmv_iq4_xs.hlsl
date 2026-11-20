// ggml backend d3d12: IQ4_XS x q8_K matmul (fork addition, Big Queef / Qwen3.5-9B).
// block_iq4_xs = 136 B: half d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128].
// 8 sub-blocks of 32 values; byte j of a sub-block holds value j (low nibble)
// and value j + 16 (high nibble), each an index into kvalues_iq4nl. Scale
// ls = (scales_l[ib/2] >> 4*(ib%2) & 15) | (scales_h >> 2*ib & 3) << 4, weight
// (d * d_y) * (ls - 32), as ggml_vec_dot_iq4_xs_q8_K. Per thread (itid 0..15):
// sub-block ib = itid / 2, half h = itid % 2 owns bytes 8h .. 8h + 7.

#include "ggml_d3d12_common.hlsli"

static const int kValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// Four nibbles (selected by shift 0 or 4) of word q mapped through kvalues, packed as int8.
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
    const uint col = gid.y;
    const uint ix = tid >> 4;
    const uint itid = tid & 15u;
    const uint ib = itid >> 1;
    const uint h = itid & 1u;
    const uint e_lo = 32u * ib + 8u * h;
    const uint e_hi = e_lo + 16u;

    float acc[NUM_ROWS];
    [unroll]
    for (uint r = 0; r < NUM_ROWS; ++r)
        acc[r] = 0.0;

    for (uint blk = ix; blk < nchunk; blk += IN_FLIGHT) {
        const uint xb = col * x_stride + blk * 292u;
        const float dy = asfloat(X.Load(xb));
        const uint2 xl = X.Load2(xb + 4u + e_lo);
        const uint2 xh = X.Load2(xb + 4u + e_hi);

        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r) {
            const uint row = min(row0 + r, n - 1u);
            const uint bb = row * w_row_bytes + blk * 136u;
            const uint hdr = W.Load(bb);
            const float d = f16tof32(hdr & 0xffffu);
            const uint sh = hdr >> 16;
            const uint sl = (W.Load(bb + 4u) >> (8u * (ib >> 1))) & 0xffu;
            const int ls = (int)(((sl >> (4u * (ib & 1u))) & 15u) | (((sh >> (2u * ib)) & 3u) << 4));
            const uint2 q = W.Load2(bb + 8u + 16u * ib + 8u * h);
            const int sumi = dot4(lut4(q.x, 0u), xl.x) + dot4(lut4(q.y, 0u), xl.y) +
                             dot4(lut4(q.x, 4u), xh.x) + dot4(lut4(q.y, 4u), xh.y);
            acc[r] += (d * dy) * (float)((ls - 32) * sumi);
        }
    }
    reduce_store(acc, tid, row0, col);
}
