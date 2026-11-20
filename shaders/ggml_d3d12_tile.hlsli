// Fork: tiled prefill variants of the ggml_d3d12_mmv_* kernels. One group
// handles NUM_ROWS rows x TILE activation columns: each weight block is
// loaded once and applied to TILE columns, instead of once per column. The
// per-thread arithmetic per column is the mmv kernels' exactly (same lanes,
// same order), so results equal the decode kernels' column for column.
// Root constant pad0 carries ncols; SV_GroupID.y is the column tile.
#ifndef TILE
#define TILE 8
#endif
#include "ggml_d3d12_common.hlsli"

#define ncols pad0

groupshared float red_t[NUM_ROWS][NUM_THREADS];

void reduce_store_tile(float acc[NUM_ROWS][TILE], uint tid, uint row0, uint col0) {
    [unroll]
    for (uint c = 0; c < TILE; ++c) {
        [unroll]
        for (uint r = 0; r < NUM_ROWS; ++r)
            red_t[r][tid] = acc[r][c];
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint stride = NUM_THREADS / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                [unroll]
                for (uint r = 0; r < NUM_ROWS; ++r)
                    red_t[r][tid] += red_t[r][tid + stride];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (tid < NUM_ROWS && row0 + tid < n && col0 + c < ncols)
            Y[(col0 + c) * y_stride + row0 + tid] = red_t[tid][0];
        GroupMemoryBarrierWithGroupSync();
    }
}
