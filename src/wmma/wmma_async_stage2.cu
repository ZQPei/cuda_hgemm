// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 00:53:54 on Mon, Feb 13, 2023
//
// Description: wmma async stage2 hgemm

#include "common.h"

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define WARP_ROWS 64
#define WARP_COLS 64

#define BLOCK_ROW_WARPS 2  // BLOCK_COLS / WARP_COLS
#define BLOCK_COL_WARPS 4  // BLOCK_ROWS / WARP_ROWS

#define BLOCK_ROW_TILES 8   // BLOCK_COLS / WMMA_N
#define BLOCK_COL_TILES 16  // BLOCK_ROWS / WMMA_M

#define WARP_ROW_TILES 4  // WARP_COLS / WMMA_N
#define WARP_COL_TILES 4  // WARP_ROWS / WMMA_M

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8      // BLOCK_ROW_WARPS * BLOCK_COL_WARPS
#define THREADS_PER_BLOCK 256  // WARP_SIZE * WARPS_PER_BLOCK

#define CHUNK_K 2  // 32 / WMMA_K

#define THREAD_COPY_BYTES 16

#define CHUNK_LINE_BYTES 64          // CHUNK_K * WMMA_K * sizeof(half)
#define CHUNK_COPY_LINES_PER_WARP 8  // WARP_SIZE * THREAD_COPY_BYTES / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES 4      // WARP_SIZE / CHUNK_COPY_LINES_PER_WARP

#define SHMEM_PADDING 8

#define AB_SHMEM_STRIDE 40  // CHUNK_K * WMMA_K + SHMEM_PADDING

#define C_SHMEM_STRIDE 136  // BLOCK_COLS + SHMEM_PADDING
#define C_SHMEM_OFFSET 64   // WARP_COLS

#define BLOCK_STRIDE 16

#define K_STAGE 2

using namespace nvcuda;

__global__ void wmmaAsyncStage2Kernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C,
                                      size_t M, size_t N, size_t K) {
    const size_t M_tiles = div_ceil(M, WMMA_M);
    const size_t N_tiles = div_ceil(N, WMMA_N);
    const size_t K_tiles = div_ceil(K, WMMA_K);

    const size_t block_tile_i =
        (blockIdx.z % 2) ? (blockIdx.y * BLOCK_COL_TILES) : ((gridDim.y - blockIdx.y - 1) * BLOCK_COL_TILES);
    const size_t block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * BLOCK_ROW_TILES;

    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) {
        return;
    }

    extern __shared__ half shmem[][AB_SHMEM_STRIDE];

    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    const size_t B_shmem_idx_off = BLOCK_ROWS;
    const size_t shmem_stage_off = BLOCK_ROWS + BLOCK_COLS;

    half *shmem_warp_tile_ptr = &shmem[0][0] + (warp_id / BLOCK_ROW_WARPS) * C_SHMEM_STRIDE * WARP_ROWS +
                                (warp_id % BLOCK_ROW_WARPS) * C_SHMEM_OFFSET;

    half *shmem_warp_stream_ptr = &shmem[0][0] + warp_id * WMMA_M * C_SHMEM_STRIDE;

    const size_t gmem_idx = (block_tile_i + warp_id) * WMMA_M * N + block_tile_j * WMMA_N;
    half *src_gmem_warp_stream_ptr = &C[gmem_idx];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag[WARP_COL_TILES][WARP_ROW_TILES];

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            wmma::fill_fragment(C_frag[i][j], 0.0f);
        }
    }

    const half *A_warp_ptr = &A[block_tile_i * WMMA_M * K] + BLOCK_ROWS / WARPS_PER_BLOCK * K * warp_id;
    const half *B_warp_ptr = &B[block_tile_j * WMMA_N * K] + BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id;

    const size_t A_shmem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    const size_t B_shmem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

    size_t shmem_store_idx = 0;
    size_t shmem_load_idx = 0;

    size_t shmem_store_off = 0;
    size_t shmem_load_off = 0;

    size_t A_shmem_idx = 0;
    int4 *A_lane_ptr = nullptr;

    size_t B_shmem_idx = 0;
    int4 *B_lane_ptr = nullptr;

    A_shmem_idx = shmem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    A_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < A_shmem_iters; ++i) {
        uint32_t A_shmem_lane_addr =
            __cvta_generic_to_shared(&shmem[A_shmem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;

        CP_ASYNC_CG(A_shmem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

        A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    B_shmem_idx = shmem_store_off + B_shmem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    B_lane_ptr = (int4 *)(B_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    B_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < B_shmem_iters; ++i) {
        uint32_t B_shmem_lane_addr =
            __cvta_generic_to_shared(&shmem[B_shmem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;

        CP_ASYNC_CG(B_shmem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

        B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        B_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);

    __syncthreads();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag[2][WARP_COL_TILES];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag[2][WARP_ROW_TILES];

    size_t reg_store_idx = 0;
    size_t reg_load_idx = 1;

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        size_t A_shmem_idx = shmem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * WMMA_M;
        const half *A_tile_ptr = &shmem[A_shmem_idx][0];

        wmma::load_matrix_sync(A_frag[reg_store_idx][i], A_tile_ptr, AB_SHMEM_STRIDE);
    }

#pragma unroll
    for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
        size_t B_shmem_idx = shmem_load_off + B_shmem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * WMMA_N;
        const half *B_tile_ptr = &shmem[B_shmem_idx][0];

        wmma::load_matrix_sync(B_frag[reg_store_idx][j], B_tile_ptr, AB_SHMEM_STRIDE);
    }

#pragma unroll
    for (size_t tile_k = CHUNK_K * (K_STAGE - 1); tile_k < K_tiles; tile_k += CHUNK_K) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_shmem_idx = shmem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * WMMA_M;
            const half *A_tile_ptr = &shmem[A_shmem_idx][WMMA_K];

            wmma::load_matrix_sync(A_frag[reg_store_idx][i], A_tile_ptr, AB_SHMEM_STRIDE);
        }

#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t B_shmem_idx =
                shmem_load_off + B_shmem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * WMMA_N;
            const half *B_tile_ptr = &shmem[B_shmem_idx][WMMA_K];

            wmma::load_matrix_sync(B_frag[reg_store_idx][j], B_tile_ptr, AB_SHMEM_STRIDE);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                wmma::mma_sync(C_frag[i][j_s], A_frag[reg_load_idx][i], B_frag[reg_load_idx][j_s], C_frag[i][j_s]);
            }
        }

        shmem_store_idx = (shmem_store_idx + 1) % K_STAGE;
        shmem_store_off = shmem_store_idx * shmem_stage_off;

        A_shmem_idx = shmem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        A_lane_ptr = (int4 *)(A_warp_ptr + tile_k * WMMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        A_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < A_shmem_iters / CHUNK_K; ++i) {
            uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(&shmem[A_shmem_idx][0]) +
                                         (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;

            CP_ASYNC_CG(A_shmem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        B_shmem_idx = shmem_store_off + B_shmem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
        B_lane_ptr = (int4 *)(B_warp_ptr + tile_k * WMMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        B_shmem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < B_shmem_iters / CHUNK_K; ++i) {
            uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(&shmem[B_shmem_idx][0]) +
                                         (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;

            CP_ASYNC_CG(B_shmem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        shmem_load_idx = (shmem_load_idx + 1) % K_STAGE;
        shmem_load_off = shmem_load_idx * shmem_stage_off;

#pragma unroll
        for (size_t i = (CHUNK_K - 1) * A_shmem_iters / CHUNK_K; i < A_shmem_iters; ++i) {
            uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(&shmem[A_shmem_idx][0]) +
                                         (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;

            CP_ASYNC_CG(A_shmem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

#pragma unroll
        for (size_t i = (CHUNK_K - 1) * B_shmem_iters / CHUNK_K; i < B_shmem_iters; ++i) {
            uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(&shmem[B_shmem_idx][0]) +
                                         (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;

            CP_ASYNC_CG(B_shmem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_shmem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();

        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_shmem_idx = shmem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * WMMA_M;
            const half *A_tile_ptr = &shmem[A_shmem_idx][0];

            wmma::load_matrix_sync(A_frag[reg_store_idx][i], A_tile_ptr, AB_SHMEM_STRIDE);
        }

#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t B_shmem_idx =
                shmem_load_off + B_shmem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * WMMA_N;
            const half *B_tile_ptr = &shmem[B_shmem_idx][0];

            wmma::load_matrix_sync(B_frag[reg_store_idx][j], B_tile_ptr, AB_SHMEM_STRIDE);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                wmma::mma_sync(C_frag[i][j_s], A_frag[reg_load_idx][i], B_frag[reg_load_idx][j_s], C_frag[i][j_s]);
            }
        }
    }

#pragma unroll
    for (size_t k_step = 1; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_shmem_idx = shmem_store_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * WMMA_M;
            const half *A_tile_ptr = &shmem[A_shmem_idx][k_step * WMMA_K];

            wmma::load_matrix_sync(A_frag[reg_store_idx][i], A_tile_ptr, AB_SHMEM_STRIDE);
        }

#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t B_shmem_idx =
                shmem_store_off + B_shmem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * WMMA_N;
            const half *B_tile_ptr = &shmem[B_shmem_idx][k_step * WMMA_K];

            wmma::load_matrix_sync(B_frag[reg_store_idx][j], B_tile_ptr, AB_SHMEM_STRIDE);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                wmma::mma_sync(C_frag[i][j_s], A_frag[reg_load_idx][i], B_frag[reg_load_idx][j_s], C_frag[i][j_s]);
            }
        }
    }

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

            wmma::mma_sync(C_frag[i][j_s], A_frag[reg_store_idx][i], B_frag[reg_store_idx][j_s], C_frag[i][j_s]);
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            half *C_tile_ptr = shmem_warp_tile_ptr + i * C_SHMEM_STRIDE * WMMA_M + j * WMMA_N;

            wmma::store_matrix_sync(C_tile_ptr, C_frag[i][j], C_SHMEM_STRIDE, wmma::mem_row_major);
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < WMMA_M; ++i) {
        *((int4 *)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16) =
            *((int4 *)(shmem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SHMEM_STRIDE) + lane_id % 16);
    }

    __syncthreads();
}

size_t initWmmaAsyncStage2() {
    int dev_id = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t shmem_max_size = std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SHMEM_STRIDE * sizeof(half) * K_STAGE,
                                     BLOCK_ROWS * C_SHMEM_STRIDE * sizeof(half));
    HLOG("shmem_max_size: %.0f KBytes (%zu Bytes)", static_cast<float>(shmem_max_size / 1024.0f), shmem_max_size);

    HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, shmem_max_size);
    HGEMM_CHECK_CUDART_ERROR(
        cudaFuncSetAttribute(wmmaAsyncStage2Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_max_size));

    return shmem_max_size;
}

void wmmaAsyncStage2(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    static size_t shmem_max_size = initWmmaAsyncStage2();

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_STRIDE * BLOCK_COLS));

    wmmaAsyncStage2Kernel<<<grid, block, shmem_max_size>>>(A, B, C, M, N, K);
}
