#pragma once
// Jacobi-preconditioned conjugate gradients for a batch of right-hand sides sharing one CSR
// matrix (one independent CG recurrence per right-hand side). Vectors are stored
// interleaved (row-major, N x BATCH) so a single SpMM pass serves the whole batch;
// SpMM, vector updates and dot products are fused into three kernels per iteration.

#include "cuda_runtime.h"
#include "linalg.cuh"
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------------

template<int BATCH_SIZE>
__device__ __forceinline__ void warpReduceSum(float val[BATCH_SIZE]) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        #pragma unroll
        for(int b = 0; b < BATCH_SIZE; ++b) {
            val[b] += __shfl_down_sync(0xffffffff, val[b], offset);
        }
    }
}

// Block-wide sum of each batch entry, accumulated into global_dst with atomics.
template<int BATCH_SIZE, int BLOCK_SIZE>
__device__ __forceinline__ void blockReduceSum(float val[BATCH_SIZE], float* global_dst) {
    static __shared__ float shared[32 * BATCH_SIZE];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    warpReduceSum<BATCH_SIZE>(val);

    if (lane == 0) {
        #pragma unroll
        for(int b=0; b < BATCH_SIZE; ++b) {
            shared[wid * BATCH_SIZE + b] = val[b];
        }
    }
    __syncthreads();

    if (wid == 0) {
        float sum[BATCH_SIZE] = {0.0f};
        if (lane < (BLOCK_SIZE / 32)) {
            #pragma unroll
            for(int b=0; b < BATCH_SIZE; ++b) {
                sum[b] = shared[lane * BATCH_SIZE + b];
            }
        }
        warpReduceSum<BATCH_SIZE>(sum);

        if (lane == 0) {
            #pragma unroll
            for(int b=0; b < BATCH_SIZE; ++b) {
                atomicAdd(&global_dst[b], sum[b]);
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// Layout conversion: column blocks (N per right-hand side) <-> interleaved
// ---------------------------------------------------------------------------------

__global__ void transpose_soa_to_aos(int N, int BATCH, const float* input_soa, float* output_aos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for(int b=0; b<BATCH; ++b) output_aos[idx * BATCH + b] = input_soa[idx + b * N];
    }
}

__global__ void transpose_aos_to_soa(int N, int BATCH, const float* input_aos, float* output_soa) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for(int b=0; b<BATCH; ++b) output_soa[idx + b * N] = input_aos[idx * BATCH + b];
    }
}

// ---------------------------------------------------------------------------------
// Fused CG kernels
// ---------------------------------------------------------------------------------

// Q = A P, sigma = P^T Q
template<int BATCH_SIZE, int BLOCK_SIZE>
__global__ void fused_spmm_dot_kernel_aos(
    int N,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colInd,
    const float* __restrict__ val,
    const float* __restrict__ P,
    float* __restrict__ Q,
    float* __restrict__ global_sigma
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float dot_local[BATCH_SIZE] = {0.0f};

    if (row < N) {
        float p_self[BATCH_SIZE];
        #pragma unroll
        for(int b=0; b<BATCH_SIZE; ++b) p_self[b] = P[row * BATCH_SIZE + b];

        int start = rowPtr[row];
        int end = rowPtr[row + 1];

        for (int i = start; i < end; ++i) {
            int col = colInd[i];
            float A_val = val[i];
            int col_base = col * BATCH_SIZE;
            #pragma unroll
            for(int b=0; b<BATCH_SIZE; ++b) {
                dot_local[b] += A_val * P[col_base + b];
            }
        }

        #pragma unroll
        for(int b=0; b<BATCH_SIZE; ++b) {
            Q[row * BATCH_SIZE + b] = dot_local[b];
            dot_local[b] *= p_self[b];
        }
    }
    blockReduceSum<BATCH_SIZE, BLOCK_SIZE>(dot_local, global_sigma);
}

// alpha = gamma_old / sigma; X += alpha P; R -= alpha Q; Z = D^{-1} R; gamma_new = R^T Z
template<int BATCH_SIZE, int BLOCK_SIZE>
__global__ void fused_update_xr_jacobi_dot_kernel_aos(
    int N,
    const float* __restrict__ inv_diag,
    const float* __restrict__ P,
    const float* __restrict__ Q,
    float* __restrict__ X,
    float* __restrict__ R,
    float* __restrict__ Z,
    const float* __restrict__ global_gamma_old,
    const float* __restrict__ global_sigma,
    float* __restrict__ global_gamma_new
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_alpha[BATCH_SIZE];
    if (threadIdx.x < BATCH_SIZE) {
        float g = global_gamma_old[threadIdx.x];
        float s = global_sigma[threadIdx.x];
        s_alpha[threadIdx.x] = (fabsf(s) > 1e-20f) ? (g / s) : 0.0f;
    }
    __syncthreads();

    float dot_local[BATCH_SIZE] = {0.0f};

    if (idx < N) {
        float d = inv_diag[idx];
        int base_idx = idx * BATCH_SIZE;

        #pragma unroll
        for(int b=0; b<BATCH_SIZE; ++b) {
            float alpha = s_alpha[b];
            float x_val = X[base_idx + b] + alpha * P[base_idx + b];
            float r_val = R[base_idx + b] - alpha * Q[base_idx + b];
            float z_val = r_val * d;

            X[base_idx + b] = x_val;
            R[base_idx + b] = r_val;
            Z[base_idx + b] = z_val;

            dot_local[b] = r_val * z_val;
        }
    }
    blockReduceSum<BATCH_SIZE, BLOCK_SIZE>(dot_local, global_gamma_new);
}

// beta = gamma_new / gamma_old; P = Z + beta P
template<int BATCH_SIZE, int BLOCK_SIZE>
__global__ void fused_update_p_kernel_aos(
    int N,
    const float* __restrict__ Z,
    float* __restrict__ P,
    const float* __restrict__ global_gamma_old,
    const float* __restrict__ global_gamma_new
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_beta[BATCH_SIZE];
    if (threadIdx.x < BATCH_SIZE) {
        float g_old = global_gamma_old[threadIdx.x];
        float g_new = global_gamma_new[threadIdx.x];
        s_beta[threadIdx.x] = (fabsf(g_old) > 1e-20f) ? (g_new / g_old) : 0.0f;
    }
    __syncthreads();

    if (idx < N) {
        int base_idx = idx * BATCH_SIZE;
        #pragma unroll
        for(int b=0; b<BATCH_SIZE; ++b) {
            P[base_idx + b] = Z[base_idx + b] + s_beta[b] * P[base_idx + b];
        }
    }
}

// R = B - A X; Z = P = D^{-1} R; gamma = R^T Z
template<int BATCH_SIZE, int BLOCK_SIZE>
__global__ void fused_init_kernel_aos(
    int N,
    const int* __restrict__ rowPtr, const int* __restrict__ colInd, const float* __restrict__ val,
    const float* __restrict__ B,
    const float* __restrict__ inv_diag,
    float* __restrict__ X, float* __restrict__ R, float* __restrict__ Z, float* __restrict__ P,
    float* __restrict__ global_gamma
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float dot_local[BATCH_SIZE] = {0.0f};

    if (row < N) {
        float Ax[BATCH_SIZE] = {0.0f};
        int start = rowPtr[row];
        int end = rowPtr[row + 1];
        for(int i=start; i<end; ++i) {
            int col = colInd[i];
            float A_val = val[i];
            int col_base = col * BATCH_SIZE;
            #pragma unroll
            for(int b=0; b<BATCH_SIZE; ++b) Ax[b] += A_val * X[col_base + b];
        }
        int base_idx = row * BATCH_SIZE;
        #pragma unroll
        for(int b=0; b<BATCH_SIZE; ++b) {
            float r_val = B[base_idx + b] - Ax[b];
            float z_val = r_val * inv_diag[row];
            R[base_idx + b] = r_val;
            Z[base_idx + b] = z_val;
            P[base_idx + b] = z_val;
            dot_local[b] = r_val * z_val;
        }
    }
    blockReduceSum<BATCH_SIZE, BLOCK_SIZE>(dot_local, global_gamma);
}

__global__ void extract_inv_diag_kernel_simple(int N, const int* rowPtr, const int* colInd, const float* val, float* inv_diag) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float diag = 1.0f;
        for (int i = rowPtr[row]; i < rowPtr[row+1]; ++i) {
            if (colInd[i] == row) { diag = val[i]; break; }
        }
        inv_diag[row] = (fabsf(diag) > 1e-15f) ? (1.0f / diag) : 1.0f;
    }
}

// ---------------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------------

class BlockPCGSolver {
private:
    int N;
    int nnz;
    int batchSize;
    const int check_interval = 10;   // convergence is tested every check_interval iterations
    const int BLOCK_SIZE = 256;

    cudaStream_t stream;

    int *d_rowPtr, *d_colInd;
    float *d_valA, *d_inv_diag;
    float *d_R, *d_P, *d_Z, *d_Q, *d_X, *d_B;

    float *d_scalars; // [0..5] gamma (buffer A), [6..11] sigma, [12..17] gamma (buffer B)
    float *d_temp_buffer;

    float *h_gamma_pinned;
    float *h_r0_pinned;

public:
    // Copies the CSR matrix (device pointers) and extracts the Jacobi preconditioner.
    BlockPCGSolver(int n_rows, int n_nonzero, int n_rhs,
                   const int* d_rowPtr_in, const int* d_colInd_in, const float* d_val_in)
        : N(n_rows), nnz(n_nonzero), batchSize(n_rhs)
    {
        if (batchSize != 6) printf("Warning: Designed for Batch=6\n");

        checkCudaErrors(cudaStreamCreate(&stream));

        checkCudaErrors(cudaMalloc((void**)&d_rowPtr, (N + 1) * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&d_colInd, nnz * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&d_valA, nnz * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&d_inv_diag, N * sizeof(float)));

        checkCudaErrors(cudaMemcpyAsync(d_rowPtr, d_rowPtr_in, (N + 1) * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(d_colInd, d_colInd_in, nnz * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(d_valA, d_val_in, nnz * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        extract_inv_diag_kernel_simple<<<grid, BLOCK_SIZE, 0, stream>>>(N, d_rowPtr, d_colInd, d_valA, d_inv_diag);

        size_t vecSize = N * batchSize * sizeof(float);
        checkCudaErrors(cudaMalloc((void**)&d_R, vecSize));
        checkCudaErrors(cudaMalloc((void**)&d_P, vecSize));
        checkCudaErrors(cudaMalloc((void**)&d_Q, vecSize));
        checkCudaErrors(cudaMalloc((void**)&d_Z, vecSize));
        checkCudaErrors(cudaMalloc((void**)&d_X, vecSize));
        checkCudaErrors(cudaMalloc((void**)&d_B, vecSize));

        checkCudaErrors(cudaMalloc((void**)&d_scalars, 3 * batchSize * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&d_temp_buffer, vecSize));

        checkCudaErrors(cudaHostAlloc((void**)&h_gamma_pinned, batchSize * sizeof(float), cudaHostAllocDefault));
        checkCudaErrors(cudaHostAlloc((void**)&h_r0_pinned, batchSize * sizeof(float), cudaHostAllocDefault));

        checkCudaErrors(cudaStreamSynchronize(stream));
    }

    // d_rhs / d_x: device arrays of batchSize consecutive vectors of length N; d_x holds the
    // initial guess on entry. A right-hand side is converged once
    // sqrt(r^T D^{-1} r) < tol * sqrt(r0^T D^{-1} r0); iteration stops when all have converged.
    void solve(const float* d_rhs, float* d_x, int max_iter, float tol) {
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t vecSize = N * batchSize * sizeof(float);

        checkCudaErrors(cudaMemcpyAsync(d_temp_buffer, d_rhs, vecSize, cudaMemcpyDeviceToDevice, stream));
        transpose_soa_to_aos<<<grid, BLOCK_SIZE, 0, stream>>>(N, batchSize, d_temp_buffer, d_B);

        checkCudaErrors(cudaMemcpyAsync(d_temp_buffer, d_x, vecSize, cudaMemcpyDeviceToDevice, stream));
        transpose_soa_to_aos<<<grid, BLOCK_SIZE, 0, stream>>>(N, batchSize, d_temp_buffer, d_X);

        checkCudaErrors(cudaMemsetAsync(d_scalars, 0, 3 * batchSize * sizeof(float), stream));

        // gamma buffers are swapped after every iteration ("curr" holds the latest r^T z)
        float* d_gamma_curr_ptr = &d_scalars[0];
        float* d_sigma_ptr      = &d_scalars[6];
        float* d_gamma_next_ptr = &d_scalars[12];

        fused_init_kernel_aos<6, 256><<<grid, BLOCK_SIZE, 0, stream>>>(
            N, d_rowPtr, d_colInd, d_valA, d_B, d_inv_diag,
            d_X, d_R, d_Z, d_P, d_gamma_curr_ptr
        );

        std::vector<bool> converged(batchSize, false);

        checkCudaErrors(cudaMemcpyAsync(h_r0_pinned, d_gamma_curr_ptr, batchSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));

        for(int i=0; i<batchSize; ++i) {
            h_r0_pinned[i] = sqrtf(h_r0_pinned[i]);
            if(h_r0_pinned[i] < 1e-15f) converged[i] = true;
        }

        int k = 0;

        while (k < max_iter) {
            for (int j = 0; j < check_interval; ++j) {
                checkCudaErrors(cudaMemsetAsync(d_sigma_ptr, 0, batchSize * sizeof(float), stream));
                checkCudaErrors(cudaMemsetAsync(d_gamma_next_ptr, 0, batchSize * sizeof(float), stream));

                fused_spmm_dot_kernel_aos<6, 256><<<grid, BLOCK_SIZE, 0, stream>>>(
                    N, d_rowPtr, d_colInd, d_valA, d_P, d_Q, d_sigma_ptr
                );

                fused_update_xr_jacobi_dot_kernel_aos<6, 256><<<grid, BLOCK_SIZE, 0, stream>>>(
                    N, d_inv_diag, d_P, d_Q, d_X, d_R, d_Z, d_gamma_curr_ptr, d_sigma_ptr, d_gamma_next_ptr
                );

                fused_update_p_kernel_aos<6, 256><<<grid, BLOCK_SIZE, 0, stream>>>(
                    N, d_Z, d_P, d_gamma_curr_ptr, d_gamma_next_ptr
                );

                std::swap(d_gamma_curr_ptr, d_gamma_next_ptr);
            }
            k += check_interval;

            checkCudaErrors(cudaMemcpyAsync(h_gamma_pinned, d_gamma_curr_ptr, batchSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
            checkCudaErrors(cudaStreamSynchronize(stream));

            int active = 0;
            for(int i=0; i<batchSize; ++i) {
                if (converged[i]) continue;
                if (sqrtf(h_gamma_pinned[i]) < tol * h_r0_pinned[i]) {
                    converged[i] = true;
                } else {
                    active++;
                }
            }
            if (active == 0) break;
        }

        transpose_aos_to_soa<<<grid, BLOCK_SIZE, 0, stream>>>(N, batchSize, d_X, d_temp_buffer);
        checkCudaErrors(cudaMemcpyAsync(d_x, d_temp_buffer, vecSize, cudaMemcpyDeviceToHost, stream));
        checkCudaErrors(cudaStreamSynchronize(stream));
    }

    ~BlockPCGSolver() {
        if(d_temp_buffer) cudaFree(d_temp_buffer);
        cudaFree(d_rowPtr); cudaFree(d_colInd); cudaFree(d_valA); cudaFree(d_inv_diag);
        cudaFree(d_R); cudaFree(d_P); cudaFree(d_Q); cudaFree(d_Z); cudaFree(d_X); cudaFree(d_B);
        cudaFree(d_scalars);
        cudaFreeHost(h_gamma_pinned);
        cudaFreeHost(h_r0_pinned);
        cudaStreamDestroy(stream);
    }
};
