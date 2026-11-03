#pragma once
// Forward homogenization: grid setup, element matrices, global CSR assembly, the six periodic
// cell problems, and the effective elasticity tensor C^H. For the combined-TPMS design space it
// also evaluates an objective J = sum_IJ W_IJ C^H_IJ and its gradient w.r.t. the coefficients.

#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <string>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <cooperative_groups.h>
#include <cuda/std/functional>
#include "linalg.cuh"
#include "sdf.cuh"
#include "grid.cuh"
#include "rm_shell.cuh"
#include "block_pcg.cuh"
#include "logger.cuh"
#include "combined_field.cuh"
#include "rm_gradient.cuh"

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------------
// Global assembly: element 24x24 matrices -> CSR. Rows/columns are grouped in 3x3
// blocks per node pair; every row lists its u columns first, then its w columns.
// ---------------------------------------------------------------------------------

// Key (row node << 32 | col node) of every node pair of every element.
__global__ void extract_node_pairs_kernel(
    const int* compact_indices,
    unsigned long long* pair_keys,
    int nTets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nTets) return;

    const int* nodes = &compact_indices[idx * 4];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned long long r = (unsigned long long)nodes[i] << 32;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            pair_keys[idx * 16 + i * 4 + j] = r | (unsigned int)nodes[j];
        }
    }
}

// Scatters each element matrix (column-major 24x24) into the UU/UW/WU/WW 3x3 blocks of the
// unique node pairs.
template<typename T>
__global__ void accumulate_block_values_kernel(
    const T* __restrict__ elem_data_flat,
    const int* __restrict__ pair_map,
    T* __restrict__ block_values,
    int nTets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nTets) return;

    const T* K_local = &elem_data_flat[idx * 576];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {

            int global_block_idx = pair_map[idx * 16 + i * 4 + j];
            T* dst = &block_values[global_block_idx * 36];

            int r_u = i * 3; int c_u = j * 3;
            int r_w = r_u + 12; int c_w = c_u + 12;

            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    // UU (offset 0)
                    atomicAdd(&dst[0 + r * 3 + c], K_local[(r_u + r) + (c_u + c) * 24]);
                    // UW (offset 9)
                    atomicAdd(&dst[9 + r * 3 + c], K_local[(r_u + r) + (c_w + c) * 24]);
                    // WU (offset 18)
                    atomicAdd(&dst[18 + r * 3 + c], K_local[(r_w + r) + (c_u + c) * 24]);
                    // WW (offset 27)
                    atomicAdd(&dst[27 + r * 3 + c], K_local[(r_w + r) + (c_w + c) * 24]);
                }
            }
        }
    }
}

__global__ void count_node_degree_kernel(
    const unsigned long long* unique_keys,
    int* node_degrees,
    int nUnique)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nUnique) return;

    int r_node = (int)(unique_keys[idx] >> 32);
    atomicAdd(&node_degrees[r_node], 1);
}

// Row lengths (6 entries per neighbour node) of the u rows and the w rows of each node.
__global__ void generate_scalar_row_ptr_kernel(
    const int* node_degrees,
    int* row_ptr_out,
    int nNodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nNodes) return;

    int degree = node_degrees[idx];
    int entries = degree * 6;

    row_ptr_out[3 * idx]     = entries;
    row_ptr_out[3 * idx + 1] = entries;
    row_ptr_out[3 * idx + 2] = entries;

    int offset_w = 3 * nNodes;
    row_ptr_out[offset_w + 3 * idx]     = entries;
    row_ptr_out[offset_w + 3 * idx + 1] = entries;
    row_ptr_out[offset_w + 3 * idx + 2] = entries;
}

template<typename T>
__global__ void expand_block_to_csr_kernel(
    const unsigned long long* __restrict__ unique_keys,
    const T* __restrict__ block_values,
    const int* __restrict__ graph_row_start,
    const int* __restrict__ scalar_row_ptr,
    int* __restrict__ d_col,
    T* __restrict__ d_val,
    int nUnique,
    int nNodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nUnique) return;

    unsigned long long key = unique_keys[idx];
    int r_node = (int)(key >> 32);
    int c_node = (int)(key & 0xFFFFFFFF);

    int neighbor_rank = idx - graph_row_start[r_node];

    const T* src = &block_values[idx * 36];
    int offset_u_rows = 3 * nNodes;
    int offset_u_cols = 3 * nNodes;

    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        // u rows
        int row_u = 3 * r_node + r;
        int start_u = scalar_row_ptr[row_u];
        int end_u   = scalar_row_ptr[row_u + 1];
        int half_len = (end_u - start_u) / 2;

        // UU
        int pos_uu = start_u + neighbor_rank * 3;
        d_col[pos_uu + 0] = 3 * c_node + 0; d_val[pos_uu + 0] = src[0 + r * 3 + 0];
        d_col[pos_uu + 1] = 3 * c_node + 1; d_val[pos_uu + 1] = src[0 + r * 3 + 1];
        d_col[pos_uu + 2] = 3 * c_node + 2; d_val[pos_uu + 2] = src[0 + r * 3 + 2];

        // UW
        int pos_uw = start_u + half_len + neighbor_rank * 3;
        d_col[pos_uw + 0] = offset_u_cols + 3 * c_node + 0; d_val[pos_uw + 0] = src[9 + r * 3 + 0];
        d_col[pos_uw + 1] = offset_u_cols + 3 * c_node + 1; d_val[pos_uw + 1] = src[9 + r * 3 + 1];
        d_col[pos_uw + 2] = offset_u_cols + 3 * c_node + 2; d_val[pos_uw + 2] = src[9 + r * 3 + 2];

        // w rows
        int row_w = 3 * r_node + r + offset_u_rows;
        int start_w = scalar_row_ptr[row_w];

        // WU
        int pos_wu = start_w + neighbor_rank * 3;
        d_col[pos_wu + 0] = 3 * c_node + 0; d_val[pos_wu + 0] = src[18 + r * 3 + 0];
        d_col[pos_wu + 1] = 3 * c_node + 1; d_val[pos_wu + 1] = src[18 + r * 3 + 1];
        d_col[pos_wu + 2] = 3 * c_node + 2; d_val[pos_wu + 2] = src[18 + r * 3 + 2];

        // WW
        int pos_ww = start_w + half_len + neighbor_rank * 3;
        d_col[pos_ww + 0] = offset_u_cols + 3 * c_node + 0; d_val[pos_ww + 0] = src[27 + r * 3 + 0];
        d_col[pos_ww + 1] = offset_u_cols + 3 * c_node + 1; d_val[pos_ww + 1] = src[27 + r * 3 + 1];
        d_col[pos_ww + 2] = offset_u_cols + 3 * c_node + 2; d_val[pos_ww + 2] = src[27 + r * 3 + 2];
    }
}

// ---------------------------------------------------------------------------------
// Fixed node: the u and w DOFs of node fix_1 are set to zero (rows and columns zeroed,
// diagonal set to 20000, right-hand side zeroed). penalty_value is not used.
// ---------------------------------------------------------------------------------

template<typename T>
__global__ void apply_fixed_point_row_kernel(
    const int* row_ptr,
    const int* col_ind,
    T* val,
    int fix_1,
    int n_nodes,
    T penalty_value
)
{
    int idx = threadIdx.x;
    if (idx >= 6) return;

    int target_row = -1;
    int offset = 3 * n_nodes;

    if (idx < 3) {
        // u (x, y, z)
        target_row = 3 * fix_1 + idx;
    }
    else if (idx < 6) {
        // w (x, y, z)
        target_row = 3 * fix_1 + offset + (idx - 3);
    }
    int start = row_ptr[target_row];
    int end = row_ptr[target_row + 1];

    for (int k = start; k < end; ++k) {
        val[k] = 0.0;
        if (col_ind[k] == target_row) {
            val[k] = 20000.0;
        }
    }
}

template <typename T>
__global__ void apply_fixed_point_col_kernel(
    const int *row_ptr,
    const int *col_ind,
    T *val,
    int fix_1,
    int n_nodes,
    int nnz,
    T penalty_value
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    int col_idx_fixed_1 = 3 * fix_1;
    int offset = 3 * n_nodes;
    if ((col_ind[idx] >= col_idx_fixed_1 && col_ind[idx] < col_idx_fixed_1 + 3) || (col_ind[idx] >= col_idx_fixed_1 + offset && col_ind[idx] < col_idx_fixed_1 + offset + 3)) {
        val[idx] = 0.0;
    }
}

template <typename T>
__global__ void apply_fixed_point_rhs_kernel(
    T *val,
    int fix_1,
    int n_nodes,
    T penalty_value
)
{
    int idx = threadIdx.x;
    if (idx >= 6)
        return;
    int offset = 3 * n_nodes;
    int row_fixed = -1;
    if (idx < 3)
        row_fixed = 3 * fix_1 + idx;
    else
        row_fixed = 3 * fix_1 + offset + (idx - 3);
    val[row_fixed] = 0.0;
}

// ---------------------------------------------------------------------------------
// Homogenized tensor
// ---------------------------------------------------------------------------------

// Per element and test strain k: local vector y = g - x, where g is the affine field
// (u = eps_k X at the element's own vertices, w = 0) and x the solution of the linear system.
template<typename T>
__global__ void pack_displacement_per_element_kernel(
    Element* elements,
    CudaVec<T, 3>* nodes,
    int* compact_indices,
    T* disp_1, T* disp_2, T* disp_3, T* disp_4, T* disp_5, T* disp_6,
    CudaMat<T, 3, 3>* test_strain_list,
    CudaMat<T, 24, 6>* out,
    int nActiveTets, int nActiveNodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nActiveTets) return;

    Element tet = elements[idx];
    int* tet_compact = &compact_indices[idx * 4];

    CudaVec<T, 3> X[4];
    X[0] = nodes[tet[0]];
    X[1] = nodes[tet[1]];
    X[2] = nodes[tet[2]];
    X[3] = nodes[tet[3]];
    T* dst_ptr = reinterpret_cast<T*>(&out[idx]);
    const T* disp_ptrs[6] = {disp_1, disp_2, disp_3, disp_4, disp_5, disp_6};

    #pragma unroll
    for (int k = 0; k < 6; ++k)
    {
        CudaMat<T, 3, 3> E_test = test_strain_list[k];
        const T* curr_disp = disp_ptrs[k];

        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            CudaVec<T, 3> u_macro = E_test * X[i];

            int global_dof_start = tet_compact[i] * 3;
            int local_row_base = i * 3;
            int col_offset = k * 24;

            dst_ptr[col_offset + local_row_base + 0] = u_macro[0] - curr_disp[global_dof_start + 0];
            dst_ptr[col_offset + local_row_base + 1] = u_macro[1] - curr_disp[global_dof_start + 1];
            dst_ptr[col_offset + local_row_base + 2] = u_macro[2] - curr_disp[global_dof_start + 2];

            dst_ptr[col_offset + local_row_base + 0 + 12] = 0.0 - curr_disp[global_dof_start + 0 + 3 * nActiveNodes];
            dst_ptr[col_offset + local_row_base + 1 + 12] = 0.0 - curr_disp[global_dof_start + 1 + 3 * nActiveNodes];
            dst_ptr[col_offset + local_row_base + 2 + 12] = 0.0 - curr_disp[global_dof_start + 2 + 3 * nActiveNodes];
        }
    }
}

template<typename T>
struct WarpSmemData
{
    T K[24 * 24];   // element stiffness
    T V[6 * 24];    // the six local vectors y
    T U[6 * 24];    // K y
};

// Element contribution C_e(i, j) = y_i^T K_e y_j. One warp per element.
template<typename T>
__global__ void homogenization_per_element_kernel(
    const CudaMat<T, 24, 24>* __restrict__ global_K,
    const CudaMat<T, 24, 6>* __restrict__ global_V,
    CudaMat<T, 6, 6>* __restrict__ global_C,
    int nElements)
{
    int warp_id_in_block = threadIdx.x / 32;
    int lane_idx = threadIdx.x % 32;
    int ele_idx = (blockIdx.x * (blockDim.x / 32)) + warp_id_in_block;

    if (ele_idx >= nElements) return;

    extern __shared__ char smem_buffer[];
    WarpSmemData<T>* smem_ptr = reinterpret_cast<WarpSmemData<T>*>(smem_buffer);
    WarpSmemData<T>& smem = smem_ptr[warp_id_in_block];

    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());

    const T* src_K = global_K[ele_idx].data();
    const T* src_V = global_V[ele_idx].data();

    for (int i = lane_idx; i < 576; i += 32) {
        smem.K[i] = src_K[i];
    }

    for (int i = lane_idx; i < 144; i += 32) {
        smem.V[i] = src_V[i];
    }

    warp.sync();

    for (int vec_idx = 0; vec_idx < 6; ++vec_idx)
    {
        if (lane_idx < 24)
        {
            int row = lane_idx;
            T val = 0.0;

            #pragma unroll
            for (int col = 0; col < 24; ++col)
            {
                val += smem.K[col * 24 + row] * smem.V[vec_idx * 24 + col];
            }

            smem.U[vec_idx * 24 + row] = val;
        }
    }

    warp.sync();

    for (int task_id = lane_idx; task_id < 36; task_id += 32)
    {
        int i = task_id / 6;
        int j = task_id % 6;

        T dot_val = 0.0;

        #pragma unroll
        for (int k = 0; k < 24; ++k)
        {
            dot_val += smem.V[i * 24 + k] * smem.U[j * 24 + k];
        }

        T* dst_ptr = global_C[ele_idx].data();
        dst_ptr[task_id] = dot_val;
    }
}

// ---------------------------------------------------------------------------------
// Design gradient of the combined TPMS field
// ---------------------------------------------------------------------------------

// grad_c[k] += sum over quadrature points of
//   w_q (dJ/dg . grad f_k + dJ/dHess : Hess f_k),  dJ/d(.) = sum_IJ W_IJ dC^H_IJ/d(.),
// with W = objective_weight. One thread per quadrature slot; block-level reduction in shared memory.
template<typename T>
__global__ void assemble_design_gradient_kernel(
    GaussianQuadSoA<T>* quad_soa,
    CudaMat<T, 6, 6> objective_weight,
    CudaVec<T, 3>* dC_hom_dg,
    CudaMat<T, 3, 3>* dC_hom_dHess,
    T* grad_c,
    int nActiveTets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_quads = nActiveTets * 4;

    __shared__ T smem_grad[6];

    if (threadIdx.x < 6) {
        smem_grad[threadIdx.x] = 0.0;
    }
    __syncthreads();

    if (idx < total_quads)
    {
        int ele_idx = idx / 4;
        int quad_idx = idx % 4;

        if (quad_idx < quad_soa[ele_idx].count)
        {
            float X = quad_soa[ele_idx].world_x[quad_idx];
            float Y = quad_soa[ele_idx].world_y[quad_idx];
            float Z = quad_soa[ele_idx].world_z[quad_idx];
            T weight = quad_soa[ele_idx].weight[quad_idx];

            CudaVec<T, 3> dT_dg(0.0, 0.0, 0.0);
            CudaMat<T, 3, 3> dT_dHess; dT_dHess.setZero();

            // contract dJ/dC^H with dC^H/dg and dC^H/dHess
            size_t per_quad_offset = (size_t)idx * 36;
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 6; ++j) {
                    T w_target = objective_weight(i, j);
                    if (abs(w_target) < 1e-10) continue;

                    int linear_k = i * 6 + j;

                    CudaVec<T, 3> dC_homo_dg_ij = dC_hom_dg[per_quad_offset + linear_k];

                    dT_dg += dC_homo_dg_ij * w_target;

                    CudaMat<T, 3, 3> dC_hom_dHess_ij = dC_hom_dHess[per_quad_offset + linear_k];
                    dT_dHess += w_target * dC_hom_dHess_ij;
                }
            }

            tpms_basis::TrigPrecalc t(X, Y, Z);

            auto accumulate_grad_smem = [&](int k, CudaVec<T, 3> df, CudaMat<T, 3, 3> d2f) {
                T term1 = dT_dg.dot(df);
                T term2 = (dT_dHess.cwiseProduct(d2f)).sum();
                atomicAdd(&smem_grad[k], (term1 + term2) * weight);
            };

            accumulate_grad_smem(0, tpms_basis::f1_grad(t), tpms_basis::f1_hessian(t));
            accumulate_grad_smem(1, tpms_basis::f2_grad(t), tpms_basis::f2_hessian(t));
            accumulate_grad_smem(2, tpms_basis::f3_grad(t), tpms_basis::f3_hessian(t));
            accumulate_grad_smem(3, tpms_basis::f4_grad(t), tpms_basis::f4_hessian(t));
            accumulate_grad_smem(4, tpms_basis::f5_grad(t), tpms_basis::f5_hessian(t));
            accumulate_grad_smem(5, tpms_basis::f6_grad(t), tpms_basis::f6_hessian(t));
        }
    }

    __syncthreads();

    if (threadIdx.x < 6) {
        atomicAdd(&grad_c[threadIdx.x], smem_grad[threadIdx.x]);
    }
}

// ---------------------------------------------------------------------------------
// Simulator
// ---------------------------------------------------------------------------------

template<typename T>
class simulator
{
public:
    // nx, ny, nz: background-grid resolution (cells per axis); E, nu: Young's modulus and
    // Poisson ratio; thickness: shell thickness; log_file: JSON file the results are appended to.
    simulator(const int _nx, const int _ny, const int _nz, const T _E, const T _nu, const T _thickness, const std::string& log_file)
    : nx(_nx), ny(_ny), nz(_nz), E(_E), nu(_nu), thickness(_thickness), background_grid(_nx, _ny, _nz),
    lambda(_E * _nu / (1 - _nu * _nu)), mu(_E / (2 * (1 + _nu))), dx(1.0 / (T)_nx), log(log_file)
    {
        objective_weight.setZero();
    }
    ~simulator(){}

    // Homogenizes the field stored in a 257^3 float64 .npy file.
    void load_data_and_run(const std::string& filename, bool write_log = true)
    {
        sdf_texture_buffer.load_npy(filename);
        sampler.texObj = sdf_texture_buffer.texObj;
        run_pipeline(filename, write_log);
    }

    // Homogenizes the built-in analytic gyroid (sampled into the same 257^3 texture).
    void run_builtin_gyroid(bool write_log = true)
    {
        BuiltinGyroid<T> sdf;
        sdf_texture_buffer.init_from_kernel(sdf);
        sampler.texObj = sdf_texture_buffer.texObj;
        run_pipeline("builtin_gyroid", write_log);
    }

    // Effective elasticity tensor of the last run (Voigt order xx, yy, zz, xy, yz, xz;
    // engineering shear strains).
    const CudaMat<T, 6, 6>& get_elastic_tensor() const { return approx_elastic_tensor; }

    // Weights W of the objective J = sum_IJ W_IJ C^H_IJ (to be minimized).
    void set_objective_weight(const CudaMat<T, 6, 6>& w) { objective_weight = w; }

    // Homogenizes the combined TPMS field sum_k coefs[k] f_k (6 coefficients) and returns J.
    T run_combined_field(const std::vector<float>& coefs, const std::string& name, bool write_log = true)
    {
        CombinedTPMSField sdf(coefs);
        sdf_texture_buffer.init_from_kernel(sdf);
        sampler.texObj = sdf_texture_buffer.texObj;
        run_pipeline(name, write_log);

        T objective = approx_elastic_tensor.cwiseProduct(objective_weight).sum();
        std::cout << "objective = " << objective << std::endl;
        return objective;
    }

    // dJ/dc_k (k = 0..5) of the last run_combined_field call, with the quadrature points, weights
    // and the active element set held fixed.
    void compute_gradient(std::vector<float>& grad_c)
    {
        // one block (4 warps) per element, one warp per quadrature point
        int blockSize = 128;
        int gridSize = nActiveTets;
        size_t smemSize = 6 * 342 * sizeof(T);  // >= 4 warps x 6 KinematicStates

        CudaVec<T, 3> *d_dC_hom_dg;
        CudaMat<T, 3, 3> *d_dC_hom_dHess;
        T *d_grad_c;

        // 36 entries (all C^H_IJ) per quadrature slot
        size_t max_quads = 4 * nActiveTets;
        checkCudaErrors(cudaMalloc((void**)&d_dC_hom_dg, max_quads * 36 * sizeof(CudaVec<T, 3>)));
        checkCudaErrors(cudaMalloc((void**)&d_dC_hom_dHess, max_quads * 36 * sizeof(CudaMat<T, 3, 3>)));
        checkCudaErrors(cudaMalloc((void**)&d_grad_c, 6 * sizeof(T)));
        checkCudaErrors(cudaMemset(d_grad_c, 0, 6 * sizeof(T)));

        RM_Gradients::compute_homogenization_gradients_kernel<<<gridSize, blockSize, smemSize>>>(
            background_grid.get_active_tets_ptr(),
            background_grid.get_quads_ptr(),
            background_grid.get_nodes_ptr(),
            thrust::raw_pointer_cast(d_displace_per_element.data()),
            d_dC_hom_dg,
            d_dC_hom_dHess, E, mu, lambda, thickness, nActiveTets, sampler);
        getLastCudaError("RM_Gradients::compute_homogenization_gradients_kernel");
        checkCudaErrors(cudaDeviceSynchronize());

        int blockAsm = 256;
        int gridAsm = (max_quads + blockAsm - 1) / blockAsm;
        assemble_design_gradient_kernel<<<gridAsm, blockAsm>>>(
            background_grid.get_quads_ptr(),
            objective_weight,
            d_dC_hom_dg,
            d_dC_hom_dHess,
            d_grad_c,
            nActiveTets
        );
        getLastCudaError("assemble_design_gradient_kernel");
        checkCudaErrors(cudaDeviceSynchronize());

        thrust::host_vector<T> h_grad_c(6);
        checkCudaErrors(cudaMemcpy(h_grad_c.data(), d_grad_c, 6 * sizeof(T), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 6; i++)
            grad_c[i] = h_grad_c[i];
        checkCudaErrors(cudaFree(d_dC_hom_dg));
        checkCudaErrors(cudaFree(d_dC_hom_dHess));
        checkCudaErrors(cudaFree(d_grad_c));
    }

private:
    // Full forward pass for the field currently bound to `sampler`.
    void run_pipeline(const std::string& name, bool write_log)
    {
        data_collector.clear();
        data_collector.name = name;
        data_collector.resolution = nx;

        ScopedMemoryMonitor mem_monitor;
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        init_background_grid();
        construct_stiffness_matrix();
        construct_stab_matrix();
        construct_test_strain_rhs();
        homogenization();
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        data_collector.total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "total time: " << data_collector.total_time << " ms" << std::endl;
        data_collector.memory = mem_monitor.get_peak_memory();
        std::cout << "peak memory: " << data_collector.memory << " MB" << std::endl;
        if (write_log)
            log.log(data_collector);
    }

    // Six unit test strains (engineering shear: off-diagonal entries 0.5).
    static thrust::device_vector<CudaMat<T, 3, 3>> make_test_strains()
    {
        CudaMat<T, 3, 3> t1(1.0, 0.0, 0.0,
                            0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0);
        CudaMat<T, 3, 3> t2(0.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 0.0);
        CudaMat<T, 3, 3> t3(0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0,
                            0.0, 0.0, 1.0);
        CudaMat<T, 3, 3> t5(0.0, 0.0, 0.0,
                            0.0, 0.0, 0.5,
                            0.0, 0.5, 0.0);
        CudaMat<T, 3, 3> t6(0.0, 0.0, 0.5,
                            0.0, 0.0, 0.0,
                            0.5, 0.0, 0.0);
        CudaMat<T, 3, 3> t4(0.0, 0.5, 0.0,
                            0.5, 0.0, 0.0,
                            0.0, 0.0, 0.0);
        return thrust::device_vector<CudaMat<T, 3, 3>>{t1, t2, t3, t4, t5, t6};
    }

    void init_background_grid()
    {
        timer.Tic();
        background_grid.refresh_grid();
        background_grid.optimize_grid(sampler);
        background_grid.filter_uncut_elements(sampler);
        background_grid.compute_quadrature_per_element(sampler);
        background_grid.re_index();
        nActiveTets = background_grid.get_nActive_tets();
        nActiveNodes = background_grid.get_nActive_nodes();
        timer.Toc();
        data_collector.init_grid_time = timer.Elapsed();
    }

    // Shell stiffness (dense per element, kept for C^H) + its CSR assembly
    // (which also includes the normal penalty of w).
    void construct_stiffness_matrix()
    {
        d_stiffness_matrices.resize(nActiveTets);
        d_K_stiffness_row_indices.resize(nActiveTets * 24 * 24);
        d_K_stiffness_col_indices.resize(nActiveTets * 24 * 24);
        d_K_stiffness_val.resize(nActiveTets * 24 * 24);
        int blockSize = 256;
        int gridSize = (nActiveTets + (blockSize / 32)) / (blockSize / 32);
        size_t smemSize = 2 * 24 * 24 * sizeof(T) * (blockSize / 32);
        timer.Tic();
        RM_Mat::construct_stiff_element_mat<T><<<gridSize, blockSize, smemSize>>>(
            background_grid.get_active_tets_ptr(),
            background_grid.get_quads_ptr(),
            background_grid.get_nodes_ptr(),
            background_grid.get_compact_node_indices_ptr(),
            thrust::raw_pointer_cast(d_stiffness_matrices.data()),
            thrust::raw_pointer_cast(d_K_stiffness_row_indices.data()),
            thrust::raw_pointer_cast(d_K_stiffness_col_indices.data()),
            thrust::raw_pointer_cast(d_K_stiffness_val.data()),
            E,
            mu,
            lambda,
            thickness,
            nActiveNodes,
            nActiveTets,
            sampler
        );
        getLastCudaError("RM_Mat::construct_stiff_element_mat");
        checkCudaErrors(cudaDeviceSynchronize());
        timer.Toc();
        std::cout << "construct stiffness matrix time = " << timer.Elapsed() << "ms" << std::endl;
        data_collector.construct_stiffness_time = timer.Elapsed();
        timer.Tic();
        assemble_global_matrix_fast(d_K_stiffness_row_indices, d_K_stiffness_col_indices, d_K_stiffness_val);
        timer.Toc();
        std::cout << "assemble stiffness matrix time = " << timer.Elapsed() << "ms" << std::endl;
        data_collector.construct_stiffness_time += timer.Elapsed();
    }

    // Volumetric stabilization, assembled into a CSR matrix with the same pattern as the stiffness.
    void construct_stab_matrix()
    {
        d_K_stab_row_indices.resize(nActiveTets * 24 * 24);
        d_K_stab_col_indices.resize(nActiveTets * 24 * 24);
        d_K_stab_val.resize(nActiveTets * 24 * 24);
        int blockSize = 128;
        int gridSize = (nActiveTets * 32 + blockSize - 1) / blockSize;
        size_t smemSize = 4 * 24 * 24 * sizeof(T);
        timer.Tic();
        // dense per-element copy written by the kernel (not used afterwards)
        thrust::device_vector<CudaMat<T, 24, 24>> d_stab_matrics(nActiveTets);
        T rho = (dx + 1.0f / dx) * 10000.0;
        RM_Mat::construct_stab_element_mat <<<gridSize, blockSize, smemSize>>>(
            background_grid.get_active_tets_ptr(),
            background_grid.get_nodes_ptr(),
            background_grid.get_compact_node_indices_ptr(),
            thrust::raw_pointer_cast(d_K_stab_row_indices.data()),
            thrust::raw_pointer_cast(d_K_stab_col_indices.data()),
            thrust::raw_pointer_cast(d_K_stab_val.data()),
            thrust::raw_pointer_cast(d_stab_matrics.data()),
            nActiveTets,
            E,
            mu,
            lambda,
            thickness,
            rho,
            nActiveNodes,
            sampler
        );
        getLastCudaError("RM_Mat::construct_stab_element_mat");
        checkCudaErrors(cudaDeviceSynchronize());
        timer.Toc();
        std::cout << "construct stab matrix time = " << timer.Elapsed() << "ms" << std::endl;
        data_collector.construct_stab_time = timer.Elapsed();
        timer.Tic();
        assemble_global_matrix_fast(d_K_stab_row_indices, d_K_stab_col_indices, d_K_stab_val);
        timer.Toc();
        data_collector.construct_stab_time += timer.Elapsed();
        std::cout << "assemble stab matrix time = " << timer.Elapsed() << "ms" << std::endl;
    }

    // Right-hand sides of the six cell problems.
    void construct_test_strain_rhs()
    {
        thrust::device_vector<CudaMat<T, 3, 3>> d_test_strains = make_test_strains();
        int blockSize = 256;
        int gridSize = (nActiveTets + (blockSize / 32)) / (blockSize / 32);
        size_t smemSize = 24 * sizeof(T) * (blockSize / 32);
        timer.Tic();
        for (int k = 0; k < 6; ++k)
        {
            d_rhs_indices[k].resize(nActiveTets * 24);
            d_rhs_val[k].resize(nActiveTets * 24);
        }
        for (int k = 0; k < 6; ++k)
        {
            RM_Mat::compute_test_strain_rhs<<<gridSize, blockSize, smemSize>>>(
                background_grid.get_active_tets_ptr(),
                background_grid.get_quads_ptr(),
                background_grid.get_nodes_ptr(),
                background_grid.get_compact_node_indices_ptr(),
                thrust::raw_pointer_cast(d_rhs_indices[k].data()),
                thrust::raw_pointer_cast(d_rhs_val[k].data()),
                thrust::raw_pointer_cast(d_test_strains.data()),
                E,
                mu,
                lambda,
                thickness, k, nActiveNodes, nActiveTets, sampler);
            getLastCudaError("RM_Mat::compute_test_strain_rhs");
            checkCudaErrors(cudaDeviceSynchronize());
        }
        timer.Toc();
        std::cout << "compute test strain rhs time = " << timer.Elapsed() << "ms" << std::endl;
        data_collector.construct_rhs_time = timer.Elapsed();
        timer.Tic();
        for (int k = 0; k < 6; ++k)
            assemble_global_vector(d_rhs_indices[k], d_rhs_val[k]);
        timer.Toc();
        data_collector.construct_rhs_time += timer.Elapsed();
        std::cout << "assemble global vector time = " << timer.Elapsed() << "ms" << std::endl;
    }

    // Element COO (row, col, val with 576 entries per element) -> CSR, in place:
    // d_row becomes the CSR row pointer, d_col / d_val the column indices / values.
    void assemble_global_matrix_fast(
        thrust::device_vector<int>& d_row,
        thrust::device_vector<int>& d_col,
        thrust::device_vector<T>& d_val)
    {
        int n_pairs = nActiveTets * 16;
        if (m_pair_keys.size() < n_pairs) {
            m_pair_keys.resize(n_pairs);
            m_sorted_keys.resize(n_pairs);
            m_pair_map.resize(n_pairs);
        }

        int blockSize = 256;
        int gridSize = (nActiveTets + blockSize - 1) / blockSize;

        extract_node_pairs_kernel<<<gridSize, blockSize>>>(
            background_grid.get_compact_node_indices_ptr(),
            thrust::raw_pointer_cast(m_pair_keys.data()),
            nActiveTets
        );

        thrust::copy(m_pair_keys.begin(), m_pair_keys.begin() + n_pairs, m_sorted_keys.begin());
        thrust::sort(m_sorted_keys.begin(), m_sorted_keys.begin() + n_pairs);

        if (m_unique_keys.size() < n_pairs) m_unique_keys.resize(n_pairs);
        auto new_end = thrust::unique_copy(
            m_sorted_keys.begin(), m_sorted_keys.begin() + n_pairs,
            m_unique_keys.begin()
        );
        int n_unique = new_end - m_unique_keys.begin();

        thrust::lower_bound(
            thrust::device,
            m_unique_keys.begin(), m_unique_keys.begin() + n_unique,
            m_pair_keys.begin(), m_pair_keys.begin() + n_pairs,
            m_pair_map.begin()
        );

        if (m_block_values.size() < n_unique * 36) m_block_values.resize(n_unique * 36);
        thrust::fill(m_block_values.begin(), m_block_values.begin() + n_unique * 36, (T)0.0);

        accumulate_block_values_kernel<T><<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_val.data()),
            thrust::raw_pointer_cast(m_pair_map.data()),
            thrust::raw_pointer_cast(m_block_values.data()),
            nActiveTets
        );

        if (m_node_degrees.size() < nActiveNodes + 1) m_node_degrees.resize(nActiveNodes + 1);
        thrust::fill(m_node_degrees.begin(), m_node_degrees.end(), 0);

        int gridUnique = (n_unique + blockSize - 1) / blockSize;
        count_node_degree_kernel<<<gridUnique, blockSize>>>(
            thrust::raw_pointer_cast(m_unique_keys.data()),
            thrust::raw_pointer_cast(m_node_degrees.data()),
            n_unique
        );

        if (m_graph_row_ptr.size() < nActiveNodes + 1) m_graph_row_ptr.resize(nActiveNodes + 1);
        thrust::exclusive_scan(m_node_degrees.begin(), m_node_degrees.begin() + nActiveNodes + 1, m_graph_row_ptr.begin());

        d_row.resize(6 * nActiveNodes + 1);
        int gridNodes = (nActiveNodes + blockSize - 1) / blockSize;
        generate_scalar_row_ptr_kernel<<<gridNodes, blockSize>>>(
            thrust::raw_pointer_cast(m_node_degrees.data()),
            thrust::raw_pointer_cast(d_row.data()),
            nActiveNodes
        );
        thrust::exclusive_scan(d_row.begin(), d_row.end(), d_row.begin());

        int total_nnz = n_unique * 36;
        d_col.resize(total_nnz);
        d_val.resize(total_nnz);

        expand_block_to_csr_kernel<T><<<gridUnique, blockSize>>>(
            thrust::raw_pointer_cast(m_unique_keys.data()),
            thrust::raw_pointer_cast(m_block_values.data()),
            thrust::raw_pointer_cast(m_graph_row_ptr.data()),
            thrust::raw_pointer_cast(d_row.data()),
            thrust::raw_pointer_cast(d_col.data()),
            thrust::raw_pointer_cast(d_val.data()),
            n_unique,
            nActiveNodes
        );
    }

    // Element right-hand sides (24 entries per element) -> dense global vector (summed by row),
    // then zeroes the fixed DOFs.
    void assemble_global_vector(thrust::device_vector<int>& d_row, thrust::device_vector<T>& d_val)
    {
        const size_t num_entries = d_row.size();
        if (num_entries == 0) return;

        thrust::sort_by_key(thrust::device, d_row.begin(), d_row.end(), d_val.begin());

        if (m_rhs_row_temp.size() < num_entries) m_rhs_row_temp.resize(num_entries);
        if (m_rhs_val_temp.size() < num_entries) m_rhs_val_temp.resize(num_entries);

        auto new_end = thrust::reduce_by_key(
            thrust::device,
            d_row.begin(), d_row.end(),
            d_val.begin(),
            m_rhs_row_temp.begin(),
            m_rhs_val_temp.begin(),
            cuda::std::equal_to<int>(),
            cuda::std::plus<T>()
        );

        size_t nnz = new_end.first - m_rhs_row_temp.begin();

        d_row.resize(nnz);
        d_val.resize(nnz);

        thrust::copy_n(thrust::device, m_rhs_row_temp.begin(), nnz, d_row.begin());
        thrust::copy_n(thrust::device, m_rhs_val_temp.begin(), nnz, d_val.begin());

        apply_fixed_point_rhs_kernel<<<1, 6>>>(thrust::raw_pointer_cast(d_val.data()), background_grid.get_fixed_node_x(), background_grid.get_nActive_nodes(), 0.0f);
        apply_fixed_point_rhs_kernel<<<1, 6>>>(thrust::raw_pointer_cast(d_val.data()), background_grid.get_fixed_node_y(), background_grid.get_nActive_nodes(), 0.0f);
    }

    // Solves (K_stiff + K_stab) x_k = rhs_k for the six test strains and evaluates
    // C^H(i, j) = sum_e y_i^T K_e y_j with y = g - x (unit-cube cell, |Omega| = 1).
    void homogenization()
    {
        timer.Tic();
        thrust::device_vector<CudaMat<T, 3, 3>> d_test_strains = make_test_strains();
        const int nDofs = 6 * background_grid.get_nActive_nodes();
        const int nnz = d_K_stiffness_val.size();
        if (nDofs + 1 != d_K_stiffness_row_indices.size() || nDofs + 1 != d_K_stab_row_indices.size() || nDofs != d_rhs_indices[0].size())
        {
            std::cout << "The linear system is not correct." << std::endl;
            return;
        }

        data_collector.matrix_size = nDofs;
        data_collector.num_activeTets = nActiveTets;
        data_collector.num_quads = 4 * nActiveTets;

        std::cout << "The matrix size = " << nDofs << std::endl;
        for (int k = 0; k < 6; ++k)
            d_disp[k].resize(nDofs);
        // system matrix A = K_stiff + K_stab (identical sparsity pattern)
        thrust::device_vector<T> d_A_val(d_K_stiffness_val.size());
        thrust::transform(d_K_stiffness_val.begin(), d_K_stiffness_val.end(), d_K_stab_val.begin(), d_A_val.begin(), cuda::std::plus<T>());
        // fixed nodes
        T penalty_value = 1e5;
        int blockSize = 256;
        int gridSize = (nnz + blockSize - 1) / blockSize;
        apply_fixed_point_col_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_K_stiffness_row_indices.data()),
                                      thrust::raw_pointer_cast(d_K_stiffness_col_indices.data()),
                                      thrust::raw_pointer_cast(d_A_val.data()), background_grid.get_fixed_node_x(), nActiveNodes, nnz, penalty_value);
        getLastCudaError("apply_fixed_point_col_kernel");
        checkCudaErrors(cudaDeviceSynchronize());
        apply_fixed_point_row_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_K_stiffness_row_indices.data()),
                                      thrust::raw_pointer_cast(d_K_stiffness_col_indices.data()),
                                      thrust::raw_pointer_cast(d_A_val.data()), background_grid.get_fixed_node_x(), nActiveNodes, penalty_value);
        getLastCudaError("apply_fixed_point_row_kernel");
        checkCudaErrors(cudaDeviceSynchronize());
        apply_fixed_point_col_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_K_stiffness_row_indices.data()),
                                      thrust::raw_pointer_cast(d_K_stiffness_col_indices.data()),
                                      thrust::raw_pointer_cast(d_A_val.data()), background_grid.get_fixed_node_y(), nActiveNodes, nnz, penalty_value);
        getLastCudaError("apply_fixed_point_col_kernel");
        checkCudaErrors(cudaDeviceSynchronize());
        apply_fixed_point_row_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_K_stiffness_row_indices.data()),
                                      thrust::raw_pointer_cast(d_K_stiffness_col_indices.data()),
                                      thrust::raw_pointer_cast(d_A_val.data()), background_grid.get_fixed_node_y(), nActiveNodes, penalty_value);
        getLastCudaError("apply_fixed_point_row_kernel");
        checkCudaErrors(cudaDeviceSynchronize());
        timer.Toc();
        std::cout << "homogenization preparation time = " << timer.Elapsed() << "ms" << std::endl;

        // linear solve (zero initial guess, relative tolerance 1e-2, at most 2000 iterations)
        {
            timer.Tic();
            BlockPCGSolver solver(nDofs, nnz, 6, thrust::raw_pointer_cast(d_K_stiffness_row_indices.data()), thrust::raw_pointer_cast(d_K_stiffness_col_indices.data()), thrust::raw_pointer_cast(d_A_val.data()));
            timer.Toc();
            std::cout << "solver setup time = " << timer.Elapsed() << "ms" << std::endl;
            timer.Tic();
            thrust::device_vector<T> d_total_rhs(6 * nDofs, (T)(0));
            thrust::device_vector<T> d_total_disp(6 * nDofs, (T)(0));
            for (int k = 0; k < 6; ++k)
                thrust::copy(d_rhs_val[k].begin(), d_rhs_val[k].end(), d_total_rhs.begin() + k * nDofs);
            timer.Toc();
            std::cout << "copy rhs time = " << timer.Elapsed() << "ms" << std::endl;
            timer.Tic();
            solver.solve(thrust::raw_pointer_cast(d_total_rhs.data()), thrust::raw_pointer_cast(d_total_disp.data()), 2000, 1e-2);
            timer.Toc();
            std::cout << "solve 6 linear systems time = " << timer.Elapsed() << "ms" << std::endl;

            data_collector.solve_time = timer.Elapsed();

            timer.Tic();
            for (int k = 0; k < 6; ++k)
                thrust::copy(d_total_disp.begin() + k * nDofs, d_total_disp.begin() + (k + 1) * nDofs, d_disp[k].begin());
            timer.Toc();
            std::cout << "copy results time = " << timer.Elapsed() << "ms" << std::endl;
        }

        approx_elastic_tensor.setZero();
        timer.Tic();
        d_displace_per_element.resize(nActiveTets);
        blockSize = 256;
        gridSize = (nActiveTets + blockSize - 1) / blockSize;
        pack_displacement_per_element_kernel<<<gridSize, blockSize>>>(
            background_grid.get_active_tets_ptr(),
            background_grid.get_nodes_ptr(),
            background_grid.get_compact_node_indices_ptr(),
            thrust::raw_pointer_cast(d_disp[0].data()),
            thrust::raw_pointer_cast(d_disp[1].data()),
            thrust::raw_pointer_cast(d_disp[2].data()),
            thrust::raw_pointer_cast(d_disp[3].data()),
            thrust::raw_pointer_cast(d_disp[4].data()),
            thrust::raw_pointer_cast(d_disp[5].data()),
            thrust::raw_pointer_cast(d_test_strains.data()),
            thrust::raw_pointer_cast(d_displace_per_element.data()),
            nActiveTets, nActiveNodes);
        getLastCudaError("pack_displacement_per_element_kernel");
        checkCudaErrors(cudaDeviceSynchronize());
        timer.Toc();
        std::cout << "pack displacement per element time = " << timer.Elapsed() << "ms" << std::endl;
        thrust::device_vector<CudaMat<T, 6, 6>> d_elastic_tensor_per_element(nActiveTets);
        // 4 warps per block, one warp per element
        blockSize = 4 * 32;
        gridSize = (nActiveTets + 4 - 1) / 4;
        size_t smem_size = 4 * sizeof(WarpSmemData<T>);
        timer.Tic();
        homogenization_per_element_kernel<<<gridSize, blockSize, smem_size>>>(
            thrust::raw_pointer_cast(d_stiffness_matrices.data()),
            thrust::raw_pointer_cast(d_displace_per_element.data()),
            thrust::raw_pointer_cast(d_elastic_tensor_per_element.data()), nActiveTets);
        getLastCudaError("homogenization_per_element_kernel");
        checkCudaErrors(cudaDeviceSynchronize());
        CudaMat<T, 6, 6> init;
        init.setZero();
        approx_elastic_tensor = thrust::reduce(
            d_elastic_tensor_per_element.begin(),
            d_elastic_tensor_per_element.end(),
            init,
            cuda::std::plus<CudaMat<T, 6, 6>>());
        timer.Toc();
        std::cout << "homogenization per element time = " << timer.Elapsed() << "ms" << std::endl;
        std::cout << approx_elastic_tensor << std::endl;
        data_collector.homogenization_matrix = approx_elastic_tensor;
    }

    GpuTimer timer;
    TetsGrid<T> background_grid;
    int nx, ny, nz;
    T dx;
    T E;
    T nu;
    T lambda;
    T mu;
    T thickness;
    int nActiveTets;
    int nActiveNodes;
    SDFSampler sampler;
    SDFTextureBuffer sdf_texture_buffer;

    // dense per-element shell stiffness (no stabilization)
    thrust::device_vector<CudaMat<T, 24, 24>> d_stiffness_matrices;
    // CSR shell stiffness (+ normal penalty of w)
    thrust::device_vector<int> d_K_stiffness_row_indices;
    thrust::device_vector<int> d_K_stiffness_col_indices;
    thrust::device_vector<T> d_K_stiffness_val;
    // CSR volumetric stabilization
    thrust::device_vector<int> d_K_stab_row_indices;
    thrust::device_vector<int> d_K_stab_col_indices;
    thrust::device_vector<T> d_K_stab_val;
    // right-hand sides and solutions of the six cell problems
    thrust::device_vector<int> d_rhs_indices[6];
    thrust::device_vector<T> d_rhs_val[6];
    thrust::device_vector<T> d_disp[6];
    // per-element local vectors y = g - x (24 x 6), also the input of the sensitivity pass
    thrust::device_vector<CudaMat<T, 24, 6>> d_displace_per_element;
    CudaMat<T, 6, 6> approx_elastic_tensor;
    CudaMat<T, 6, 6> objective_weight;

    // assembly workspace
    thrust::device_vector<unsigned long long> m_pair_keys;
    thrust::device_vector<unsigned long long> m_sorted_keys;
    thrust::device_vector<unsigned long long> m_unique_keys;
    thrust::device_vector<int> m_pair_map;
    thrust::device_vector<T> m_block_values;
    thrust::device_vector<int> m_node_degrees;
    thrust::device_vector<int> m_graph_row_ptr;
    thrust::device_vector<int> m_rhs_row_temp;
    thrust::device_vector<T> m_rhs_val_temp;

    log_data data_collector;
    logger log;
};
