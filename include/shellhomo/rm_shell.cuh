#pragma once
// Reissner--Mindlin shell operators (tangential differential calculus) evaluated from the
// implicit field, and the element kernels of the TraceFEM discretization with linear tets.
//
// Local element DOFs (24): index 0..11 -> u, 12..23 -> w; within each block,
// DOF i belongs to vertex i / 3, component i % 3.
// Global DOFs: u of compact node a at 3a + c, w at 3 * nActiveNodes + 3a + c.

#include <cmath>
#include <thrust/device_vector.h>
#include "linalg.cuh"
#include "sdf.cuh"
#include "grid.cuh"

constexpr int nNodes = 4;
constexpr int element_mat_size = 24;

namespace RM_Mat
{
    // ---------------------------------------------------------------------
    // Geometry from the implicit field f
    // ---------------------------------------------------------------------

    // g = grad f
    template <typename T>
    __device__ CudaVec<T, 3> dF_dx(CudaVec<T, 3> &pos, SDFSampler& sampler)
    {
        return TextureBuffer::shell_grad(pos, sampler);
    }

    // n = g / |g|
    template <typename T>
    __device__ CudaVec<T, 3> dF_dx_norm(CudaVec<T, 3> &pos, SDFSampler& sampler)
    {
        CudaVec<T, 3> dF_dx = RM_Mat::dF_dx(pos, sampler);
        return dF_dx / dF_dx.norm();
    }

    // P = I - n n^T
    template <typename T>
    __device__ CudaMat<T, 3, 3> project_mat_P(CudaVec<T, 3> &pos, SDFSampler& sampler)
    {
        CudaVec<T, 3> n = RM_Mat::dF_dx(pos, sampler);
        n /= n.norm();
        CudaMat<T, 3, 3> I;
        I.setIdentity();
        CudaMat<T, 3, 3> P = I - outer_product<T, 3>(n, n);
        return P;
    }

    // Q = n n^T
    template <typename T>
    __device__ CudaMat<T, 3, 3> project_mat_Q(CudaVec<T, 3> &pos, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> P = RM_Mat::project_mat_P(pos, sampler);
        CudaMat<T, 3, 3> I;
        I.setIdentity();
        CudaMat<T, 3, 3> Q = I - P;
        return Q;
    }

    // M = Hessian of f: forward difference of g with a one-texel step, symmetrized.
    template <typename T>
    __device__ CudaMat<T, 3, 3> d2F_dx2(CudaVec<T, 3> &pos, SDFSampler& sampler)
    {
        T delta = sampler.texel_size;
        CudaVec<T, 3> g0 = RM_Mat::dF_dx(pos, sampler);
        CudaVec<T, 3> pos_x = pos;
        pos_x[0] += delta;
        CudaVec<T, 3> g_x = RM_Mat::dF_dx(pos_x, sampler);
        CudaVec<T, 3> pos_y = pos;
        pos_y[1] += delta;
        CudaVec<T, 3> g_y = RM_Mat::dF_dx(pos_y, sampler);
        CudaVec<T, 3> pos_z = pos;
        pos_z[2] += delta;
        CudaVec<T, 3> g_z = RM_Mat::dF_dx(pos_z, sampler);
        CudaMat<T, 3, 3> H;
        H.setCol(0, (g_x - g0) / delta);
        H.setCol(1, (g_y - g0) / delta);
        H.setCol(2, (g_z - g0) / delta);
        H = 0.5 * (H + H.transpose());
        return H;
    }

    // grad n = P M / |g|
    template <typename T>
    __device__ CudaMat<T, 3, 3> dn_dx(CudaVec<T, 3> &pos, SDFSampler& sampler)
    {
        CudaVec<T, 3> dF_dx = RM_Mat::dF_dx(pos, sampler);
        CudaMat<T, 3, 3> P = RM_Mat::project_mat_P(pos, sampler);
        CudaMat<T, 3, 3> Hessian = RM_Mat::d2F_dx2(pos, sampler);
        CudaMat<T, 3, 3> dn_dx = (1.0 / dF_dx.norm()) * P * Hessian;
        return dn_dx;
    }

    template <typename T>
    struct CudaMat_wrap_3
    {
        CudaMat<T, 3, 3> val[3];
        __device__ __host__ CudaMat<T, 3, 3>& operator[](int i) { return val[i]; }
        __device__ __host__ const CudaMat<T, 3, 3>& operator[](int i) const { return val[i]; }
    };

    // dP/dx_i = -(dn/dx_i n^T + n dn/dx_i^T), i = 0, 1, 2
    template <typename T>
    __device__ CudaMat_wrap_3<T> grad_P(CudaVec<T, 3> &pos, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> dn_dx = RM_Mat::dn_dx(pos, sampler);
        CudaVec<T, 3> dF_dx = RM_Mat::dF_dx(pos, sampler);  // unused; kept to match the reference build
        CudaVec<T, 3> n = RM_Mat::dF_dx_norm(pos, sampler);
        CudaMat_wrap_3<T> grad_P;
        #pragma unroll
        for (int i = 0; i < 3; i++)
            grad_P[i] = -1.0 * (outer_product<T, 3>(dn_dx.col(i), n) + outer_product<T, 3>(n, dn_dx.col(i)));
        return grad_P;
    }

    // Weingarten map H = grad n P = P M P / |g|
    template <typename T>
    __device__ CudaMat<T, 3, 3> Weingarten_mat(CudaVec<T, 3> &pos, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> dn_dx = RM_Mat::dn_dx(pos, sampler);
        CudaMat<T, 3, 3> P = RM_Mat::project_mat_P(pos, sampler);
        CudaMat<T, 3, 3> Weingarten_mat = dn_dx * P;
        return Weingarten_mat;
    }

    // ---------------------------------------------------------------------
    // Discrete differential operators of a single local DOF (lane_id in 0..11).
    // D = dN/dxi (constant for linear tets), XD_inv = (X D)^{-1}.
    // ---------------------------------------------------------------------

    // G_i = E_i D (XD)^{-1}
    template <typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> grad_operator(CudaVec<T, 3> &pos, CudaMat<T, nNodes, 3> & D, CudaMat<T, 3, 3> & XD_inv, int lane_id, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> P = RM_Mat::project_mat_P(pos, sampler);  // unused; kept to match the reference build
        CudaMat<T, 3, 3> dir_grad;
        CudaMat<T, 3, nNodes> dU_dUi;
        dU_dUi.setZero();
        dU_dUi(lane_id % 3, lane_id / 3) = 1.0;
        dir_grad = dU_dUi * D * XD_inv;
        return dir_grad;
    }

    // Directional derivative G_i P
    template <typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> dir_grad_operator(CudaVec<T, 3> &pos, CudaMat<T, nNodes, 3> & D, CudaMat<T, 3, 3> & XD_inv, int lane_id, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> P = RM_Mat::project_mat_P(pos, sampler);
        CudaMat<T, 3, 3> dir_grad;
        CudaMat<T, 3, nNodes> dU_dUi;
        dU_dUi.setZero();
        dU_dUi(lane_id % 3, lane_id / 3) = 1.0;
        dir_grad = dU_dUi * D * XD_inv * P;
        return dir_grad;
    }

    // Covariant derivative P G_i P
    template <typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> cov_grad_operator(CudaVec<T, 3> &pos, CudaMat<T, nNodes, 3> & D, CudaMat<T, 3, 3> & XD_inv, int lane_id, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> P = RM_Mat::project_mat_P(pos, sampler);
        CudaMat<T, 3, 3> cov_grad;
        CudaMat<T, 3, nNodes> dU_dUi;
        dU_dUi.setZero();
        dU_dUi(lane_id % 3, lane_id / 3) = 1.0;
        cov_grad = P * dU_dUi * D * XD_inv * P;
        return cov_grad;
    }

    // Directional derivative of the tangential restriction P w (test side):
    // P G_i P + [dP/dx_0 E_i N, dP/dx_1 E_i N, dP/dx_2 E_i N]
    // (only contracted with in-plane resultants, so the outer projections are implied).
    template <typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> dir_grad_tilde_operator(CudaVec<T, 3> &pos, CudaMat<T, nNodes, 3> &D, CudaMat<T, 3, 3> &XD_inv, CudaVec<T, nNodes> &N, int lane_id, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> P = RM_Mat::project_mat_P(pos, sampler);
        CudaMat_wrap_3<T> grad_P = RM_Mat::grad_P(pos, sampler);
        CudaMat<T, 3, nNodes> dU_dUi;
        CudaMat<T, 3, 3> dir_grad_tilde;
        dU_dUi.setZero();
        dU_dUi(lane_id % 3, lane_id / 3) = 1.0;
        CudaMat<T, 3, 3> cov = P * dU_dUi * D * XD_inv * P;
        CudaMat<T, 3, 3> part1;
        part1.setCol(0, grad_P[0] * dU_dUi * N);
        part1.setCol(1, grad_P[1] * dU_dUi * N);
        part1.setCol(2, grad_P[2] * dU_dUi * N);
        dir_grad_tilde = cov + part1;
        return dir_grad_tilde;
    }

    // Covariant derivative of the tangential restriction P w (trial side):
    // P G_i P + P [dP/dx_0 E_i N, dP/dx_1 E_i N, dP/dx_2 E_i N]
    template <typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> cov_grad_tilde_operator(CudaVec<T, 3> &pos, CudaMat<T, nNodes, 3> &D, CudaMat<T, 3, 3> &XD_inv, CudaVec<T, nNodes> &N, int lane_id, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> P = RM_Mat::project_mat_P(pos, sampler);
        CudaMat_wrap_3<T> grad_P = RM_Mat::grad_P(pos, sampler);
        CudaMat<T, 3, nNodes> dU_dUi;
        CudaMat<T, 3, 3> cov_grad_tilde;
        dU_dUi.setZero();
        dU_dUi(lane_id % 3, lane_id / 3) = 1.0;
        CudaMat<T, 3, 3> cov = P * dU_dUi * D * XD_inv * P;
        CudaMat<T, 3, 3> part1;
        part1.setCol(0, grad_P[0] * dU_dUi * N);
        part1.setCol(1, grad_P[1] * dU_dUi * N);
        part1.setCol(2, grad_P[2] * dU_dUi * N);
        cov_grad_tilde = cov + P * part1;
        return cov_grad_tilde;
    }

    // ---------------------------------------------------------------------
    // Strain operators
    // ---------------------------------------------------------------------

    // Membrane: sym(cov u)
    template <typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> plane_mem_strain_operator(CudaVec<T, 3> &pos, CudaMat<T, 3, 3> &cov_grad_operator, SDFSampler& sampler)
    {
        return 0.5 * (cov_grad_operator + cov_grad_operator.transpose());
    }

    // Bending, u part: sym(H dir u)
    template<typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> plane_bend_strain_operator_u_component(CudaVec<T, 3> &pos, CudaMat<T, 3, 3> &dir_grad_u_operator, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> H = RM_Mat::Weingarten_mat(pos, sampler);
        return 0.5 * (H * dir_grad_u_operator + dir_grad_u_operator.transpose() * H);
    }

    // Bending, w part: sym(cov P w)
    template<typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> plane_bend_strain_operator_w_component(CudaVec<T, 3> &pos, CudaMat<T, 3, 3>& cov_grad_tilde_w_operator, SDFSampler& sampler)
    {
        return 0.5 * (cov_grad_tilde_w_operator + cov_grad_tilde_w_operator.transpose());
    }

    // Transverse shear, u part: sym(Q dir u)
    template<typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> transverse_shear_strain_operator_u_component(CudaVec<T, 3> &pos, CudaMat<T, 3, 3>& dir_grad_u_operator, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> Q = RM_Mat::project_mat_Q(pos, sampler);
        return 0.5 * (Q * dir_grad_u_operator + dir_grad_u_operator.transpose() * Q);
    }

    // Transverse shear, w part: sym(n (P w)^T)
    template<typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> transverse_shear_strain_operator_w_component(CudaVec<T, 3> &pos, CudaVec<T, nNodes> & N_w, int lane_id, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> P = RM_Mat::project_mat_P(pos, sampler);
        CudaVec<T, 3> n = RM_Mat::dF_dx_norm(pos, sampler);
        CudaMat<T, 3, nNodes> dW_dwi;
        dW_dwi.setZero();
        dW_dwi(lane_id % 3, lane_id / 3) = 1.0;
        CudaVec<T, 3> aux = P * dW_dwi * N_w;
        return 0.5 * (outer_product(n, aux) + outer_product(aux, n));
    }

    // ---------------------------------------------------------------------
    // Stress and stress resultants
    // ---------------------------------------------------------------------

    // sigma = 2 mu eps + lambda tr(eps) I
    template<typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> linear_stress_operator(CudaMat<T, 3, 3>& strain_subMat, const T mu, const T lambda)
    {
        CudaMat<T, 3, 3> I;
        I.setIdentity();
        CudaMat<T, 3, 3> stress = 2 * mu * strain_subMat + lambda * strain_subMat.trace() * I;
        return stress;
    }

    // Bending moment m = t^3/12 P sigma P
    template<typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> stress_resultant_m_operator(CudaVec<T, 3> &pos, CudaMat<T, 3, 3>& strain_subMat, const T mu, const T lambda, const T t, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> P = RM_Mat::project_mat_P(pos, sampler);
        CudaMat<T, 3, 3> stress = linear_stress_operator(strain_subMat, mu, lambda);
        return (t * t * t / 12.0) * P * stress * P;
    }

    // Membrane force n = t P sigma P
    template<typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> stress_resultant_n_operator(CudaVec<T, 3> &pos, CudaMat<T, 3, 3>& strain_subMat, const T mu, const T lambda, const T t, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> P = RM_Mat::project_mat_P(pos, sampler);
        CudaMat<T, 3, 3> stress = linear_stress_operator(strain_subMat, mu, lambda);
        return t * P * stress * P;
    }

    // Transverse shear force q = t (Q sigma + sigma Q)
    template<typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> stress_resultant_q_operator(CudaVec<T, 3> &pos, CudaMat<T, 3, 3>& strain_subMat, const T mu, const T lambda, const T t, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> Q = RM_Mat::project_mat_Q(pos, sampler);
        CudaMat<T, 3, 3> stress = linear_stress_operator(strain_subMat, mu, lambda);
        return t * (Q * stress + stress * Q);
    }

    // Resultants of a constant macroscopic test strain eps (u = eps x, w = 0).
    template<typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> stress_resultant_m_operator_test_strain(CudaVec<T, 3> &pos, CudaMat<T, 3, 3> &test_strain, const T mu, const T lambda, const T t, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> P = RM_Mat::project_mat_P(pos, sampler);
        CudaMat<T, 3, 3> H = RM_Mat::Weingarten_mat(pos, sampler);
        CudaMat<T, 3, 3> in_plane_bend_strain = 0.5 * (H * test_strain * P + (test_strain * P).transpose() * H);
        CudaMat<T, 3, 3> stress = RM_Mat::linear_stress_operator(in_plane_bend_strain, mu, lambda);
        CudaMat<T, 3, 3> m = t * t * t * P * stress * P / 12.0;
        return m;
    }

    template<typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> stress_resultant_n_operator_test_strain(CudaVec<T, 3> &pos, CudaMat<T, 3, 3> &test_strain, const T mu, const T lambda, const T t, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> P = RM_Mat::project_mat_P(pos, sampler);
        CudaMat<T, 3, 3> in_plane_strain = P * test_strain * P;
        CudaMat<T, 3, 3> stress = RM_Mat::linear_stress_operator(in_plane_strain, mu, lambda);
        CudaMat<T, 3, 3> n = t * P * stress * P;
        return n;
    }

    template<typename T>
    __device__ __forceinline__ CudaMat<T, 3, 3> stress_resultant_q_operator_test_strain(CudaVec<T, 3> &pos, CudaMat<T, 3, 3> &test_strain, const T mu, const T lambda, const T t, SDFSampler& sampler)
    {
        CudaMat<T, 3, 3> P = RM_Mat::project_mat_P(pos, sampler);
        CudaMat<T, 3, 3> Q = RM_Mat::project_mat_Q(pos, sampler);
        CudaMat<T, 3, 3> transverse_strain = 0.5 * (Q * test_strain + test_strain * Q);
        CudaMat<T, 3, 3> stress = RM_Mat::linear_stress_operator(transverse_strain, mu, lambda);
        CudaMat<T, 3, 3> q = t * (Q * stress + stress * Q);
        return q;
    }

    // ---------------------------------------------------------------------
    // Element kernels
    // ---------------------------------------------------------------------

    // Shell stiffness of each active element (24x24, column-major), integrated over its surface
    // quadrature points. One warp per element: 8 lanes per quadrature point, each lane owns
    // columns j = sub_lane + 8k; quadrature contributions are summed with warp shuffles.
    // K_stiff receives the shell stiffness only (used to evaluate C^H); the COO values in
    // K_val additionally contain the normal penalty of w, E t (w.n)(v_w.n).
    template<typename T>
    __global__ void construct_stiff_element_mat(
        Element* elements, GaussianQuadSoA<T>* quad_soa, CudaVec<T, 3>* nodes, int* compact_indices,
        CudaMat<T, element_mat_size, element_mat_size> * K_stiff,
        int* K_row_idx, int* K_col_idx, T* K_val,
        const T E, const T mu, const T lambda, const T thickness, const int nActiveNodes, const int nActiveTets, SDFSampler sampler)
    {
        extern __shared__ T smem_pool[];

        int warp_id_in_block = threadIdx.x / 32;
        int lane_idx = threadIdx.x % 32;

        int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        int ele_idx = global_warp_id;
        if (ele_idx >= nActiveTets) return;

        T* smem_phys = &smem_pool[warp_id_in_block * 2 * 576];
        T* smem_stab = &smem_pool[warp_id_in_block * 2 * 576 + 576];

        for (int k = 0; k < 18; ++k) {
            smem_phys[lane_idx + k * 32] = 0.0;
            smem_stab[lane_idx + k * 32] = 0.0;
        }

        int quad_idx = lane_idx / 8;      // quadrature point handled by this lane
        int sub_lane = lane_idx % 8;

        Element tet = elements[ele_idx];
        int num_quad = quad_soa[ele_idx].count;
        bool active_quad = (quad_idx < num_quad);

        T q_weight = 0.0;
        CudaVec<T, 3> pos;
        CudaVec<T, 4> N_vec;
        CudaMat<T, 3, 3> P_mat, Q_mat, H_mat;
        CudaVec<T, 3> normal;
        CudaMat<T, 4, 3> D(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, -1.0);
        CudaMat<T, 3, 3> XD_inv;
        CudaMat<T, 3, 4> X;

        if (active_quad)
        {
            q_weight = quad_soa[ele_idx].weight[quad_idx];
            pos[0] = quad_soa[ele_idx].world_x[quad_idx];
            pos[1] = quad_soa[ele_idx].world_y[quad_idx];
            pos[2] = quad_soa[ele_idx].world_z[quad_idx];

            N_vec[0] = quad_soa[ele_idx].shape_r[quad_idx];
            N_vec[1] = quad_soa[ele_idx].shape_s[quad_idx];
            N_vec[2] = quad_soa[ele_idx].shape_t[quad_idx];
            N_vec[3] = quad_soa[ele_idx].shape_w[quad_idx];

            X.setCol(0, nodes[tet[0]]);
            X.setCol(1, nodes[tet[1]]);
            X.setCol(2, nodes[tet[2]]);
            X.setCol(3, nodes[tet[3]]);

            XD_inv = (X * D).inverse();

            P_mat = RM_Mat::project_mat_P(pos, sampler);
            Q_mat = RM_Mat::project_mat_Q(pos, sampler);
            H_mat = RM_Mat::Weingarten_mat(pos, sampler);
            normal = RM_Mat::dF_dx_norm(pos, sampler);
        }

        for (int k = 0; k < 3; k++)
        {
            int j = sub_lane + k * 8;
            bool j_is_u = (j < 12);

            CudaMat<T, 3, 4> dW_dwi;
            dW_dwi.setZero();
            dW_dwi((j % 12) % 3, (j % 12) / 3) = 1.0;

            // resultants of trial DOF j
            CudaMat<T, 3, 3> n_res, m_res, q_res;
            bool compute_j = active_quad;

            if (compute_j)
            {
                CudaMat<T, 3, 3> op_dir_j       = RM_Mat::dir_grad_operator(pos, D, XD_inv, j % 12, sampler);
                CudaMat<T, 3, 3> op_cov_j       = RM_Mat::cov_grad_operator(pos, D, XD_inv, j % 12, sampler);
                CudaMat<T, 3, 3> op_cov_tilde_j = RM_Mat::cov_grad_tilde_operator(pos, D, XD_inv, N_vec, j % 12, sampler);

                CudaMat<T, 3, 3> strain_mem = RM_Mat::plane_mem_strain_operator(pos, op_cov_j, sampler);
                n_res = RM_Mat::stress_resultant_n_operator(pos, strain_mem, mu, lambda, thickness, sampler);

                CudaMat<T, 3, 3> strain_bend;
                if (j_is_u)
                    strain_bend = RM_Mat::plane_bend_strain_operator_u_component(pos, op_dir_j, sampler);
                else
                    strain_bend = RM_Mat::plane_bend_strain_operator_w_component(pos, op_cov_tilde_j, sampler);
                m_res = RM_Mat::stress_resultant_m_operator(pos, strain_bend, mu, lambda, thickness, sampler);

                CudaMat<T, 3, 3> strain_shear;
                if (j_is_u)
                    strain_shear = RM_Mat::transverse_shear_strain_operator_u_component(pos, op_dir_j, sampler);
                else
                    strain_shear = RM_Mat::transverse_shear_strain_operator_w_component(pos, N_vec, j % 12, sampler);
                q_res = RM_Mat::stress_resultant_q_operator(pos, strain_shear, mu, lambda, thickness, sampler);
            }

            // contract with every test DOF i
            for (int i = 0; i < 24; i++)
            {
                T val_phys = 0.0;
                T val_stab = 0.0;
                if (compute_j)
                {
                    bool i_is_u = (i < 12);
                    CudaMat<T, 3, 4> dV_dvi;
                    dV_dvi.setZero();
                    dV_dvi((i % 12) % 3, (i % 12) / 3) = 1.0;

                    CudaMat<T, 3, 3> op_dir_i = RM_Mat::dir_grad_operator(pos, D, XD_inv, i % 12, sampler);

                    if (i_is_u)
                    {
                        // dir v_u : n + (H dir v_u) : m + (Q dir v_u) : q
                        val_phys += j < 12 ? op_dir_i.cwiseProduct(n_res).sum() : 0.0;
                        val_phys += (H_mat * op_dir_i).cwiseProduct(m_res).sum();
                        val_phys += (Q_mat * op_dir_i).cwiseProduct(q_res).sum();
                    }
                    else
                    {
                        // dir (P v_w) : m + (P v_w) . (q n)
                        CudaMat<T, 3, 3> op_dir_tilde_i = RM_Mat::dir_grad_tilde_operator(pos, D, XD_inv, N_vec, i % 12, sampler);
                        val_phys += op_dir_tilde_i.cwiseProduct(m_res).sum();
                        val_phys += (P_mat * dV_dvi * N_vec).dot(q_res * normal);
                        // normal penalty of w
                        if (j >= 12)
                            val_stab += E * thickness * (dW_dwi * N_vec).dot(normal) * (dV_dvi * N_vec).dot(normal);
                    }

                    val_phys *= q_weight;
                    val_stab *= q_weight;
                }

                // sum over the 4 quadrature points (lanes with equal sub_lane)
                val_phys += __shfl_xor_sync(0xffffffff, val_phys, 16);
                val_phys += __shfl_xor_sync(0xffffffff, val_phys, 8);

                val_stab += __shfl_xor_sync(0xffffffff, val_stab, 16);
                val_stab += __shfl_xor_sync(0xffffffff, val_stab, 8);

                if (lane_idx < 8)
                {
                    smem_phys[i + j * 24] = val_phys;
                    smem_stab[i + j * 24] = val_stab;
                }
            }
        }

        __syncthreads();

        int* global_row_ptr = &K_row_idx[ele_idx * 576];
        int* global_col_ptr = &K_col_idx[ele_idx * 576];
        T* global_val_ptr = &K_val[ele_idx * 576];
        int *tet_compact_indices = &compact_indices[ele_idx * 4];
        T *global_dense_ptr = reinterpret_cast<T *>(&K_stiff[ele_idx]);

        for (int idx = lane_idx; idx < 576; idx += 32)
        {
            T p_val = smem_phys[idx];
            T s_val = smem_stab[idx];

            global_dense_ptr[idx] = p_val;

            int row_local = idx % 24;
            int col_local = idx / 24;

            int r_dof_idx = row_local % 12;
            int r_node = r_dof_idx / 3;
            int r_dim  = r_dof_idx % 3;
            bool r_is_w = (row_local >= 12);
            int r_global_node = tet_compact_indices[r_node];
            global_row_ptr[idx] = 3 * r_global_node + r_dim + (r_is_w ? nActiveNodes * 3 : 0);

            int c_dof_idx = col_local % 12;
            int c_node = c_dof_idx / 3;
            int c_dim  = c_dof_idx % 3;
            bool c_is_w = (col_local >= 12);
            int c_global_node = tet_compact_indices[c_node];
            global_col_ptr[idx] = 3 * c_global_node + c_dim + (c_is_w ? nActiveNodes * 3 : 0);

            global_val_ptr[idx] = p_val + s_val;
        }
    }

    // Volumetric normal-derivative stabilization rho (grad u n).(grad v n) for the u-u and
    // w-w blocks, one-point rule at the element centroid. One warp per element.
    // Note: E, mu, lambda and thickness are not used by this kernel.
    template<typename T>
    __global__ void construct_stab_element_mat(
        Element* elements, CudaVec<T, 3>* nodes, int* compact_indices,
        int* K_row_idx, int* K_col_idx, T* K_val, CudaMat<T, 24, 24>* K_stab, const int nTets,
        const T E, const T mu, const T lambda, const T thickness, const T rho, const int nActiveNodes, SDFSampler sampler)
    {
        int global_thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        int ele_idx = global_thread_id / 32;
        int lane_idx = threadIdx.x % 32;
        int warp_id_in_block = threadIdx.x / 32;

        if (ele_idx >= nTets) return;

        extern __shared__ T smem_pool[];
        T* my_smem_K = &smem_pool[warp_id_in_block * 576];

        for (int k = lane_idx; k < 576; k += 32) {
            my_smem_K[k] = 0.0;
        }

        Element tet = elements[ele_idx];
        CudaMat<T, 3, 4> X;
        X.setCol(0, nodes[tet[0]]);
        X.setCol(1, nodes[tet[1]]);
        X.setCol(2, nodes[tet[2]]);
        X.setCol(3, nodes[tet[3]]);
        CudaVec<T, 3> E1 = X.col(1) - X.col(0);
        CudaVec<T, 3> E2 = X.col(2) - X.col(0);
        CudaVec<T, 3> E3 = X.col(3) - X.col(0);
        T volume = abs(E1.dot(E2.cross(E3)) / 6.0);
        CudaMat<T, 4, 3> D( 1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0,
                           -1.0, -1.0, -1.0);
        CudaMat<T, 3, 3> XD_inv = (X * D).inverse();

        CudaVec<T, 3> pos = (X.col(0) + X.col(1) + X.col(2) + X.col(3)) / 4.0;
        CudaVec<T, 3> normal = RM_Mat::dF_dx_norm(pos, sampler);

        // lane j owns column j
        if (lane_idx < 24)
        {
            int j = lane_idx;
            bool j_is_u = (j < 12);
            int j_dof = j % 12;

            CudaMat<T, 3, 3> op_grad_j = RM_Mat::grad_operator(pos, D, XD_inv, j_dof, sampler);

            for (int i = 0; i < 24; i++)
            {
                bool i_is_u = (i < 12);
                int i_dof = i % 12;

                CudaMat<T, 3, 3> op_grad_i = RM_Mat::grad_operator(pos, D, XD_inv, i_dof, sampler);

                T val = 0.0;
                if (i_is_u && j_is_u) {
                    val += (op_grad_i * normal).dot(op_grad_j * normal);
                }
                else if (!i_is_u && !j_is_u) {
                    val += (op_grad_i * normal).dot(op_grad_j * normal);
                }

                my_smem_K[j * 24 + i] = val * rho * volume;
            }
        }

        __syncthreads();

        int* global_row_ptr = &K_row_idx[ele_idx * 576];
        int* global_col_ptr = &K_col_idx[ele_idx * 576];
        T* global_val_ptr   = &K_val[ele_idx * 576];

        int *tet_compact_indices = &compact_indices[ele_idx * 4];
        T *K_stab_ptr = reinterpret_cast<T *>(&K_stab[ele_idx]);
        for (int k = lane_idx; k < 576; k += 32)
        {
            int i = k % 24;
            int j = k / 24;

            int global_row = tet_compact_indices[(i % 12) / 3] * 3 + (i % 12) % 3 + (i < 12 ? 0 : nActiveNodes * 3);
            int global_col = tet_compact_indices[(j % 12) / 3] * 3 + (j % 12) % 3 + (j < 12 ? 0 : nActiveNodes * 3);

            global_row_ptr[k] = global_row;
            global_col_ptr[k] = global_col;
            global_val_ptr[k] = my_smem_K[k];
            K_stab_ptr[k] = my_smem_K[k];
        }
    }

    // Element right-hand side of test strain id_strain: the shell bilinear form of every local
    // DOF against the affine field u = eps x, w = 0. One warp per element, 8 lanes per
    // quadrature point. Note: E is not used by this kernel.
    template<typename T>
    __global__ void compute_test_strain_rhs(
        Element* elements, GaussianQuadSoA<T>* quad_soa, CudaVec<T, 3>* nodes, int* compact_indices,
        int* K_row_idx, T* K_val, CudaMat<T, 3, 3>* test_strain_list,
        const T E, const T mu, const T lambda, const T thickness, int id_strain, const int nActiveNodes, const int nActiveTets, SDFSampler sampler)
    {
        extern __shared__ T smem_pool[];

        int warp_id_in_block = threadIdx.x / 32;
        int lane_idx = threadIdx.x % 32;

        int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        int ele_idx = global_warp_id;
        if (ele_idx >= nActiveTets) return;

        T* smem_rhs = &smem_pool[warp_id_in_block * 24];

        if (lane_idx < 24) {
            smem_rhs[lane_idx] = 0.0;
        }

        int quad_idx = lane_idx / 8;
        int sub_lane = lane_idx % 8;

        CudaMat<T, 3, 3> test_strain = test_strain_list[id_strain];
        Element tet = elements[ele_idx];
        int num_quad = quad_soa[ele_idx].count;
        bool active_quad = (quad_idx < num_quad);

        T q_weight = 0.0;
        CudaVec<T, 3> pos;
        CudaVec<T, 4> N_vec;

        CudaMat<T, 3, 3> P_mat, Q_mat, H_mat;
        CudaVec<T, 3> normal;
        CudaMat<T, 3, 3> n_res, m_res, q_res;

        CudaMat<T, 4, 3> D(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, -1.0);
        CudaMat<T, 3, 3> XD_inv;

        if (active_quad)
        {
            q_weight = quad_soa[ele_idx].weight[quad_idx];
            pos[0] = quad_soa[ele_idx].world_x[quad_idx];
            pos[1] = quad_soa[ele_idx].world_y[quad_idx];
            pos[2] = quad_soa[ele_idx].world_z[quad_idx];

            N_vec[0] = quad_soa[ele_idx].shape_r[quad_idx];
            N_vec[1] = quad_soa[ele_idx].shape_s[quad_idx];
            N_vec[2] = quad_soa[ele_idx].shape_t[quad_idx];
            N_vec[3] = quad_soa[ele_idx].shape_w[quad_idx];

            CudaMat<T, 3, 4> X;
            X.setCol(0, nodes[tet[0]]);
            X.setCol(1, nodes[tet[1]]);
            X.setCol(2, nodes[tet[2]]);
            X.setCol(3, nodes[tet[3]]);

            XD_inv = (X * D).inverse();

            P_mat = RM_Mat::project_mat_P(pos, sampler);
            Q_mat = RM_Mat::project_mat_Q(pos, sampler);
            H_mat = RM_Mat::Weingarten_mat(pos, sampler);
            normal = RM_Mat::dF_dx_norm(pos, sampler);

            n_res = RM_Mat::stress_resultant_n_operator_test_strain(pos, test_strain, mu, lambda, thickness, sampler);
            m_res = RM_Mat::stress_resultant_m_operator_test_strain(pos, test_strain, mu, lambda, thickness, sampler);
            q_res = RM_Mat::stress_resultant_q_operator_test_strain(pos, test_strain, mu, lambda, thickness, sampler);
        }

        for (int k = 0; k < 3; k++)
        {
            int i = sub_lane + k * 8;
            T val = 0.0;

            if (active_quad)
            {
                bool i_is_u = (i < 12);

                CudaMat<T, 3, 4> dV_dvi;
                dV_dvi.setZero();
                dV_dvi((i % 12) % 3, (i % 12) / 3) = 1.0;

                CudaMat<T, 3, 3> op_dir_grad_i = RM_Mat::dir_grad_operator(pos, D, XD_inv, i % 12, sampler);

                if (i_is_u)
                {
                    // dir v_u : n + (H dir v_u) : m + (Q dir v_u) : q
                    val += op_dir_grad_i.cwiseProduct(n_res).sum();
                    val += ((H_mat * op_dir_grad_i).cwiseProduct(m_res)).sum();
                    val += ((Q_mat * op_dir_grad_i).cwiseProduct(q_res)).sum();
                }
                else
                {
                    // dir (P v_w) : m + (P v_w) . (q n)
                    CudaMat<T, 3, 3> op_dir_grad_tilde_i = RM_Mat::dir_grad_tilde_operator(pos, D, XD_inv, N_vec, i % 12, sampler);
                    val += (op_dir_grad_tilde_i.cwiseProduct(m_res)).sum();
                    val += (P_mat * dV_dvi * N_vec).dot(q_res * normal);
                }
                val *= q_weight;
            }

            // sum over the 4 quadrature points
            val += __shfl_xor_sync(0xffffffff, val, 16);
            val += __shfl_xor_sync(0xffffffff, val, 8);

            if (lane_idx < 8)
            {
                smem_rhs[i] = val;
            }
        }

        __syncthreads();

        int *global_row_ptr = &K_row_idx[ele_idx * 24];
        T *global_val_ptr = &K_val[ele_idx * 24];
        int *tet_compact_indices = &compact_indices[ele_idx * 4];

        if (lane_idx < 24)
        {
            int idx = lane_idx;

            int node_idx = (idx % 12) / 3;
            int dof_idx  = (idx % 12) % 3;
            bool is_w    = (idx >= 12);
            int global_node = tet_compact_indices[node_idx];

            global_row_ptr[idx] = 3 * global_node + dof_idx + (is_w ? nActiveNodes * 3 : 0);
            global_val_ptr[idx] = smem_rhs[idx];
        }
    }
}
