#pragma once
// Sensitivity pass: derivatives of every entry C^H_IJ with respect to the field gradient g and
// the field Hessian at each surface quadrature point, with the quadrature points, weights and
// the active element set held fixed. Uses the cell-problem solutions of the forward pass.

#include "linalg.cuh"
#include "sdf.cuh"
#include "grid.cuh"
#include "rm_shell.cuh"

namespace RM_Gradients {

    // Displacement derivatives and stress resultants of one cell-problem solution
    // (test strain + fluctuation) at a quadrature point.
    template<typename T>
    struct KinematicState {
        CudaMat<T, 3, 3> grad_u;
        CudaMat<T, 3, 3> grad_w;
        CudaVec<T, 3> val_w;
        CudaMat<T, 3, 3> n_mem;
        CudaMat<T, 3, 3> m_bend;
        CudaMat<T, 3, 3> q_shear;
        CudaMat<T, 3, 3> sig_shear;
    };

    // Membrane term of d C^H / d g.
    template<typename T>
    __device__ __forceinline__ CudaVec<T, 3> Mem_operator(
        const CudaMat<T, 3, 3>& n_mem, const CudaMat<T, 3, 3>& grad_v,
        const CudaVec<T, 3>& n, const CudaMat<T, 3, 3>& P)
    {
        CudaMat<T, 3, 3> dir_grad_v = grad_v * P;
        CudaVec<T, 3> term1 = dir_grad_v.transpose() * n;
        CudaVec<T, 3> term2 = P * grad_v * n;
        return n_mem * (term1 + term2);
    }

    // Bending term of d C^H / d g.
    template<typename T>
    __device__ __forceinline__ CudaVec<T, 3> D_operator_k(
        const CudaMat<T, 3, 3>& M, const CudaMat<T, 3, 3>& grad_a, const CudaMat<T, 3, 3>& grad_b,
        const CudaVec<T, 3>& val_b, const CudaVec<T, 3>& n, const CudaMat<T, 3, 3>& P,
        const CudaMat<T, 3, 3>& H, const CudaMat<T, 3, 3>& Hess, T norm_g)
    {
        CudaMat<T, 3, 3> dir_grad_a = grad_a * P;
        CudaMat<T, 3, 3> cov_grad_a = P * grad_a * P;
        CudaMat<T, 3, 3> dir_grad_b = grad_b * P;
        CudaVec<T, 3> Hess_n = Hess * n;
        CudaVec<T, 3> P_Hess_n = P * Hess_n;

        CudaVec<T, 3> t1 = M.cwiseProduct(H * dir_grad_a).sum() * n;
        CudaVec<T, 3> t2 = (M * cov_grad_a.transpose() * Hess_n) / norm_g;
        CudaVec<T, 3> t3 = (H * M * dir_grad_a.transpose() * n);
        CudaVec<T, 3> t4 = (cov_grad_a * M * P_Hess_n) / norm_g;
        CudaVec<T, 3> t5 = (M * H * grad_a * n);
        CudaVec<T, 3> t6 = (P * val_b) * M.cwiseProduct(H).sum();
        CudaVec<T, 3> t7 = (M * dir_grad_b.transpose() * n);
        CudaVec<T, 3> t8 = (M * P * grad_b * n);

        T n_dot_b = n.dot(val_b);
        CudaVec<T, 3> t9 = n_dot_b * (n * M.cwiseProduct(H).sum() / norm_g);
        CudaVec<T, 3> t10 = n_dot_b * (2.0f * (M * P_Hess_n) / (norm_g * norm_g));

        return -(1.0f / norm_g) * (t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8) + t9 + t10;
    }

    // Transverse-shear term of d C^H / d g (variation of the strain).
    template<typename T>
    __device__ __forceinline__ CudaVec<T, 3> F_operator_k(
        const CudaMat<T, 3, 3>& q, const CudaMat<T, 3, 3>& grad_a, const CudaMat<T, 3, 3>& grad_b,
        const CudaVec<T, 3>& val_b, const CudaVec<T, 3>& n, const CudaMat<T, 3, 3>& P, const CudaMat<T, 3, 3>& Q, T norm_g)
    {
        CudaMat<T, 3, 3> dir_grad_a = grad_a * P;
        CudaMat<T, 3, 3> cov_grad_a = P * grad_a * P;

        CudaVec<T, 3> v1 = (P * q * dir_grad_a.transpose() * n);
        CudaVec<T, 3> v2 = (cov_grad_a * q * n);
        CudaVec<T, 3> v3 = (P * grad_a.transpose() * Q * q * n);
        CudaVec<T, 3> v4 = (P * q * Q * grad_a * n);

        T n_dot_b = n.dot(val_b);
        CudaVec<T, 3> v5 = (P * q * n);
        T n_q_n = n.dot(q * n);
        CudaVec<T, 3> P_b_k = (P * val_b);

        return (1.0f / norm_g) * (v1 + v2 - v3 - v4 - n_dot_b * v5 - n_q_n * P_b_k);
    }

    // Transverse-shear term of d C^H / d g (variation of the constitutive projection).
    template<typename T>
    __device__ __forceinline__ CudaVec<T, 3> G_operator_k(
        const CudaMat<T, 3, 3>& sig, const CudaMat<T, 3, 3>& grad_a, const CudaVec<T, 3>& val_b,
        const CudaVec<T, 3>& n, const CudaMat<T, 3, 3>& P, const CudaMat<T, 3, 3>& Q, T norm_g, T t)
    {
        CudaMat<T, 3, 3> dir_grad_a = grad_a * P;

        CudaVec<T, 3> t1 = n.dot(sig * n) * (P * val_b) + (P * sig * dir_grad_a.transpose() * n) + (dir_grad_a.transpose() * Q * sig * n);

        return t1 * (t / norm_g);
    }

    // d C^H_IJ / d g (dC_dg_out) and d C^H_IJ / d Hess (dC_dH_out) at every quadrature point,
    // already multiplied by the quadrature weight. Output index: (element * 4 + point) * 36 + I * 6 + J.
    // One block per element (4 warps), one warp per quadrature point; global_V_local holds the
    // per-element 24 x 6 vectors y = g - x of the forward pass.
    template<typename T>
    __global__ void compute_homogenization_gradients_kernel(
        Element* elements, GaussianQuadSoA<T>* quad_soa, CudaVec<T, 3>* nodes,
        const CudaMat<T, 24, 6>* __restrict__ global_V_local,
        CudaVec<T, 3>* dC_dg_out,
        CudaMat<T, 3, 3>* dC_dH_out,
        const T E, const T mu, const T lambda, const T thickness,
        const int nActiveTets, SDFSampler sampler)
    {
        extern __shared__ char smem_pool[];

        int ele_idx = blockIdx.x;
        int warp_idx = threadIdx.x / 32;
        int lane_idx = threadIdx.x % 32;

        if (ele_idx >= nActiveTets) return;

        // six KinematicStates (one per cell problem) per warp
        KinematicState<T>* my_states = reinterpret_cast<KinematicState<T>*>(&smem_pool[warp_idx * 6 * sizeof(KinematicState<T>)]);

        int num_quad = quad_soa[ele_idx].count;
        if (warp_idx < num_quad)
        {
            T q_weight = quad_soa[ele_idx].weight[warp_idx];
            CudaVec<T, 3> pos(quad_soa[ele_idx].world_x[warp_idx], quad_soa[ele_idx].world_y[warp_idx], quad_soa[ele_idx].world_z[warp_idx]);
            CudaVec<T, 4> N_vec(quad_soa[ele_idx].shape_r[warp_idx], quad_soa[ele_idx].shape_s[warp_idx], quad_soa[ele_idx].shape_t[warp_idx], quad_soa[ele_idx].shape_w[warp_idx]);

            CudaMat<T, 3, 4> X;
            X.setCol(0, nodes[elements[ele_idx][0]]); X.setCol(1, nodes[elements[ele_idx][1]]);
            X.setCol(2, nodes[elements[ele_idx][2]]); X.setCol(3, nodes[elements[ele_idx][3]]);

            CudaMat<T, 4, 3> D(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, -1.0);
            CudaMat<T, 3, 3> XD_inv = (X * D).inverse();

            CudaMat<T, 3, 3> P_mat = RM_Mat::project_mat_P(pos, sampler);
            CudaMat<T, 3, 3> Q_mat = RM_Mat::project_mat_Q(pos, sampler);
            CudaMat<T, 3, 3> H_mat = RM_Mat::Weingarten_mat(pos, sampler);
            CudaMat<T, 3, 3> Hess = RM_Mat::d2F_dx2(pos, sampler);
            CudaVec<T, 3> normal = RM_Mat::dF_dx_norm(pos, sampler);
            T norm_g = RM_Mat::dF_dx(pos, sampler).norm();

            CudaMat<T, 24, 6> V_local = global_V_local[ele_idx];

            // Step 1: lanes 0..5 build the kinematic state of cell problem I = lane.
            if (lane_idx < 6)
            {
                int I = lane_idx;
                KinematicState<T> state;
                state.grad_u.setZero(); state.grad_w.setZero(); state.val_w.setZero();

                for (int b = 0; b < 12; ++b) {
                    state.grad_u += V_local(b, I) * RM_Mat::grad_operator(pos, D, XD_inv, b, sampler);
                }
                for (int b = 12; b < 24; ++b) {
                    T v_w = V_local(b, I);
                    state.grad_w += v_w * RM_Mat::grad_operator(pos, D, XD_inv, b % 12, sampler);
                    CudaMat<T, 3, 4> dW_dwb; dW_dwb.setZero(); dW_dwb((b % 12) % 3, (b % 12) / 3) = 1.0;
                    state.val_w += v_w * (dW_dwb * N_vec);
                }

                CudaMat<T, 3, 3> dir_grad_u = state.grad_u * P_mat;
                CudaMat<T, 3, 3> cov_grad_u = P_mat * state.grad_u * P_mat;

                RM_Mat::CudaMat_wrap_3<T> grad_P = RM_Mat::grad_P(pos, sampler);
                CudaMat<T, 3, 3> part1;
                part1.setCol(0, grad_P[0] * state.val_w);
                part1.setCol(1, grad_P[1] * state.val_w);
                part1.setCol(2, grad_P[2] * state.val_w);
                CudaMat<T, 3, 3> cov_grad_tilde_w = P_mat * state.grad_w * P_mat + P_mat * part1;

                CudaMat<T, 3, 3> strain_mem = 0.5 * (cov_grad_u + cov_grad_u.transpose());
                CudaMat<T, 3, 3> strain_bend = 0.5 * (H_mat * dir_grad_u + dir_grad_u.transpose() * H_mat + cov_grad_tilde_w + cov_grad_tilde_w.transpose());
                CudaMat<T, 3, 3> strain_shear = 0.5 * (Q_mat * dir_grad_u + dir_grad_u.transpose() * Q_mat + outer_product(normal, P_mat * state.val_w) + outer_product(P_mat * state.val_w, normal));

                state.n_mem = RM_Mat::stress_resultant_n_operator(pos, strain_mem, mu, lambda, thickness, sampler);
                state.m_bend = RM_Mat::stress_resultant_m_operator(pos, strain_bend, mu, lambda, thickness, sampler);
                state.q_shear = RM_Mat::stress_resultant_q_operator(pos, strain_shear, mu, lambda, thickness, sampler);
                state.sig_shear = RM_Mat::linear_stress_operator(strain_shear, mu, lambda);

                my_states[I] = state;
            }
            __syncwarp();

            // Step 2: the warp evaluates all 36 (I, J) pairs.
            for (int pair_idx = lane_idx; pair_idx < 36; pair_idx += 32)
            {
                int I = pair_idx / 6;
                int J = pair_idx % 6;

                const KinematicState<T>& s_I = my_states[I];
                const KinematicState<T>& s_J = my_states[J];

                // d C^H_IJ / d g
                CudaVec<T, 3> dC_dg_mem = (-1.0f / norm_g) * (Mem_operator(s_J.n_mem, s_I.grad_u, normal, P_mat) +
                                                             Mem_operator(s_I.n_mem, s_J.grad_u, normal, P_mat));

                CudaVec<T, 3> dC_dg_bend = D_operator_k(s_J.m_bend, s_I.grad_u, s_I.grad_w, s_I.val_w, normal, P_mat, H_mat, Hess, norm_g) +
                                D_operator_k(s_I.m_bend, s_J.grad_u, s_J.grad_w, s_J.val_w, normal, P_mat, H_mat, Hess, norm_g);

                CudaVec<T, 3> dC_dg_shear = F_operator_k(s_J.q_shear, s_I.grad_u, s_I.grad_w, s_I.val_w, normal, P_mat, Q_mat, norm_g) +
                                    F_operator_k(s_I.q_shear, s_J.grad_u, s_J.grad_w, s_J.val_w, normal, P_mat, Q_mat, norm_g) +
                                    G_operator_k(s_J.sig_shear, s_I.grad_u, s_I.val_w,
                                                normal, P_mat, Q_mat, norm_g, thickness);

                CudaVec<T, 3> dC_dg = (dC_dg_mem + dC_dg_bend + dC_dg_shear) * q_weight;

                // d C^H_IJ / d Hess (symmetrized)
                CudaMat<T, 3, 3> cov_grad_I_u = P_mat * s_I.grad_u * P_mat;
                CudaMat<T, 3, 3> cov_grad_J_u = P_mat * s_J.grad_u * P_mat;

                CudaMat<T, 3, 3> term1_H = s_J.m_bend * cov_grad_I_u.transpose();
                CudaMat<T, 3, 3> term2_H = normal.dot(s_I.val_w) * s_J.m_bend;
                CudaMat<T, 3, 3> term3_H = s_I.m_bend * cov_grad_J_u.transpose();
                CudaMat<T, 3, 3> term4_H = normal.dot(s_J.val_w) * s_I.m_bend;

                CudaMat<T, 3, 3> dC_dH = (1.0 / norm_g) * (term1_H - term2_H + term3_H - term4_H);
                dC_dH = q_weight * 0.5 * (dC_dH + dC_dH.transpose());

                int out_idx = ((ele_idx * 4) + warp_idx) * 36 + pair_idx;
                dC_dg_out[out_idx] = dC_dg;
                dC_dH_out[out_idx] = dC_dH;
            }
        }
    }
}
