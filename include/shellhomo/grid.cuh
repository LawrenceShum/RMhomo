#pragma once
// Background tetrahedral grid on the unit cube: vertex relaxation away from the implicit
// surface, cut-element detection, per-element surface quadrature, and periodic DOF indexing.

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include "linalg.cuh"
#include "sdf.cuh"

// ---------------------------------------------------------------------------
// Surface quadrature on cut tetrahedra
// ---------------------------------------------------------------------------

#define MAX_QUAD_POINTS 4

// Quadrature points of one cut element. A cut triangle uses 3 points (weight[3] = 0),
// a cut quadrilateral uses 2x2 Gauss points.
template <typename T>
struct GaussianQuadSoA {
    int count;
    // local (polygon) coordinates
    T local_u[MAX_QUAD_POINTS];
    T local_v[MAX_QUAD_POINTS];
    // world coordinates
    T world_x[MAX_QUAD_POINTS];
    T world_y[MAX_QUAD_POINTS];
    T world_z[MAX_QUAD_POINTS];
    // barycentric (linear tet shape function) values
    T shape_r[MAX_QUAD_POINTS];
    T shape_s[MAX_QUAD_POINTS];
    T shape_t[MAX_QUAD_POINTS];
    T shape_w[MAX_QUAD_POINTS];
    T weight[MAX_QUAD_POINTS];
    __device__ __host__ GaussianQuadSoA() : count(0) {}
    __device__ __host__ void clear() {
        count = 0;
        #pragma unroll
        for(int i=0; i<MAX_QUAD_POINTS; ++i) {
            weight[i] = 0;
        }
    }
};

namespace GaussianQuadrature {

// Zero crossing of the implicit field on segment [origin, end]: linear initial guess
// followed by clamped Newton iterations.
template <typename T>
__device__ __forceinline__ CudaVec<T, 3> find_intersection(
    CudaVec<T, 3> origin,
    CudaVec<T, 3> end,
    SDFSampler& sampler)
{
    CudaVec<T, 3> direction = end - origin;
    T l = direction.norm();

    if (l < 1e-8) return origin;

    direction = direction / l;

    T sdf_origin = TextureBuffer::shell_sdf(origin, sampler);
    T sdf_end    = TextureBuffer::shell_sdf(end,    sampler);

    T t_hit = 0.0;
    if (sdf_origin * sdf_end < 0.0) {
        t_hit = l * abs(sdf_origin) / (abs(sdf_origin) + abs(sdf_end));
    } else {
            t_hit = 0.0;
    }

    t_hit = clamp(t_hit, (T)0.0, l);

    #pragma unroll
    for (int i = 0; i < 20; i++)
    {
        CudaVec<T, 3> p_hit = origin + direction * t_hit;
        T g_t = TextureBuffer::shell_sdf(p_hit, sampler);

        if (abs(g_t) < 1e-6)
            return p_hit;

        CudaVec<T, 3> grad_f = TextureBuffer::shell_grad(p_hit, sampler);
        T g_prime_t = grad_f.dot(direction);

        T delta_t;
        if (abs(g_prime_t) < 1e-6) {
                delta_t = -g_t;
        } else {
                delta_t = g_t / g_prime_t;
        }

        T t_next = t_hit - delta_t;

        if (t_next < 0.0) {
            t_next = 0.0;
            if (abs(t_next - t_hit) < 1e-6) break;
        } else if (t_next > l) {
            t_next = l;
            if (abs(t_next - t_hit) < 1e-6) break;
        }

        t_hit = t_next;
    }

    return origin + direction * t_hit;
}

// Barycentric coordinates of p w.r.t. the tet (entry i belongs to vertex i).
template <typename T>
__device__ __forceinline__ CudaVec<T, 4> barycentric_coord_tet(
    CudaVec<T, 3> p,
    const CudaVec<T, 3>* tet_nodes)
{
    CudaVec<T, 3> T1 = tet_nodes[0];
    CudaVec<T, 3> T2 = tet_nodes[1];
    CudaVec<T, 3> T3 = tet_nodes[2];
    CudaVec<T, 3> T4 = tet_nodes[3];
    CudaVec<T, 3> V1 = T2 - T1;
    CudaVec<T, 3> V2 = T3 - T1;
    CudaVec<T, 3> V3 = T4 - T1;
    CudaVec<T, 3> Vpos = p - T1;
    const T det_A = abs(V1.dot(V2.cross(V3)));
    const T epsilon = 1e-12;
    CudaVec<T, 4> result;
    if (std::abs(det_A) < epsilon)
    {
        result[0] = -1;
        result[1] = -1;
        result[2] = -1;
        result[3] = -1;
        return result;
    }
    const T det_s = abs(Vpos.dot(V2.cross(V3)));
    const T det_t = abs(V1.dot(Vpos.cross(V3)));
    const T det_w = abs(V1.dot(V2.cross(Vpos)));
    result[1] = det_s / det_A;
    result[2] = det_t / det_A;
    result[3] = det_w / det_A;
    result[0] = 1.0 - result[1] - result[2] - result[3];
    return result;
}

// Intersects the zero level set with a tet. Returns the number of polygon vertices
// written to cut_plane (3: triangle, 4: quadrilateral in cyclic order, 0: not cut).
template <typename T>
__device__ int cut_element(
    const CudaVec<T, 3>* tet_vertices,
    CudaVec<T, 3>* cut_plane,
    SDFSampler& sampler)
{
    int pos_nodes[4];
    int neg_nodes[4];
    int n_pos = 0;
    int n_neg = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (TextureBuffer::shell_sdf(tet_vertices[i], sampler) > 0)
            pos_nodes[n_pos++] = i;
        else
            neg_nodes[n_neg++] = i;
    }

    // Triangle: one vertex is separated from the other three.
    if (n_pos == 1 || n_neg == 1) {
        int* single = (n_pos == 1) ? pos_nodes : neg_nodes;
        int* triple = (n_pos == 1) ? neg_nodes : pos_nodes;

        int v_single = single[0];
        CudaVec<T, 3> P_s = tet_vertices[v_single];

        cut_plane[0] = find_intersection(P_s, tet_vertices[triple[0]], sampler);
        cut_plane[1] = find_intersection(P_s, tet_vertices[triple[1]], sampler);
        cut_plane[2] = find_intersection(P_s, tet_vertices[triple[2]], sampler);
        return 3;
    }
    // Quadrilateral: edges n0->p0, n0->p1, n1->p1, n1->p0 (cyclic order).
    else if (n_pos == 2 && n_neg == 2) {
        int n0 = neg_nodes[0];
        int n1 = neg_nodes[1];
        int p0 = pos_nodes[0];
        int p1 = pos_nodes[1];

        CudaVec<T, 3> P_n0 = tet_vertices[n0];
        CudaVec<T, 3> P_n1 = tet_vertices[n1];

        cut_plane[0] = find_intersection(P_n0, tet_vertices[p0], sampler);
        cut_plane[1] = find_intersection(P_n0, tet_vertices[p1], sampler);
        cut_plane[2] = find_intersection(P_n1, tet_vertices[p1], sampler);
        cut_plane[3] = find_intersection(P_n1, tet_vertices[p0], sampler);
        return 4;
    }

    return 0;
}

template <typename T>
__device__ void get_gaussian_quadratures_soa(
    const CudaVec<T, 3>* tet_nodes,
    GaussianQuadSoA<T>& out_soa,
    SDFSampler& sampler)
{
    CudaVec<T, 3> cut_plane[4];
    int num_cut = cut_element(tet_nodes, cut_plane, sampler);

    out_soa.count = num_cut;
    if (num_cut == 0) return;

    CudaVec<T, 4> cut_bary[4];
    for(int i=0; i<num_cut; ++i) {
        cut_bary[i] = barycentric_coord_tet(cut_plane[i], tet_nodes);
    }

    if (num_cut == 3) {
        // 3-point rule on the cut triangle (exact for quadratics)
        T w_const = 1.0 / 6.0;
        const T u_const[3] = {1.0/6.0, 2.0/3.0, 1.0/6.0};
        const T v_const[3] = {1.0/6.0, 1.0/6.0, 2.0/3.0};

        CudaVec<T, 3> dPdu = cut_plane[0] - cut_plane[2];
        CudaVec<T, 3> dPdv = cut_plane[1] - cut_plane[2];

        CudaVec<T, 3> cross_prod;
        cross_prod[0] = dPdu[1]*dPdv[2] - dPdu[2]*dPdv[1];
        cross_prod[1] = dPdu[2]*dPdv[0] - dPdu[0]*dPdv[2];
        cross_prod[2] = dPdu[0]*dPdv[1] - dPdu[1]*dPdv[0];

        T area_scale = cross_prod.norm();

        #pragma unroll
        for(int i=0; i<3; ++i) {
            T u = u_const[i];
            T v = v_const[i];

            out_soa.local_u[i] = u;
            out_soa.local_v[i] = v;

            T w = 1.0 - u - v;
            CudaVec<T, 4> b = cut_bary[0]*u + cut_bary[1]*v + cut_bary[2]*w;

            CudaVec<T, 3> w_pos = tet_nodes[0]*b[0] + tet_nodes[1]*b[1] + tet_nodes[2]*b[2] + tet_nodes[3]*b[3];

            out_soa.world_x[i] = w_pos[0];
            out_soa.world_y[i] = w_pos[1];
            out_soa.world_z[i] = w_pos[2];

            out_soa.weight[i] = w_const * area_scale;
            CudaVec<T, 4> N = barycentric_coord_tet(w_pos, tet_nodes);
            out_soa.shape_r[i] = N[0];
            out_soa.shape_s[i] = N[1];
            out_soa.shape_t[i] = N[2];
            out_soa.shape_w[i] = N[3];
        }

        out_soa.weight[3] = 0;
    }
    else {
        // 2x2 Gauss rule on the bilinearly parameterized cut quadrilateral
        T inv_sqrt3 = 1.0 / sqrt(3.0);
        const T u_const[4] = {-inv_sqrt3, inv_sqrt3, inv_sqrt3, -inv_sqrt3};
        const T v_const[4] = {-inv_sqrt3, -inv_sqrt3, inv_sqrt3, inv_sqrt3};

        #pragma unroll
        for(int i=0; i<4; ++i) {
            T u = u_const[i];
            T v = v_const[i];

            out_soa.local_u[i] = u;
            out_soa.local_v[i] = v;

            T n[4];
            n[0] = 0.25*(1-u)*(1-v); n[1] = 0.25*(1+u)*(1-v);
            n[2] = 0.25*(1+u)*(1+v); n[3] = 0.25*(1-u)*(1+v);

            CudaVec<T, 4> b = cut_bary[0]*n[0] + cut_bary[1]*n[1] + cut_bary[2]*n[2] + cut_bary[3]*n[3];
            CudaVec<T, 3> w_pos = tet_nodes[0]*b[0] + tet_nodes[1]*b[1] + tet_nodes[2]*b[2] + tet_nodes[3]*b[3];

            out_soa.world_x[i] = w_pos[0];
            out_soa.world_y[i] = w_pos[1];
            out_soa.world_z[i] = w_pos[2];
            CudaVec<T, 4> N = barycentric_coord_tet(w_pos, tet_nodes);
            out_soa.shape_r[i] = N[0];
            out_soa.shape_s[i] = N[1];
            out_soa.shape_t[i] = N[2];
            out_soa.shape_w[i] = N[3];

            T dn_du[4] = { -T(0.25)*(1-v),  T(0.25)*(1-v), T(0.25)*(1+v), -T(0.25)*(1+v) };
            T dn_dv[4] = { -T(0.25)*(1-u), -T(0.25)*(1+u), T(0.25)*(1+u),  T(0.25)*(1-u) };

            CudaVec<T, 3> dPdu(0,0,0);
            CudaVec<T, 3> dPdv(0,0,0);

            for(int k=0; k<4; ++k) {
                dPdu = dPdu + cut_plane[k] * dn_du[k];
                dPdv = dPdv + cut_plane[k] * dn_dv[k];
            }

            CudaVec<T, 3> cross_prod;
            cross_prod[0] = dPdu[1]*dPdv[2] - dPdu[2]*dPdv[1];
            cross_prod[1] = dPdu[2]*dPdv[0] - dPdu[0]*dPdv[2];
            cross_prod[2] = dPdu[0]*dPdv[1] - dPdu[1]*dPdv[0];

            // Gauss weights are 1 on [-1,1]^2
            out_soa.weight[i] = 1.0 * cross_prod.norm();
        }
    }
}
} // namespace GaussianQuadrature

// ---------------------------------------------------------------------------
// Grid relaxation: push vertices out of a narrow band around the surface
// (log-barrier force along the field gradient) with Laplacian smoothing.
// Red-black Gauss-Seidel; boundary vertices stay on their faces.
// ---------------------------------------------------------------------------

#define SUB_BLOCK_SIZE 8

struct SimulationParams {
    int res;
    float h;
    float alpha;
    float spring_k;
    float dt;
    float bounds[6];
};

__global__ void relax_kernel(CudaVec<float, 3>* points, SimulationParams params, int color_phase, SDFSampler sampler) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= params.res || j >= params.res || k >= params.res) return;

    if ((i + j + k) % 2 != color_phase) return;

    int idx = i + params.res * (j + params.res * k);
    float3 p = make_float3(points[idx][0], points[idx][1], points[idx][2]);

    float phi = TextureBuffer::shell_sdf(CudaVec<float, 3>(p.x, p.y, p.z), sampler);

    auto get_phi = [&](int ix, int iy, int iz) {
        if (ix < 0 || ix >= params.res || iy < 0 || iy >= params.res || iz < 0 || iz >= params.res)
            return phi;
        int n_idx = ix + params.res * (iy + params.res * iz);
        float3 np = make_float3(points[n_idx][0], points[n_idx][1], points[n_idx][2]);
        return TextureBuffer::shell_sdf(CudaVec<float, 3>(np.x, np.y, np.z), sampler);
    };

    float grad_x = (get_phi(i + 1, j, k) - get_phi(i - 1, j, k)) * 0.5f;
    float grad_y = (get_phi(i, j + 1, k) - get_phi(i, j - 1, k)) * 0.5f;
    float grad_z = (get_phi(i, j, k + 1) - get_phi(i, j, k - 1)) * 0.5f;

    float3 normal = make_float3(grad_x, grad_y, grad_z);
    float norm_sq = normal.x*normal.x + normal.y*normal.y + normal.z*normal.z;
    if (norm_sq > 1e-12f) {
        float inv = rsqrtf(norm_sq);
        normal.x *= inv; normal.y *= inv; normal.z *= inv;
    }

    // Repulsion from the surface inside the band |phi| < 1.5h
    bool mask_active = fabsf(phi) < (params.h * 1.5f);

    float3 f_penalty = make_float3(0.0f, 0.0f, 0.0f);
    if (mask_active) {
        float ratio = fabsf(phi) / params.h + 1e-6f;
        if (ratio < 0.01f) ratio = 0.01f;
        if (ratio > 1.0f) ratio = 1.0f;
        float mag = -logf(ratio);
        float sign = (phi > 0.0f) ? 1.0f : -1.0f;
        f_penalty = make_float3(params.alpha * mag * sign * normal.x,
                                params.alpha * mag * sign * normal.y,
                                params.alpha * mag * sign * normal.z);
    }

    // Graph Laplacian; a missing neighbour on a boundary face is replaced by the mirrored one.
    float3 laplacian = make_float3(0.0f, 0.0f, 0.0f);

    // X axis
    float3 px_p = (i < params.res - 1) ? make_float3(points[idx + 1][0], points[idx + 1][1], points[idx + 1][2]) : make_float3(0.0f,0.0f,0.0f);
    float3 px_m = (i > 0)              ? make_float3(points[idx - 1][0], points[idx - 1][1], points[idx - 1][2]) : make_float3(0.0f,0.0f,0.0f);

    float3 dx_fwd = (i < params.res - 1) ? make_float3(px_p.x-p.x, px_p.y-p.y, px_p.z-p.z) : make_float3(0.0f,0.0f,0.0f);
    float3 dx_bwd = (i > 0)              ? make_float3(p.x-px_m.x, p.y-px_m.y, p.z-px_m.z) : make_float3(0.0f,0.0f,0.0f);

    if (i == 0) dx_bwd = make_float3(-dx_fwd.x, -dx_fwd.y, -dx_fwd.z);
    if (i == params.res - 1) dx_fwd = make_float3(-dx_bwd.x, -dx_bwd.y, -dx_bwd.z);

    laplacian.x += dx_fwd.x - dx_bwd.x;
    laplacian.y += dx_fwd.y - dx_bwd.y;
    laplacian.z += dx_fwd.z - dx_bwd.z;

    // Y axis
    float3 py_p = (j < params.res - 1) ? make_float3(points[idx + params.res][0], points[idx + params.res][1], points[idx + params.res][2]) : make_float3(0.0f,0.0f,0.0f);
    float3 py_m = (j > 0)              ? make_float3(points[idx - params.res][0], points[idx - params.res][1], points[idx - params.res][2]) : make_float3(0.0f,0.0f,0.0f);
    float3 dy_fwd = (j < params.res - 1) ? make_float3(py_p.x-p.x, py_p.y-p.y, py_p.z-p.z) : make_float3(0.0f,0.0f,0.0f);
    float3 dy_bwd = (j > 0)              ? make_float3(p.x-py_m.x, p.y-py_m.y, p.z-py_m.z) : make_float3(0.0f,0.0f,0.0f);
    if (j == 0) dy_bwd = make_float3(-dy_fwd.x, -dy_fwd.y, -dy_fwd.z);
    if (j == params.res - 1) dy_fwd = make_float3(-dy_bwd.x, -dy_bwd.y, -dy_bwd.z);
    laplacian.x += dy_fwd.x - dy_bwd.x;
    laplacian.y += dy_fwd.y - dy_bwd.y;
    laplacian.z += dy_fwd.z - dy_bwd.z;

    // Z axis
    int stride = params.res * params.res;
    float3 pz_p = (k < params.res - 1) ? make_float3(points[idx + stride][0], points[idx + stride][1], points[idx + stride][2]) : make_float3(0.0f,0.0f,0.0f);
    float3 pz_m = (k > 0)              ? make_float3(points[idx - stride][0], points[idx - stride][1], points[idx - stride][2]) : make_float3(0.0f,0.0f,0.0f);
    float3 dz_fwd = (k < params.res - 1) ? make_float3(pz_p.x-p.x, pz_p.y-p.y, pz_p.z-p.z) : make_float3(0.0f,0.0f,0.0f);
    float3 dz_bwd = (k > 0)              ? make_float3(p.x-pz_m.x, p.y-pz_m.y, p.z-pz_m.z) : make_float3(0.0f,0.0f,0.0f);
    if (k == 0) dz_bwd = make_float3(-dz_fwd.x, -dz_fwd.y, -dz_fwd.z);
    if (k == params.res - 1) dz_fwd = make_float3(-dz_bwd.x, -dz_bwd.y, -dz_bwd.z);
    laplacian.x += dz_fwd.x - dz_bwd.x;
    laplacian.y += dz_fwd.y - dz_bwd.y;
    laplacian.z += dz_fwd.z - dz_bwd.z;

    float3 total = make_float3(params.spring_k * laplacian.x + f_penalty.x,
                               params.spring_k * laplacian.y + f_penalty.y,
                               params.spring_k * laplacian.z + f_penalty.z);

    // Boundary vertices only move within their face.
    if (i == 0 || i == params.res - 1) total.x = 0.0f;
    if (j == 0 || j == params.res - 1) total.y = 0.0f;
    if (k == 0 || k == params.res - 1) total.z = 0.0f;

    p.x += total.x * params.dt;
    p.y += total.y * params.dt;
    p.z += total.z * params.dt;

    if (i == 0) p.x = params.bounds[0];
    if (i == params.res - 1) p.x = params.bounds[1];
    if (j == 0) p.y = params.bounds[2];
    if (j == params.res - 1) p.y = params.bounds[3];
    if (k == 0) p.z = params.bounds[4];
    if (k == params.res - 1) p.z = params.bounds[5];

    points[idx][0] = p.x;
    points[idx][1] = p.y;
    points[idx][2] = p.z;
}

// 40 red-black sweeps (h = 0.2, alpha = 0.1, spring_k = 1.5, dt = 0.05), launched as one CUDA graph.
inline void relax_grid_implementation(CudaVec<float, 3>* points, int nRes, SDFSampler& sampler)
{
    SimulationParams params;
    params.res = nRes;
    params.h = 0.2f;
    params.alpha = 0.1f;
    params.spring_k = 1.5f;
    params.dt = 0.05f;
    float bounds_h[6] = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    memcpy(params.bounds, bounds_h, 6 * sizeof(float));
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dim3 block(SUB_BLOCK_SIZE, SUB_BLOCK_SIZE, SUB_BLOCK_SIZE);
    dim3 grid((params.res + SUB_BLOCK_SIZE - 1) / SUB_BLOCK_SIZE,
              (params.res + SUB_BLOCK_SIZE - 1) / SUB_BLOCK_SIZE,
              (params.res + SUB_BLOCK_SIZE - 1) / SUB_BLOCK_SIZE);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    int iterations = 40;
    for (int i = 0; i < iterations; ++i) {
        relax_kernel<<<grid, block, 0, stream>>>(points, params, 0, sampler); // red
        relax_kernel<<<grid, block, 0, stream>>>(points, params, 1, sampler); // black
    }
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    cudaGraphLaunch(instance, stream);

    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// Tetrahedral background grid
// ---------------------------------------------------------------------------

struct Element {
    int v[4];
    __host__ __device__ int& operator[](int i) { return v[i]; }
    __host__ __device__ const int& operator[](int i) const { return v[i]; }
};

__device__ inline int get_node_idx(int x, int y, int z, int nX, int nY) {
    return x + y * (nX + 1) + z * (nX + 1) * (nY + 1);
}

// Writes the (periodic) master node id of every vertex of the active elements.
__global__ void flatten_active_elements_with_pbc_kernel(
    Element* active_tets,
    int nActiveTets,
    int* pbc_map,
    int* flat_indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nActiveTets) return;

    Element t = active_tets[idx];

    for(int k=0; k<4; ++k) {
        int original_node = t[k];
        int master_node = pbc_map[original_node];

        flat_indices[idx * 4 + k] = master_node;
    }
}

// Periodic identification: nodes on the faces x = 1, y = 1, z = 1 map to their partners on x/y/z = 0.
__global__ void build_pbc_node_map_kernel(
    int* pbc_map,
    int total_nodes,
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_nodes) return;

    int stride_y = nx + 1;
    int stride_z = (nx + 1) * (ny + 1);

    int iz = idx / stride_z;
    int rem = idx % stride_z;
    int iy = rem / stride_y;
    int ix = rem % stride_y;

    int map_ix = (ix == nx) ? 0 : ix;
    int map_iy = (iy == ny) ? 0 : iy;
    int map_iz = (iz == nz) ? 0 : iz;

    int master_id = map_ix + map_iy * stride_y + map_iz * stride_z;

    pbc_map[idx] = master_id;
}

template<typename T>
__global__ void init_nodes_kernel(CudaVec<T, 3>* nodes, int nX, int nY, int nZ, T dx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_nodes = (nX + 1) * (nY + 1) * (nZ + 1);
    if (idx >= total_nodes) return;

    int stride_y = nX + 1;
    int stride_z = (nX + 1) * (nY + 1);

    int k = idx / stride_z;
    int rem = idx % stride_z;
    int j = rem / stride_y;
    int i = rem % stride_y;

    nodes[idx] = CudaVec<T, 3>((T)i * dx, (T)j * dx, (T)k * dx);
}

// Note: `indices` is not used by the kernel.
template<typename T>
__global__ void quadrature_per_element_kernel(Element* tets, CudaVec<T, 3>* nodes, int* indices, GaussianQuadSoA<T>* quad, int nTets, SDFSampler sampler)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nTets) return;
    Element tet = tets[idx];
    CudaVec<T, 3> tet_coord[4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        tet_coord[i] = nodes[tet[i]];
    GaussianQuadSoA<T> soa;
    GaussianQuadrature::get_gaussian_quadratures_soa(tet_coord, soa, sampler);
    quad[idx] = soa;
}

// is_cut != 0 iff the field changes sign over the tet vertices.
template<typename T>
__global__ void check_is_cut_kernel(Element* tets, int* is_cut, CudaVec<T, 3>* nodes, int nTets, SDFSampler sampler)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nTets)
        return;
    CudaVec<T, 3> tet_coord[4];
    int neg_num = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        tet_coord[i] = nodes[tets[idx][i]];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        neg_num += TextureBuffer::shell_sdf(tet_coord[i], sampler) < 0 ? 1 : 0;
    is_cut[idx] = neg_num % 4;
}

// Five tets per cube cell, alternating orientation in a checkerboard pattern
// (the grid resolution must be even for a conforming periodic mesh).
__global__ void init_tets_kernel(Element* tets, int nX, int nY, int nZ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = nX * nY * nZ;
    if (idx >= total_cells) return;

    int stride_y = nX;
    int stride_z = nX * nY;

    int k = idx / stride_z;
    int rem = idx % stride_z;
    int j = rem / stride_y;
    int i = rem % stride_y;

    int v000 = get_node_idx(i, j, k, nX, nY);
    int v100 = get_node_idx(i + 1, j, k, nX, nY);
    int v101 = get_node_idx(i + 1, j, k + 1, nX, nY);
    int v001 = get_node_idx(i, j, k + 1, nX, nY);
    int v010 = get_node_idx(i, j + 1, k, nX, nY);
    int v110 = get_node_idx(i + 1, j + 1, k, nX, nY);
    int v111 = get_node_idx(i + 1, j + 1, k + 1, nX, nY);
    int v011 = get_node_idx(i, j + 1, k + 1, nX, nY);

    int base_tet_idx = idx * 5;

    bool flag = (i + j + k) % 2 == 0;

    if (flag) {
        tets[base_tet_idx + 0] = {v010, v100, v111, v001};
        tets[base_tet_idx + 1] = {v000, v010, v100, v001};
        tets[base_tet_idx + 2] = {v100, v101, v111, v001};
        tets[base_tet_idx + 3] = {v110, v100, v010, v111};
        tets[base_tet_idx + 4] = {v011, v010, v111, v001};
    } else {
        tets[base_tet_idx + 0] = {v010, v000, v011, v110};
        tets[base_tet_idx + 1] = {v100, v000, v110, v101};
        tets[base_tet_idx + 2] = {v111, v011, v110, v101};
        tets[base_tet_idx + 3] = {v001, v000, v101, v011};
        tets[base_tet_idx + 4] = {v000, v101, v110, v011};
    }
}

__global__ void copy_to_active_elements(Element* in, Element* out, int* indices, const int nActiveTets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nActiveTets)
        return;
    out[idx] = in[indices[idx]];
}

template<typename T>
class TetsGrid
{
public:
    using Vec3 = CudaVec<T, 3>;

    TetsGrid(int nx, int ny, int nz) : nX(nx), nY(ny), nZ(nz)
    {
        int n_cells = nX * nY * nZ;
        nTets = 5 * n_cells;
        int nNodes = (nX + 1) * (nY + 1) * (nZ + 1);

        d_tets.resize(nTets);
        d_nodes.resize(nNodes);
        d_nodes_origin.resize(nNodes);
        T dx = 1.0 / (T)nX;

        int blockSize = 256;
        int gridSizeNodes = (nNodes + blockSize - 1) / blockSize;
        init_nodes_kernel<<<gridSizeNodes, blockSize>>>(
            thrust::raw_pointer_cast(d_nodes_origin.data()),
            nX, nY, nZ, dx
        );
        getLastCudaError("init_nodes_kernel");
        int gridSizeCells = (n_cells + blockSize - 1) / blockSize;
        init_tets_kernel<<<gridSizeCells, blockSize>>>(
            thrust::raw_pointer_cast(d_tets.data()),
            nX, nY, nZ
        );

        getLastCudaError("init_tets_kernel");
        checkCudaErrors(cudaDeviceSynchronize());
    }

    ~TetsGrid() = default;

    int get_nActive_tets() const { return nCutTets; }
    int get_nActive_nodes() const { return nActiveNodes; }
    int get_fixed_node_x() const { return fixed_node_idx_x; }
    int get_fixed_node_y() const { return fixed_node_idx_y; }

    Vec3* get_nodes_ptr() { return thrust::raw_pointer_cast(d_nodes.data()); }
    GaussianQuadSoA<T>* get_quads_ptr() { return thrust::raw_pointer_cast(d_quads.data()); }
    Element* get_active_tets_ptr() { return thrust::raw_pointer_cast(d_active_tets.data()); }
    int *get_compact_node_indices_ptr() { return thrust::raw_pointer_cast(d_compact_indices.data()); }

    struct is_cut {
        __device__ __host__ bool operator()(const thrust::tuple<int, int>& t) {
            int status = thrust::get<0>(t);
            return status != 0;
        }
    };

    // Re-creates the tets of the regular grid.
    void refresh_grid()
    {
        d_active_tets.clear();
        d_quads.clear();
        d_indices.clear();
        d_compact_indices.clear();
        fixed_node_idx_x = -1;
        fixed_node_idx_y = -1;
        nActiveNodes = 0;
        nCutTets = 0;

        int n_cells = nX * nY * nZ;
        int blockSize = 256;
        int gridSizeCells = (n_cells + blockSize - 1) / blockSize;

        init_tets_kernel<<<gridSizeCells, blockSize>>>(
            thrust::raw_pointer_cast(d_tets.data()),
            nX, nY, nZ
        );
        getLastCudaError("refresh_grid: reset_tets_kernel");
    }

    // Resets the vertices to the regular grid and relaxes them w.r.t. the current field.
    void optimize_grid(SDFSampler& sampler)
    {
        thrust::copy(d_nodes_origin.begin(), d_nodes_origin.end(), d_nodes.begin());
        relax_grid_implementation(thrust::raw_pointer_cast(d_nodes.data()), nX + 1, sampler);
    }

    // Keeps only the tets cut by the zero level set (the active elements).
    void filter_uncut_elements(SDFSampler& sampler)
    {
        thrust::device_vector<int> d_status(nTets);
        int blockSize = 256;
        int gridSize = (nTets + blockSize - 1) / blockSize;
        check_is_cut_kernel<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_tets.data()),
            thrust::raw_pointer_cast(d_status.data()),
            thrust::raw_pointer_cast(d_nodes.data()),
            nTets, sampler
        );
        getLastCudaError("check_is_cut_kernel");
        checkCudaErrors(cudaDeviceSynchronize());
        thrust::device_vector<int> temp_indices(nTets);
        thrust::sequence(temp_indices.begin(), temp_indices.end());
        thrust::device_vector<int> temp_status(nTets);
        d_indices.resize(nTets);
        auto zip_in_begin = thrust::make_zip_iterator(thrust::make_tuple(d_status.begin(), temp_indices.begin()));
        auto zip_in_end = thrust::make_zip_iterator(thrust::make_tuple(d_status.end(), temp_indices.end()));
        auto zip_out_begin = thrust::make_zip_iterator(thrust::make_tuple(temp_status.begin(), d_indices.begin()));
        auto zip_out_end = thrust::copy_if(zip_in_begin, zip_in_end, zip_out_begin, is_cut());
        nCutTets = zip_out_end - zip_out_begin;
        d_indices.resize(nCutTets);
        std::cout << "num of cut elements = " << nCutTets << std::endl;
        d_active_tets.resize(nCutTets);
        d_quads.resize(nCutTets);
        blockSize = 256;
        gridSize = (nCutTets + blockSize - 1) / blockSize;
        copy_to_active_elements<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_tets.data()),
            thrust::raw_pointer_cast(d_active_tets.data()),
            thrust::raw_pointer_cast(d_indices.data()),
            nCutTets);
        getLastCudaError("copy_to_active_elements");
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void compute_quadrature_per_element(SDFSampler& sampler)
    {
        int blockSize = 256;
        int gridSize = (nCutTets + blockSize - 1) / blockSize;
        quadrature_per_element_kernel<<<gridSize, blockSize>>>(
            thrust::raw_pointer_cast(d_active_tets.data()),
            thrust::raw_pointer_cast(d_nodes.data()),
            thrust::raw_pointer_cast(d_indices.data()),
            thrust::raw_pointer_cast(d_quads.data()),
            nCutTets, sampler
        );
        getLastCudaError("compute_quadrature_per_element_kernel");
    }

    // Assigns compact DOF indices to the (periodically identified) vertices of the active
    // elements, in order of first appearance. Also picks the vertices used to remove the
    // rigid-translation null space: the first new vertex with ix == 0 and the first with iy == 0.
    void re_index()
    {
        int total_nodes = d_nodes.size();
        int nActiveTets = d_active_tets.size();
        thrust::device_vector<int> d_pbc_map(total_nodes);
        int blockSize = 256;
        int gridSizeNodes = (total_nodes + blockSize - 1) / blockSize;
        build_pbc_node_map_kernel<<<gridSizeNodes, blockSize>>>(
            thrust::raw_pointer_cast(d_pbc_map.data()),
            total_nodes, nX, nY, nZ
        );
        thrust::device_vector<int> d_flattened_nodes(nActiveTets * 4);
        int gridSizeTets = (nActiveTets + blockSize - 1) / blockSize;
        flatten_active_elements_with_pbc_kernel<<<gridSizeTets, blockSize>>>(
            thrust::raw_pointer_cast(d_active_tets.data()),
            nActiveTets,
            thrust::raw_pointer_cast(d_pbc_map.data()),
            thrust::raw_pointer_cast(d_flattened_nodes.data())
        );
        getLastCudaError("flatten_active_elements_with_pbc_kernel");

        thrust::host_vector<int> h_flattened_nodes = d_flattened_nodes;
        thrust::host_vector<int> h_compact_indices(nActiveTets * 4);
        std::vector<int> global_to_compact_map(total_nodes, -1);

        int next_compact_id = 0;
        fixed_node_idx_x = -1;
        fixed_node_idx_y = -1;
        int stride_y = nX + 1;
        int stride_z = (nX + 1) * (nY + 1);

        for (int i = 0; i < nActiveTets * 4; ++i) {
            int master_id = h_flattened_nodes[i];

            if (global_to_compact_map[master_id] == -1) {
                int current_compact_index = next_compact_id++;
                global_to_compact_map[master_id] = current_compact_index;

                int rem = master_id % stride_z;
                int iy = rem / stride_y;
                int ix = rem % stride_y;
                if (fixed_node_idx_x == -1 && ix == 0) {
                    fixed_node_idx_x = current_compact_index;
                }
                if (fixed_node_idx_y == -1 && iy == 0) {
                    fixed_node_idx_y = current_compact_index;
                }
            }

            h_compact_indices[i] = global_to_compact_map[master_id];
        }

        nActiveNodes = next_compact_id;

        d_compact_indices = h_compact_indices;
        checkCudaErrors(cudaDeviceSynchronize());
        std::cout << "fixed node at x = 0 idx = " << fixed_node_idx_x << std::endl;
        std::cout << "fixed node at y = 0 idx = " << fixed_node_idx_y << std::endl;
    }

private:
    int nX, nY, nZ;
    int nTets;
    int nCutTets;
    int nActiveNodes;
    thrust::device_vector<Element> d_tets;
    thrust::device_vector<Element> d_active_tets;
    thrust::device_vector<Vec3> d_nodes;
    thrust::device_vector<Vec3> d_nodes_origin;
    thrust::device_vector<GaussianQuadSoA<T>> d_quads;
    thrust::device_vector<int> d_indices;
    thrust::device_vector<int> d_compact_indices;
    int fixed_node_idx_x = -1; // compact index of the fixed vertex on x = 0
    int fixed_node_idx_y = -1; // compact index of the fixed vertex on y = 0
};
