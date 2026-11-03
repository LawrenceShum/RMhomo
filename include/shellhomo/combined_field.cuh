#pragma once
// Design space of the inverse-design example: a linear combination of six TPMS basis
// functions on the unit cube,
//   f(x) = sum_k c_k f_k(x),
//   f1 = cos X + cos Y + cos Z                     (Schwarz P)
//   f2 = sin X sin Y + sin Y sin Z + sin Z sin X
//   f3 = cos X cos Y + cos Y cos Z + cos Z cos X
//   f4 = sin X cos Y + sin Y cos Z + sin Z cos X   (gyroid)
//   f5 = sin X sin Y cos Z + sin Y sin Z cos X + sin Z sin X cos Y
//   f6 = cos X cos Y cos Z
// with X = 2 pi x, etc. Values, gradients and Hessians are analytic; the gradients and
// Hessians are used by the sensitivity pass (chain rule from d C^H / d grad f, d C^H / d Hess f).

#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include "linalg.cuh"

namespace tpms_basis {

constexpr float pi = 3.14159265358979323846f;
constexpr float pi2 = 6.28318530717958647692f;
constexpr float four_pi_sq = 4.0f * pi * pi;

// sin / cos of 2 pi x, 2 pi y, 2 pi z, shared by all basis functions at one point.
struct TrigPrecalc {
    float sX, cX;
    float sY, cY;
    float sZ, cZ;

    __device__ __forceinline__ TrigPrecalc(const float &X, const float &Y, const float &Z) {
        sincosf(pi2 * X, &sX, &cX);
        sincosf(pi2 * Y, &sY, &cY);
        sincosf(pi2 * Z, &sZ, &cZ);
    }
};

// f1 -------------------------------------------------------------------------

__device__ __forceinline__ float f1(const TrigPrecalc &t) {
    return t.cX + t.cY + t.cZ;
}

__device__ __forceinline__ CudaVec<float, 3> f1_grad(const TrigPrecalc &t) {
    return CudaVec<float, 3>(-pi2 * t.sX, -pi2 * t.sY, -pi2 * t.sZ);
}

__device__ __forceinline__ CudaMat<float, 3, 3> f1_hessian(const TrigPrecalc &t) {
    return CudaMat<float, 3, 3>(
        -four_pi_sq * t.cX, 0.0f, 0.0f,
        0.0f, -four_pi_sq * t.cY, 0.0f,
        0.0f, 0.0f, -four_pi_sq * t.cZ
    );
}

// f2 -------------------------------------------------------------------------

__device__ __forceinline__ float f2(const TrigPrecalc &t) {
    return t.sX * t.sY + t.sY * t.sZ + t.sZ * t.sX;
}

__device__ __forceinline__ CudaVec<float, 3> f2_grad(const TrigPrecalc &t) {
    return CudaVec<float, 3>(
        pi2 * t.cX * (t.sY + t.sZ),
        pi2 * t.cY * (t.sZ + t.sX),
        pi2 * t.cZ * (t.sX + t.sY)
    );
}

__device__ __forceinline__ CudaMat<float, 3, 3> f2_hessian(const TrigPrecalc &t) {
    float h00 = -four_pi_sq * t.sX * (t.sY + t.sZ);
    float h11 = -four_pi_sq * t.sY * (t.sZ + t.sX);
    float h22 = -four_pi_sq * t.sZ * (t.sX + t.sY);

    float h01 = four_pi_sq * t.cX * t.cY;
    float h02 = four_pi_sq * t.cX * t.cZ;
    float h12 = four_pi_sq * t.cY * t.cZ;

    return CudaMat<float, 3, 3>(
        h00, h01, h02,
        h01, h11, h12,
        h02, h12, h22
    );
}

// f3 -------------------------------------------------------------------------

__device__ __forceinline__ float f3(const TrigPrecalc &t) {
    return t.cX * t.cY + t.cY * t.cZ + t.cZ * t.cX;
}

__device__ __forceinline__ CudaVec<float, 3> f3_grad(const TrigPrecalc &t) {
    return CudaVec<float, 3>(
        -pi2 * t.sX * (t.cY + t.cZ),
        -pi2 * t.sY * (t.cZ + t.cX),
        -pi2 * t.sZ * (t.cX + t.cY)
    );
}

__device__ __forceinline__ CudaMat<float, 3, 3> f3_hessian(const TrigPrecalc &t) {
    float h00 = -four_pi_sq * t.cX * (t.cY + t.cZ);
    float h11 = -four_pi_sq * t.cY * (t.cZ + t.cX);
    float h22 = -four_pi_sq * t.cZ * (t.cX + t.cY);

    float h01 = four_pi_sq * t.sX * t.sY;
    float h02 = four_pi_sq * t.sX * t.sZ;
    float h12 = four_pi_sq * t.sY * t.sZ;

    return CudaMat<float, 3, 3>(
        h00, h01, h02,
        h01, h11, h12,
        h02, h12, h22
    );
}

// f4 -------------------------------------------------------------------------

__device__ __forceinline__ float f4(const TrigPrecalc &t) {
    return t.sX * t.cY + t.sY * t.cZ + t.sZ * t.cX;
}

__device__ __forceinline__ CudaVec<float, 3> f4_grad(const TrigPrecalc &t) {
    return CudaVec<float, 3>(
        pi2 * (t.cX * t.cY - t.sX * t.sZ),
        pi2 * (t.cY * t.cZ - t.sY * t.sX),
        pi2 * (t.cZ * t.cX - t.sZ * t.sY)
    );
}

__device__ __forceinline__ CudaMat<float, 3, 3> f4_hessian(const TrigPrecalc &t) {
    float h00 = -four_pi_sq * (t.sX * t.cY + t.sZ * t.cX);
    float h11 = -four_pi_sq * (t.sY * t.cZ + t.sX * t.cY);
    float h22 = -four_pi_sq * (t.sZ * t.cX + t.sY * t.cZ);

    float h01 = -four_pi_sq * t.cX * t.sY;
    float h02 = -four_pi_sq * t.sX * t.cZ;
    float h12 = -four_pi_sq * t.cY * t.sZ;

    return CudaMat<float, 3, 3>(
        h00, h01, h02,
        h01, h11, h12,
        h02, h12, h22
    );
}

// f5 -------------------------------------------------------------------------

__device__ __forceinline__ float f5(const TrigPrecalc &t) {
    return t.sX * t.sY * t.cZ + t.sY * t.sZ * t.cX + t.sZ * t.sX * t.cY;
}

__device__ __forceinline__ CudaVec<float, 3> f5_grad(const TrigPrecalc &t) {
    return CudaVec<float, 3>(
        pi2 * (t.cX * t.sY * t.cZ - t.sX * t.sY * t.sZ + t.cX * t.sZ * t.cY),
        pi2 * (t.sX * t.cY * t.cZ + t.cY * t.sZ * t.cX - t.sY * t.sZ * t.sX),
        pi2 * (-t.sX * t.sY * t.sZ + t.sY * t.cZ * t.cX + t.cZ * t.sX * t.cY)
    );
}

__device__ __forceinline__ CudaMat<float, 3, 3> f5_hessian(const TrigPrecalc &t) {
    // all three diagonal entries equal -4 pi^2 f5
    float f5_val = t.sX * t.sY * t.cZ + t.sY * t.sZ * t.cX + t.sZ * t.sX * t.cY;
    float h00 = -four_pi_sq * f5_val;

    float term_xyz = t.cX * t.cY * t.cZ;

    float h01 = four_pi_sq * (term_xyz - t.sX * t.sZ * t.cY - t.sY * t.sZ * t.cX);
    float h02 = four_pi_sq * (term_xyz - t.sX * t.sY * t.cZ - t.sZ * t.sY * t.cX);
    float h12 = four_pi_sq * (term_xyz - t.sY * t.sX * t.cZ - t.sZ * t.sX * t.cY);

    return CudaMat<float, 3, 3>(
        h00, h01, h02,
        h01, h00, h12,
        h02, h12, h00
    );
}

// f6 -------------------------------------------------------------------------

__device__ __forceinline__ float f6(const TrigPrecalc &t) {
    return t.cX * t.cY * t.cZ;
}

__device__ __forceinline__ CudaVec<float, 3> f6_grad(const TrigPrecalc &t) {
    return CudaVec<float, 3>(
        -pi2 * t.sX * t.cY * t.cZ,
        -pi2 * t.cX * t.sY * t.cZ,
        -pi2 * t.cX * t.cY * t.sZ
    );
}

__device__ __forceinline__ CudaMat<float, 3, 3> f6_hessian(const TrigPrecalc &t) {
    // all three diagonal entries equal -4 pi^2 f6
    float h00 = -four_pi_sq * t.cX * t.cY * t.cZ;

    float h01 = four_pi_sq * t.sX * t.sY * t.cZ;
    float h02 = four_pi_sq * t.sX * t.sZ * t.cY;
    float h12 = four_pi_sq * t.sY * t.sZ * t.cX;

    return CudaMat<float, 3, 3>(
        h00, h01, h02,
        h01, h00, h12,
        h02, h12, h00
    );
}

__device__ float combined_field(float c1, float c2, float c3, float c4, float c5, float c6, const float &X, const float &Y, const float &Z)
{
    TrigPrecalc t(X, Y, Z);
    return c1 * f1(t) + c2 * f2(t) + c3 * f3(t) + c4 * f4(t) + c5 * f5(t) + c6 * f6(t);
}

} // namespace tpms_basis

// Field functor for SDFTextureBuffer::init_from_kernel.
struct CombinedTPMSField
{
    float c1, c2, c3, c4, c5, c6;
    explicit CombinedTPMSField(const std::vector<float>& c) : c1(c[0]), c2(c[1]), c3(c[2]), c4(c[3]), c5(c[4]), c6(c[5]) {}
    __device__ __forceinline__ float operator()(const CudaVec<float, 3>& p) {
        return tpms_basis::combined_field(c1, c2, c3, c4, c5, c6, p[0], p[1], p[2]);
    }
};

// Writes a dim^3 float64 array (C order) as a .npy file.
inline void save_npy_float64(const std::string& path, const std::vector<double>& data, int dim)
{
    const char magic[] = "\x93NUMPY";
    uint8_t major = 1, minor = 0;

    std::string header_dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (" +
                              std::to_string(dim) + ", " + std::to_string(dim) + ", " + std::to_string(dim) + "), }";
    // magic(6) + version(2) + header_len(2) + header must be a multiple of 64 bytes
    int prefix_len = 6 + 2 + 2;
    int total = prefix_len + (int)header_dict.size() + 1;
    int padding = (64 - (total % 64)) % 64;
    header_dict += std::string(padding, ' ');
    header_dict += '\n';
    uint16_t header_len = (uint16_t)header_dict.size();

    std::ofstream f(path, std::ios::binary);
    f.write(magic, 6);
    f.write(reinterpret_cast<const char*>(&major), 1);
    f.write(reinterpret_cast<const char*>(&minor), 1);
    f.write(reinterpret_cast<const char*>(&header_len), 2);
    f.write(header_dict.data(), header_len);
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
}

// Samples the combined field at the dim^3 cell centres of the unit cube and saves it in the
// input format of `shellhomo` (array index [x][y][z]), so a snapshot can be homogenized again.
inline void export_combined_field_npy(const std::vector<float>& coefs, int dim, const std::string& path)
{
    const float pi2 = 2.0f * M_PI;
    size_t total = (size_t)dim * dim * dim;
    std::vector<double> sdf(total);
    float dx = 1.0f / dim;

    for (int z = 0; z < dim; z++) {
        float Z = (z + 0.5f) * dx;
        float sZ, cZ; sincosf(pi2 * Z, &sZ, &cZ);
        for (int y = 0; y < dim; y++) {
            float Y = (y + 0.5f) * dx;
            float sY, cY; sincosf(pi2 * Y, &sY, &cY);
            for (int x = 0; x < dim; x++) {
                float X = (x + 0.5f) * dx;
                float sX, cX; sincosf(pi2 * X, &sX, &cX);

                float f1 = cX + cY + cZ;
                float f2 = sX*sY + sY*sZ + sZ*sX;
                float f3 = cX*cY + cY*cZ + cZ*cX;
                float f4 = sX*cY + sY*cZ + sZ*cX;
                float f5 = sX*sY*cZ + sY*sZ*cX + sZ*sX*cY;
                float f6 = cX*cY*cZ;

                float val = coefs[0]*f1 + coefs[1]*f2 + coefs[2]*f3 +
                            coefs[3]*f4 + coefs[4]*f5 + coefs[5]*f6;

                size_t idx = (size_t)x * dim * dim + (size_t)y * dim + z;
                sdf[idx] = (double)val;
            }
        }
    }
    save_npy_float64(path, sdf, dim);
}
