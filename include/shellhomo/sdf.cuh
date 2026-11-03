#pragma once
// Implicit shell geometry: a 257^3 cell-centred scalar field on the unit cube, stored in a
// 3D CUDA texture (trilinear filtering). Values are fetched from the texture; gradients
// are central differences with a one-texel step.

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "linalg.cuh"

namespace NpyLoader {

    // Parses the `shape` tuple of an .npy header.
    inline std::vector<int> parseHeader(const std::string& header) {
        std::vector<int> shape;
        size_t loc1 = header.find("shape");
        if (loc1 == std::string::npos) return shape;

        size_t loc2 = header.find("(", loc1);
        size_t loc3 = header.find(")", loc2);
        if (loc2 == std::string::npos || loc3 == std::string::npos) return shape;

        std::string shape_str = header.substr(loc2 + 1, loc3 - loc2 - 1);
        std::replace(shape_str.begin(), shape_str.end(), ',', ' ');

        std::stringstream ss(shape_str);
        int dim;
        while (ss >> dim) {
            shape.push_back(dim);
        }
        return shape;
    }

    // Reads a float64 .npy file (3D, or flattened 1D of a cube) indexed as [x][y][z]
    // (z fastest), converts it to float and reorders it to the CUDA texture layout (x fastest).
    inline void loadDoubleToFloat(const std::string& filepath, thrust::host_vector<float>& input_vector) {
        std::ifstream stream(filepath, std::ifstream::binary);
        if (!stream) {
            throw std::runtime_error("NpyLoader: Could not open file: " + filepath);
        }

        char magic[6];
        stream.read(magic, 6);
        if (std::string(magic, 6) != "\x93NUMPY") {
            throw std::runtime_error("NpyLoader: Invalid NPY file.");
        }

        stream.seekg(2, std::ios::cur);
        unsigned short header_len;
        stream.read(reinterpret_cast<char*>(&header_len), 2);

        std::vector<char> header_buf(header_len);
        stream.read(header_buf.data(), header_len);
        std::string header(header_buf.begin(), header_buf.end());

        std::vector<int> shape = parseHeader(header);

        std::streampos data_offset = 10 + header_len;

        stream.seekg(0, std::ios::end);
        std::streampos file_size = stream.tellg();
        std::streampos data_byte_size = file_size - data_offset;

        if (data_byte_size % sizeof(double) != 0) {
            throw std::runtime_error("NpyLoader: File size does not match float64 alignment.");
        }
        size_t num_elements = data_byte_size / sizeof(double);

        std::vector<double> host_double_data(num_elements);
        stream.seekg(data_offset, std::ios::beg);
        stream.read(reinterpret_cast<char*>(host_double_data.data()), data_byte_size);

        if (!stream) {
            throw std::runtime_error("NpyLoader: Error reading data content.");
        }

        int dimX = 0, dimY = 0, dimZ = 0;
        bool do_transpose = false;

        if (shape.size() == 3) {
            dimX = shape[0]; dimY = shape[1]; dimZ = shape[2];
            do_transpose = true;
        }
        else if (shape.size() == 1) {
            // Flattened array: assume a cube.
            int d = std::round(std::cbrt(num_elements));
            if ((size_t)d * d * d == num_elements) {
                dimX = d; dimY = d; dimZ = d;
                do_transpose = true;
                std::cout << "NpyLoader: Detected 1D array (" << num_elements << "), inferred 3D shape: "
                          << dimX << "x" << dimY << "x" << dimZ << std::endl;
            } else {
                std::cerr << "Warning: 1D array size " << num_elements << " is not a perfect cube. Cannot transpose." << std::endl;
            }
        }

        input_vector.resize(num_elements);

        if (do_transpose) {
            std::cout << "NpyLoader: Transposing from Numpy (Z-Fast) to CUDA (X-Fast)..." << std::endl;

            if (dimX != 257) {
                std::cout << "Note: Inferred dim is " << dimX << " (expected 257?)" << std::endl;
            }

            for (int x = 0; x < dimX; ++x) {
                for (int y = 0; y < dimY; ++y) {
                    for (int z = 0; z < dimZ; ++z) {
                        size_t src_idx = (size_t)x * dimY * dimZ + (size_t)y * dimZ + z;
                        size_t dst_idx = (size_t)z * dimY * dimX + (size_t)y * dimX + x;

                        if (dst_idx < num_elements && src_idx < num_elements) {
                            input_vector[dst_idx] = static_cast<float>(host_double_data[src_idx]);
                        }
                    }
                }
            }
        } else {
            for (size_t i = 0; i < num_elements; ++i) {
                input_vector[i] = static_cast<float>(host_double_data[i]);
            }
        }

        std::cout << ".npy file loaded: " << num_elements << " elements." << std::endl;
    }
}

// Samples an implicit functor at the cell centres of a dim^3 grid (x fastest).
template<typename SDF>
__global__ void generate_sdf_kernel(float* output, SDF _sdf, int dim)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int x = idx % dim;
    int y = (idx % (dim * dim)) / dim;
    int z = idx / (dim * dim);

    if (x >= dim || y >= dim || z >= dim) return;

    float dx = 1.0f / dim;
    CudaVec<float, 3> p((x + 0.5f) * dx, (y + 0.5f) * dx, (z + 0.5f) * dx);

    output[idx] = _sdf(p);
}

struct SDFSampler
{
    cudaTextureObject_t texObj;
    float texel_size = 1.0f / 257.0f;
};

class SDFTextureBuffer
{
public:
    cudaTextureObject_t texObj = 0;
    cudaArray *d_buffer = nullptr;

    thrust::host_vector<float> h_npy_buffer;

    int dim = 257;

    SDFTextureBuffer(int _dim = 257) : dim(_dim)
    {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaExtent extent = make_cudaExtent(dim, dim, dim);
        checkCudaErrors(cudaMalloc3DArray(&d_buffer, &channelDesc, extent));

        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = d_buffer;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));

        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;

        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1; // texture coordinates == world coordinates in [0,1]

        checkCudaErrors(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
    };

    // Fills the texture by evaluating an implicit functor on the GPU.
    template<typename SDF>
    void init_from_kernel(SDF _sdf) {
        float *d_linear_buffer;
        size_t size_bytes = dim * dim * dim * sizeof(float);
        checkCudaErrors(cudaMalloc((void**)&d_linear_buffer, size_bytes));

        int blockSize = 512;
        int gridSize = (dim * dim * dim + blockSize - 1) / blockSize;

        generate_sdf_kernel<<<gridSize, blockSize>>>(d_linear_buffer, _sdf, dim);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)d_linear_buffer, dim * sizeof(float), dim, dim);
        copyParams.dstArray = d_buffer;
        copyParams.extent = make_cudaExtent(dim, dim, dim);
        copyParams.kind = cudaMemcpyDeviceToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));

        checkCudaErrors(cudaFree(d_linear_buffer));
    }

    void refresh_buffer(float *h_buffer)
    {
        if (h_buffer == nullptr || d_buffer == nullptr) return;

        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)h_buffer, dim * sizeof(float), dim, dim);
        copyParams.dstArray = d_buffer;
        copyParams.extent = make_cudaExtent(dim, dim, dim);
        copyParams.kind = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));
    };

    // The file must hold 257^3 float64 values (see NpyLoader::loadDoubleToFloat).
    void load_npy(std::string filename)
    {
        NpyLoader::loadDoubleToFloat(filename, h_npy_buffer);
        dim = 257;
        refresh_buffer(thrust::raw_pointer_cast(h_npy_buffer.data()));
    }

    ~SDFTextureBuffer()
    {
        if (texObj) cudaDestroyTextureObject(texObj);
        if (d_buffer) cudaFreeArray(d_buffer);
    };
};

namespace TextureBuffer
{
    __device__ __forceinline__ float shell_sdf(const CudaVec<float, 3>& pos, const SDFSampler& sampler)
    {
        return tex3D<float>(sampler.texObj, pos[0], pos[1], pos[2]);
    }

    __device__ __forceinline__ CudaVec<float, 3> shell_grad(const CudaVec<float, 3>& pos, const SDFSampler& sampler)
    {
        float eps = sampler.texel_size;

        float v_x1 = tex3D<float>(sampler.texObj, pos[0] + eps, pos[1], pos[2]);
        float v_x0 = tex3D<float>(sampler.texObj, pos[0] - eps, pos[1], pos[2]);
        float dx = (v_x1 - v_x0) / (2.0f * eps);

        float v_y1 = tex3D<float>(sampler.texObj, pos[0], pos[1] + eps, pos[2]);
        float v_y0 = tex3D<float>(sampler.texObj, pos[0], pos[1] - eps, pos[2]);
        float dy = (v_y1 - v_y0) / (2.0f * eps);

        float v_z1 = tex3D<float>(sampler.texObj, pos[0], pos[1], pos[2] + eps);
        float v_z0 = tex3D<float>(sampler.texObj, pos[0], pos[1], pos[2] - eps);
        float dz = (v_z1 - v_z0) / (2.0f * eps);

        return CudaVec<float, 3>(dx, dy, dz);
    }
}

// Built-in analytic field (Schoen gyroid), used when no input file is given.
template<typename T>
struct BuiltinGyroid
{
    __device__ T operator()(const CudaVec<T, 3>& p) {
        T tx = 2 * M_PI * p[0], ty = 2 * M_PI * p[1], tz = 2 * M_PI * p[2];
        T cx = cos(tx), cy = cos(ty), cz = cos(tz);
        T sx = sin(tx), sy = sin(ty), sz = sin(tz);
        return sx * cy + sy * cz + sz * cx;
    }
};
