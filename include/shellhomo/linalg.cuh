#pragma once
// Small fixed-size vector / matrix types usable on host and device, plus CUDA helpers.

#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

template<typename T, int N>
class CudaVec
{
public:
  __host__ __device__ CudaVec() {}
  __host__ __device__ explicit CudaVec(const T &value)
  {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dataPtr[i] = value;
    }
  }

  __host__ __device__ explicit CudaVec(const T &v0, const T &v1, const T &v2, const T &v3)
  {
    static_assert(N == 4, "CudaVec constructor only supports 4 components.");
    dataPtr[0] = v0;
    dataPtr[1] = v1;
    dataPtr[2] = v2;
    dataPtr[3] = v3;
  }

  __host__ __device__ explicit CudaVec(const T &v0, const T &v1, const T &v2)
  {
    static_assert(N == 3, "CudaVec constructor only supports 3 components.");
    dataPtr[0] = v0;
    dataPtr[1] = v1;
    dataPtr[2] = v2;
  }

  __host__ __device__ T &operator[](int index)
  {
    return dataPtr[index];
  }

  __host__ __device__ const T &operator[](int index) const
  {
    return dataPtr[index];
  }

  __host__ __device__ CudaVec<T, N> operator+(const CudaVec<T, N> &other) const
  {
    CudaVec<T, N> result;

#pragma unroll
    for (int i = 0; i < N; ++i) {
      result[i] = dataPtr[i] + other[i];
    }
    return result;
  }

  __host__ __device__ CudaVec<T, N> operator-(const CudaVec<T, N> &other) const
  {
    CudaVec<T, N> result;

#pragma unroll
    for (int i = 0; i < N; ++i) {
      result[i] = dataPtr[i] - other[i];
    }
    return result;
  }

  __host__ __device__ CudaVec<T, N> operator*(const T &scalar) const
  {
    CudaVec<T, N> result;

#pragma unroll
    for (int i = 0; i < N; ++i) {
      result[i] = dataPtr[i] * scalar;
    }
    return result;
  }

  __host__ __device__ CudaVec<T, N> operator/(const T &scalar) const
  {
    CudaVec<T, N> result;

#pragma unroll
    for (int i = 0; i < N; ++i) {
      result[i] = dataPtr[i] / scalar;
    }
    return result;
  }

  __host__ __device__ CudaVec<T, N> &operator+=(const CudaVec<T, N> &other)
  {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dataPtr[i] += other[i];
    }
    return *this;
  }

  __host__ __device__ CudaVec<T, N> &operator-=(const CudaVec<T, N> &other)
  {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dataPtr[i] -= other[i];
    }
    return *this;
  }

  __host__ __device__ CudaVec<T, N> &operator*=(const T &scalar)
  {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dataPtr[i] *= scalar;
    }
    return *this;
  }

  __host__ __device__ CudaVec<T, N> &operator/=(const T &scalar)
  {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dataPtr[i] /= scalar;
    }
    return *this;
  }

  __host__ __device__ T norm() const
  {
    T s = 0;

#pragma unroll
    for (int i = 0; i < N; ++i) {
      s += dataPtr[i] * dataPtr[i];
    }
    return std::sqrt(s);
  }

  __host__ __device__ void setZero()
  {
  #pragma unroll
    for (int i = 0; i < N; ++i) {
      dataPtr[i] = 0;
    }
  }

  __host__ __device__ T dot(const CudaVec<T, N> &other) const
  {
    T s = 0;

#pragma unroll
    for (int i = 0; i < N; ++i) {
      s += dataPtr[i] * other.dataPtr[i];
    }
    return s;
  }

  __host__ __device__ CudaVec<T, N> cross (const CudaVec<T, N> &other) const
  {
    CudaVec<T, N> result;
    result[0] = dataPtr[1] * other.dataPtr[2] - dataPtr[2] * other.dataPtr[1];
    result[1] = dataPtr[2] * other.dataPtr[0] - dataPtr[0] * other.dataPtr[2];
    result[2] = dataPtr[0] * other.dataPtr[1] - dataPtr[1] * other.dataPtr[0];
    return result;
  }

  __host__ __device__ T *data()
  {
    return dataPtr;
  }

  __host__ __device__ const T *data() const
  {
    return dataPtr;
  }

private:
  T dataPtr[N];
};

// scalar * vec
template<typename T, int N>
__host__ __device__ CudaVec<T, N> operator*(const T &scalar, const CudaVec<T, N> &v)
{
    return v * scalar;
}

// Column-major fixed-size matrix.
template<typename T, int Rows, int Cols>
class CudaMat
{
public:
  __host__ __device__ CudaMat() noexcept
  {
  }

  // Row-major list of Rows*Cols entries.
  template <typename... Args,
            typename = typename std::enable_if<sizeof...(Args) == Rows * Cols>::type>
  __host__ __device__ explicit CudaMat(Args... args)
  {
      const T inputs[Rows * Cols] = { static_cast<T>(args)... };

      #pragma unroll
      for (int k = 0; k < Rows * Cols; ++k) {
          int r = k / Cols;
          int c = k % Cols;
          dataPtr[c * Rows + r] = inputs[k];
      }
  }
  __host__ __device__ T &operator()(int row, int col)
  {
    assert(row >= 0 && row < Rows);
    assert(col >= 0 && col < Cols);

    return dataPtr[col * Rows + row];
  }

  __host__ __device__ const T &operator()(int row, int col) const
  {
    assert(row >= 0 && row < Rows);
    assert(col >= 0 && col < Cols);

    return dataPtr[col * Rows + row];
  }

  __host__ __device__ CudaMat<T, Rows, Cols> operator+(const CudaMat<T, Rows, Cols> &other) const
  {
    CudaMat<T, Rows, Cols> result;

#pragma unroll
    for (int i = 0; i < Rows * Cols; ++i) {
      result.dataPtr[i] = dataPtr[i] + other.dataPtr[i];
    }
    return result;
  }

  __host__ __device__ CudaMat<T, Rows, Cols> operator-(const CudaMat<T, Rows, Cols> &other) const
  {
    CudaMat<T, Rows, Cols> result;

#pragma unroll
    for (int i = 0; i < Rows * Cols; ++i) {
      result.dataPtr[i] = dataPtr[i] - other.dataPtr[i];
    }
    return result;
  }

  template<int Cols2>
  __host__ __device__ CudaMat<T, Rows, Cols2> operator*(const CudaMat<T, Cols, Cols2>& other) const
  {
    CudaMat<T, Rows, Cols2> result;
    #pragma unroll
    for (int i = 0; i < Rows; ++i)
    {
      #pragma unroll
      for (int j = 0; j < Cols2; ++j)
      {
          T sum = T(0);
          #pragma unroll
          for (int k = 0; k < Cols; ++k)
          {
              sum += (*this)(i, k) * other(k, j);
          }
          result(i, j) = sum;
      }
    }
    return result;
  }

  __host__ __device__ CudaMat<T, Rows, Cols> operator*(T scalar) const
  {
    CudaMat<T, Rows, Cols> result;
#pragma unroll
    for (int i = 0; i < Rows * Cols; ++i) {
      result.dataPtr[i] = dataPtr[i] * scalar;
    }
    return result;
  }

  __host__ __device__ CudaVec<T, Rows> operator*(const CudaVec<T, Cols> &vec) const
  {
    CudaVec<T, Rows> result;

#pragma unroll
    for (int i = 0; i < Rows; ++i) {
      T sum = T(0);

#pragma unroll
      for (int j = 0; j < Cols; ++j) {
        sum += (*this)(i, j) * vec[j];
      }
      result[i] = sum;
    }
    return result;
  }

  __host__ __device__ CudaMat<T, Rows, Cols> operator/(T scalar) const
  {
    CudaMat<T, Rows, Cols> result;
#pragma unroll
    for (int i = 0; i < Rows * Cols; ++i) {
      result.dataPtr[i] = dataPtr[i] / scalar;
    }
    return result;
  }

  __host__ __device__ CudaMat<T, Rows, Cols> &operator+=(const CudaMat<T, Rows, Cols> &other)
  {
#pragma unroll
    for (int i = 0; i < Rows * Cols; ++i) {
      dataPtr[i] += other.dataPtr[i];
    }
    return *this;
  }

  __host__ __device__ CudaMat<T, Rows, Cols> &operator-=(const CudaMat<T, Rows, Cols> &other)
  {
#pragma unroll
    for (int i = 0; i < Rows * Cols; ++i) {
      dataPtr[i] -= other.dataPtr[i];
    }
    return *this;
  }

  __host__ __device__ CudaMat<T, Rows, Cols> &operator*=(T scalar)
  {
#pragma unroll
    for (int i = 0; i < Rows * Cols; ++i) {
      dataPtr[i] *= scalar;
    }
    return *this;
  }

  __host__ __device__ CudaMat<T, Rows, Cols> &operator/=(T scalar)
  {
#pragma unroll
    for (int i = 0; i < Rows * Cols; ++i) {
      dataPtr[i] /= scalar;
    }
    return *this;
  }

  __host__ __device__ void setZero()
  {
#pragma unroll
    for (int i = 0; i < Rows * Cols; ++i) {
      dataPtr[i] = T(0);
    }
  }

  __host__ __device__ void setIdentity()
  {
    setZero();
#pragma unroll
    for (int i = 0; i < Rows; ++i) {
      for (int j = 0; j < Cols; ++j) {
        if (i == j) {
          (*this)(i, j) = T(1);
        }
      }
    }
  }

  __host__ __device__ T *data()
  {
    return dataPtr;
  }

  __host__ __device__ const T *data() const
  {
    return dataPtr;
  }

  __host__ __device__ CudaVec<T, Rows> col(int col) const
  {
    CudaVec<T, Rows> result;
    #pragma unroll
    for (int i = 0; i < Rows; ++i)
      result[i] = (*this)(i, col);
    return result;
  }

__host__ __device__ __forceinline__ CudaMat<T, 3, 3> inverse(T* det_out = nullptr) const
{
    static_assert(Rows == 3 && Cols == 3, "Inverse only implemented for 3x3 matrices");

    // Col 0
    const T m00 = dataPtr[0]; const T m10 = dataPtr[1]; const T m20 = dataPtr[2];
    // Col 1
    const T m01 = dataPtr[3]; const T m11 = dataPtr[4]; const T m21 = dataPtr[5];
    // Col 2
    const T m02 = dataPtr[6]; const T m12 = dataPtr[7]; const T m22 = dataPtr[8];

    // Adjugate (transpose of the cofactor matrix)
    T inv00 = m11 * m22 - m12 * m21;
    T inv01 = m02 * m21 - m01 * m22;
    T inv02 = m01 * m12 - m02 * m11;

    T inv10 = m12 * m20 - m10 * m22;
    T inv11 = m00 * m22 - m02 * m20;
    T inv12 = m02 * m10 - m00 * m12;

    T inv20 = m10 * m21 - m11 * m20;
    T inv21 = m01 * m20 - m00 * m21;
    T inv22 = m00 * m11 - m01 * m10;

    T det = m00 * inv00 + m10 * inv01 + m20 * inv02;

    if (det_out != nullptr) {
        *det_out = det;
    }

    // Singular matrices return the zero matrix.
    const T epsilon = T(1e-12);
    T invDet;

    if (abs(det) < epsilon) {
        invDet = T(0);
    } else {
        invDet = T(1) / det;
    }

    CudaMat<T, 3, 3> result;

    // Col 0
    result.dataPtr[0] = inv00 * invDet;
    result.dataPtr[1] = inv10 * invDet;
    result.dataPtr[2] = inv20 * invDet;

    // Col 1
    result.dataPtr[3] = inv01 * invDet;
    result.dataPtr[4] = inv11 * invDet;
    result.dataPtr[5] = inv21 * invDet;

    // Col 2
    result.dataPtr[6] = inv02 * invDet;
    result.dataPtr[7] = inv12 * invDet;
    result.dataPtr[8] = inv22 * invDet;

    return result;
}

__device__ __host__ void setCol(int col, const CudaVec<T, Rows>& vec)
{
  #pragma unroll
  for (int i = 0; i < Rows; i++)
    (*this)(i, col) = T(vec[i]);
}

// Square matrices only.
__device__ __host__ CudaMat<T, Rows, Cols> transpose() const
{
  CudaMat<T, Cols, Rows> result;
  #pragma unroll
  for (int i = 0; i < Rows; i++)
  {
    #pragma unroll
    for (int j = 0; j < Cols; j++)
      result(i, j) = (*this)(j, i);
  }
  return result;
}

  __device__ __host__ T trace() const
  {
    T sum = T(0);
    #pragma unroll
    for (int i = 0; i < Rows; i++)
      sum += (*this)(i, i);
    return sum;
  }

  __device__ __host__ CudaMat<T, Rows, Cols> cwiseProduct(const CudaMat<T, Rows, Cols> & other) const
  {
    CudaMat<T, Rows, Cols> result;
    #pragma unroll
    for (int i = 0; i < Rows * Cols; i++)
      result.dataPtr[i] = dataPtr[i] * other.dataPtr[i];
    return result;
  }

  __device__ __host__ T sum() const
  {
    T sum = T(0);
    #pragma unroll
    for (int i = 0; i < Rows * Cols; i++)
      sum += dataPtr[i];
    return sum;
  }

private:
  T dataPtr[Rows * Cols];
};

template <typename T, int Rows, int Cols>
std::ostream& operator<<(std::ostream& os, const CudaMat<T, Rows, Cols>& mat)
{
    std::ios state(nullptr);
    state.copyfmt(os);

    os << std::scientific << std::setprecision(4);

    for (int i = 0; i < Rows; ++i) {
        os << "[ ";
        for (int j = 0; j < Cols; ++j) {
            os << std::setw(12) << mat(i, j) << " ";
        }
        os << "]\n";
    }

    os.copyfmt(state);
    return os;
}

// scalar * mat (the scalar is first converted to the matrix scalar type)
template <typename T1, typename T2, int Rows, int Cols,
          typename = typename std::enable_if<std::is_arithmetic<T2>::value>::type>
__host__ __device__ inline CudaMat<T1, Rows, Cols> operator*(T2 scalar, const CudaMat<T1, Rows, Cols>& mat)
{
    return mat * (T1)scalar;
}

template<typename T, int N>
__host__ __device__ CudaMat<T, N, N> outer_product(const CudaVec<T, N> &a, const CudaVec<T, N> &b)
{
    CudaMat<T, N, N> result;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            result(i, j) = a[i] * b[j];
        }
    }
    return result;
}

template <typename T>
__device__ __forceinline__ T clamp(T x, T a, T b) {
    return max(a, min(b, x));
}

// Event-based timer on the legacy default stream (milliseconds).
struct GpuTimer {
    cudaEvent_t start, stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Tic() {
        cudaEventRecord(start, 0);
    }

    void Toc() {
        cudaEventRecord(stop, 0);
    }

    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result != cudaSuccess)
    {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << " \"" << func << "\" \n";
        std::cerr << "Error string: " << cudaGetErrorString(result) << std::endl;
        exit(99);
    }
}

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        std::cerr << "CUDA error at " << file << ":" << line << " : " << errorMessage << " : " << cudaGetErrorString(err) << ".\n";
        exit(99);
    }

#ifdef _DEBUG
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err)
    {
        std::cerr << "CUDA error at " << file << ":" << line << " : " << errorMessage << " (Sync) : " << cudaGetErrorString(err) << ".\n";
        exit(99);
    }
#endif
}
