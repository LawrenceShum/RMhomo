#pragma once
// Stiffness objectives on C^H and the Adam optimizer used by the inverse-design example.

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "linalg.cuh"

// The three objectives of the paper. Each maximizes a modulus, i.e. minimizes
// J = sum_IJ W_IJ C^H_IJ with negative weights (Voigt indices 0..5 = xx, yy, zz, xy, yz, xz):
//   bulk:       K = (C11 + C22 + C33 + 2 (C12 + C13 + C23)) / 9   (off-diagonals: upper triangle, weight 2/9)
//   shear:      G = (C44 + C55 + C66) / 3
//   stiffness:  (C11 + C22 + C33 + C44 + C55 + C66) / 6          (mean of the diagonal)
enum class Objective { Bulk, Shear, Stiffness };

inline bool parse_objective(const std::string& name, Objective& objective)
{
    if (name == "bulk")      { objective = Objective::Bulk;      return true; }
    if (name == "shear")     { objective = Objective::Shear;     return true; }
    if (name == "stiffness") { objective = Objective::Stiffness; return true; }
    return false;
}

inline CudaMat<float, 6, 6> objective_weight(Objective objective)
{
    CudaMat<float, 6, 6> w;
    w.setZero();
    switch (objective) {
    case Objective::Bulk:
        w(0, 0) = -1.0f / 9.0f;
        w(1, 1) = -1.0f / 9.0f;
        w(2, 2) = -1.0f / 9.0f;
        w(0, 1) = -2.0f / 9.0f;
        w(0, 2) = -2.0f / 9.0f;
        w(1, 2) = -2.0f / 9.0f;
        break;
    case Objective::Shear:
        w(3, 3) = -1.0f / 3.0f;
        w(4, 4) = -1.0f / 3.0f;
        w(5, 5) = -1.0f / 3.0f;
        break;
    case Objective::Stiffness:
        w(0, 0) = -1.0f / 6.0f;
        w(1, 1) = -1.0f / 6.0f;
        w(2, 2) = -1.0f / 6.0f;
        w(3, 3) = -1.0f / 6.0f;
        w(4, 4) = -1.0f / 6.0f;
        w(5, 5) = -1.0f / 6.0f;
        break;
    }
    return w;
}

// Adam (Kingma & Ba) on a host parameter vector.
template<typename T>
class Adam {
public:
    Adam(int _n_params, T _lr = 1e-2, T _beta1 = 0.95, T _beta2 = 0.999, T _eps = 1e-8)
        : n_params(_n_params), lr(_lr), beta1(_beta1), beta2(_beta2), epsilon(_eps), t(0)
    {
        m.resize(n_params, 0.0);
        v.resize(n_params, 0.0);
    }

    void step(std::vector<T>& params, const std::vector<T>& grads) {
        if ((int)params.size() != n_params || (int)grads.size() != n_params) {
            std::cerr << "Adam: Parameter size mismatch!" << std::endl;
            return;
        }

        t++;

        for (int i = 0; i < n_params; ++i) {
            T g = grads[i];

            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

            T m_hat = m[i] / (1.0 - std::pow(beta1, t));
            T v_hat = v[i] / (1.0 - std::pow(beta2, t));

            params[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }

    void set_lr(T new_lr) { lr = new_lr; }

private:
    int n_params;
    T lr;
    T beta1;
    T beta2;
    T epsilon;
    int t;  // time step

    std::vector<T> m;  // first moment
    std::vector<T> v;  // second moment
};
