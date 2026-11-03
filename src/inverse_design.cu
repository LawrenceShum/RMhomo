// shellhomo_inverse: gradient-based inverse design in the space of six combined TPMS basis
// functions. Starting from random coefficients, Adam with a cosine learning-rate decay
// maximizes the bulk modulus, the shear modulus, or the mean diagonal stiffness of C^H.
//
// Usage: shellhomo_inverse <config.json> [--out DIR]

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "shellhomo/config.h"
#include "shellhomo/simulator.cuh"
#include "shellhomo/inverse_design.cuh"

namespace fs = std::filesystem;

static void print_usage(const char* prog)
{
    std::cout <<
        "Usage: " << prog << " <config.json> [--out DIR]\n"
        "\n"
        "  config.json  grid, material and objective, e.g.\n"
        "               {\"nx\": 48, \"ny\": 48, \"nz\": 48, \"thickness\": 0.02, \"E\": 2e6, \"nu\": 0.3,\n"
        "                \"objective\": \"bulk\" | \"shear\" | \"stiffness\", \"seed\": 42}\n"
        "  --out DIR    output directory (default: inverse_<objective>)\n";
}

static void print_vector(const char* label, const std::vector<float>& v)
{
    std::cout << label << ": [";
    for (size_t i = 0; i < v.size(); i++)
        std::cout << v[i] << (i + 1 < v.size() ? ", " : "");
    std::cout << "]" << std::endl;
}

int main(int argc, char** argv)
{
    std::string config_path;
    std::string out_dir;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--out" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (config_path.empty()) {
            config_path = arg;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (config_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    rapidjson::Document config;
    simulation_setting settings;
    if (!load_json(config_path, config) || !read_simulation_setting(config, settings)) {
        std::cerr << "Failed to read settings from " << config_path << std::endl;
        return 1;
    }
    if (!check_simulation_setting(settings))
        return 1;

    Objective objective;
    if (!config.HasMember("objective") || !config["objective"].IsString() ||
        !parse_objective(config["objective"].GetString(), objective)) {
        std::cerr << "error: \"objective\" must be \"bulk\", \"shear\" or \"stiffness\"" << std::endl;
        return 1;
    }
    const std::string objective_name = config["objective"].GetString();

    unsigned int seed = 42;
    if (config.HasMember("seed")) {
        if (!config["seed"].IsUint()) {
            std::cerr << "error: \"seed\" must be a non-negative integer" << std::endl;
            return 1;
        }
        seed = config["seed"].GetUint();
    }
    std::cout << "objective: " << objective_name << std::endl;
    std::cout << "seed: " << seed << std::endl;

    if (out_dir.empty())
        out_dir = "inverse_" + objective_name;
    fs::create_directories(out_dir);

    simulator<float> sim(settings.nx, settings.ny, settings.nz, settings.E, settings.nu, settings.thickness, out_dir + "/results.json");
    sim.set_objective_weight(objective_weight(objective));

    // Optimization parameters
    const int max_iter = 300;
    const int n_params = 6;
    const float lr = 0.005f;      // initial learning rate
    const float lr_min = 1e-4f;   // final learning rate of the cosine decay
    const int snapshot_interval = 5;
    const int snapshot_dim = 257;

    // Random initial coefficients in [0, 1)
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> coefs(n_params);
    for (int i = 0; i < n_params; i++)
        coefs[i] = dist(rng);

    std::cout << "\n===== Inverse Homogenization Optimization =====" << std::endl;
    print_vector("Initial coefficients", coefs);

    Adam<float> adam(n_params, lr);
    std::vector<float> grad_coefs(n_params, 0.0f);

    float best_obj = 1e30f;
    std::vector<float> best_coefs = coefs;

    // One row per iteration: coefficients, objective and gradient before the Adam step.
    std::ofstream csv(out_dir + "/optimization_history.csv");
    csv << "iter,objective,lr,c0,c1,c2,c3,c4,c5,g0,g1,g2,g3,g4,g5" << std::endl;
    csv << std::setprecision(8);

    for (int iter = 0; iter < max_iter; iter++)
    {
        std::cout << "\n--- Iteration " << iter << " ---" << std::endl;

        float cos_lr = lr_min + 0.5f * (lr - lr_min) * (1.0f + std::cos((float)iter / (float)max_iter * M_PI));
        adam.set_lr(cos_lr);
        print_vector("Coefficients", coefs);

        // Forward pass
        float obj = sim.run_combined_field(coefs, std::to_string(iter));
        std::cout << "Objective: " << obj << "  (lr=" << cos_lr << ")" << std::endl;

        if (obj < best_obj) {
            best_obj = obj;
            best_coefs = coefs;
        }

        if (iter % snapshot_interval == 0 || iter == max_iter - 1) {
            char name[32];
            std::snprintf(name, sizeof(name), "sdf_iter_%04d.npy", iter);
            export_combined_field_npy(coefs, snapshot_dim, out_dir + "/" + name);
        }

        // Backward pass
        sim.compute_gradient(grad_coefs);
        print_vector("Gradient", grad_coefs);

        csv << iter << "," << obj << "," << cos_lr;
        for (int j = 0; j < n_params; j++)
            csv << "," << coefs[j];
        for (int j = 0; j < n_params; j++)
            csv << "," << grad_coefs[j];
        csv << std::endl;

        adam.step(coefs, grad_coefs);
    }

    std::cout << "\n===== Optimization Complete =====" << std::endl;
    std::cout << "Best objective: " << best_obj << std::endl;
    print_vector("Best coefficients", best_coefs);
    export_combined_field_npy(best_coefs, snapshot_dim, out_dir + "/sdf_best.npy");
    std::cout << "Results written to " << out_dir << std::endl;
    return 0;
}
