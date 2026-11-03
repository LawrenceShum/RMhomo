// shellhomo: GPU homogenization of Reissner--Mindlin shellular unit cells given as implicit fields.
//
// Usage: shellhomo <config.json> [input.npy | input_dir ...] [--log results.json] [--warmup N]

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "shellhomo/config.h"
#include "shellhomo/simulator.cuh"

namespace fs = std::filesystem;

// Every *.npy file in a directory, sorted by path.
static std::vector<std::string> get_all_npy_files(const std::string& folder_path) {
    std::vector<std::string> npy_files;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".npy") {
            npy_files.push_back(entry.path().string());
        }
    }
    std::sort(npy_files.begin(), npy_files.end());
    return npy_files;
}

static void print_usage(const char* prog)
{
    std::cout <<
        "Usage: " << prog << " <config.json> [input ...] [--log FILE] [--warmup N]\n"
        "\n"
        "  config.json  grid resolution and material, e.g.\n"
        "               {\"nx\": 48, \"ny\": 48, \"nz\": 48, \"thickness\": 0.02, \"E\": 2e6, \"nu\": 0.3}\n"
        "  input        a 257^3 float64 .npy implicit field, or a directory of them;\n"
        "               without inputs the built-in gyroid is homogenized\n"
        "  --log FILE   JSON file the results are appended to (default: results.json)\n"
        "  --warmup N   untimed warm-up runs on the built-in gyroid before the inputs (default: 3)\n";
}

int main(int argc, char** argv)
{
    std::string config_path;
    std::string log_path = "results.json";
    int warmup = 3;
    std::vector<std::string> inputs;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--log" && i + 1 < argc) {
            log_path = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if (config_path.empty()) {
            config_path = arg;
        } else {
            inputs.push_back(arg);
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

    std::vector<std::string> npy_files;
    for (const auto& in : inputs) {
        if (fs::is_directory(in)) {
            std::vector<std::string> files = get_all_npy_files(in);
            npy_files.insert(npy_files.end(), files.begin(), files.end());
        } else if (fs::is_regular_file(in)) {
            npy_files.push_back(in);
        } else {
            std::cerr << "error: input not found: " << in << std::endl;
            return 1;
        }
    }

    simulator<float> sim(settings.nx, settings.ny, settings.nz, settings.E, settings.nu, settings.thickness, log_path);

    // Warm-up (CUDA context, module loading, allocations); not logged.
    for (int i = 0; i < warmup; i++)
        sim.run_builtin_gyroid(false);

    if (inputs.empty()) {
        sim.run_builtin_gyroid();
    } else {
        for (const auto& f : npy_files)
            sim.load_data_and_run(f);
    }

    std::cout << "Results appended to " << log_path << std::endl;
    return 0;
}
