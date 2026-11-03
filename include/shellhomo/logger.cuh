#pragma once
// Per-run statistics (timings, problem size, peak GPU memory, C^H), appended to a JSON array file.

#include <string>
#include <fstream>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include "linalg.cuh"

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/prettywriter.h"

// Samples the device memory in use (cudaMemGetInfo) on a background thread and keeps the peak.
class ScopedMemoryMonitor {
public:
    ScopedMemoryMonitor(int interval_ms = 10)
        : stop_signal(false), peak_memory_mb(0.0f), interval(interval_ms)
    {
        initial_memory_mb = get_memory_snapshot();
        peak_memory_mb = initial_memory_mb;
        worker = std::thread(&ScopedMemoryMonitor::monitor_loop, this);
    }

    ~ScopedMemoryMonitor() {
        stop();
    }

    void stop() {
        if (worker.joinable()) {
            stop_signal = true;
            worker.join();
        }
    }

    // Peak device memory in use during the monitored scope (MB).
    float get_peak_memory() const {
        return peak_memory_mb;
    }

private:
    std::thread worker;
    std::atomic<bool> stop_signal;
    std::atomic<float> peak_memory_mb;
    float initial_memory_mb;
    int interval;

    float get_memory_snapshot() {
        size_t free, total;
        if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
            return static_cast<float>(total - free) / (1024.0f * 1024.0f);
        }
        return 0.0f;
    }

    void monitor_loop() {
        while (!stop_signal) {
            float current = get_memory_snapshot();
            float old_peak = peak_memory_mb.load();
            if (current > old_peak) {
                peak_memory_mb.store(current);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(interval));
        }
    }
};

struct log_data
{
    std::string name;
    float memory;
    float total_time;
    float resolution;
    int matrix_size;

    int num_quads;
    int num_activeTets;

    float init_grid_time;
    float construct_stiffness_time;
    float construct_stab_time;
    float construct_rhs_time;
    float solve_time;
    CudaMat<float, 6, 6> homogenization_matrix;

    void clear()
    {
        name = "";
        memory = 0.0f;
        total_time = 0.0f;
        resolution = 0.0f;
        matrix_size = 0;
        construct_stiffness_time = 0.0f;
        solve_time = 0.0f;
        homogenization_matrix.setZero();
        num_quads = 0;
        num_activeTets = 0;
        init_grid_time = 0.0f;
        construct_stab_time = 0.0f;
        construct_rhs_time = 0.0f;
    }
};

class logger
{
public:
    logger(std::string _outfile)
        : outfile(_outfile) {};

    // Appends one record to the JSON array in `outfile` (created if missing).
    void log(const log_data& data)
    {
        rapidjson::Document d;

        std::ifstream ifs(outfile);
        if (ifs.good()) {
            rapidjson::IStreamWrapper isw(ifs);
            d.ParseStream(isw);
            ifs.close();
        } else {
            d.SetArray();
        }

        if (!d.IsArray()) {
            d.SetArray();
        }

        rapidjson::Document::AllocatorType& allocator = d.GetAllocator();

        rapidjson::Value obj(rapidjson::kObjectType);

        rapidjson::Value nameVal;
        nameVal.SetString(data.name.c_str(), static_cast<rapidjson::SizeType>(data.name.length()), allocator);
        obj.AddMember("name", nameVal, allocator);
        obj.AddMember("memory_usage_mb", data.memory, allocator);
        obj.AddMember("total_time_ms", data.total_time, allocator);
        obj.AddMember("resolution", data.resolution, allocator);
        obj.AddMember("matrix_size", data.matrix_size, allocator);
        obj.AddMember("init_grid_time", data.init_grid_time, allocator);
        obj.AddMember("construct_stiffness_time_ms", data.construct_stiffness_time, allocator);
        obj.AddMember("construct_stab_time_ms", data.construct_stab_time, allocator);
        obj.AddMember("construct_rhs_time_ms", data.construct_rhs_time, allocator);
        obj.AddMember("solve_time_ms", data.solve_time, allocator);
        obj.AddMember("num_quads", data.num_quads, allocator);
        obj.AddMember("num_activeTets", data.num_activeTets, allocator);

        rapidjson::Value matrixArray(rapidjson::kArrayType);
        for(int i = 0; i < 6; ++i) {
            rapidjson::Value row(rapidjson::kArrayType);
            for(int j = 0; j < 6; ++j) {
                float val = data.homogenization_matrix(i, j);
                row.PushBack(val, allocator);
            }
            matrixArray.PushBack(row, allocator);
        }
        obj.AddMember("homogenization_matrix", matrixArray, allocator);

        d.PushBack(obj, allocator);

        std::ofstream ofs(outfile);
        rapidjson::OStreamWrapper osw(ofs);
        rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(osw);
        d.Accept(writer);
    }

private:
    std::string outfile;
};
