#pragma once
// JSON configuration shared by the command-line tools.

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

// Background grid and material (keys nx, ny, nz, thickness, E, nu; all required).
struct simulation_setting
{
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float E = 0.0;
    float nu = 0.0;
    float thickness = 0.0;
};

// Parses a JSON file whose root is an object.
inline bool load_json(const std::string& filename, rapidjson::Document& document)
{
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open file  " << filename << std::endl;
        return false;
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    std::string jsonContent = buffer.str();

    rapidjson::ParseResult ok = document.Parse(jsonContent.c_str());
    if (!ok) {
        std::cerr << "error: "
                  << rapidjson::GetParseError_En(ok.Code())
                  << " (offset: " << ok.Offset() << ")" << std::endl;
        return false;
    }
    if (!document.IsObject()) {
        std::cerr << "error: root is not an object" << std::endl;
        return false;
    }
    return true;
}

inline bool read_simulation_setting(const rapidjson::Document& document, simulation_setting& settings)
{
    bool complete = true;

    #define GET_INT_PARAM(name) \
        if (document.HasMember(#name) && document[#name].IsInt()) { \
            settings.name = document[#name].GetInt(); \
        } else { \
            std::cerr << "error: \"" #name "\" is missing or not an integer" << std::endl; \
            complete = false; \
        }

    GET_INT_PARAM(nx);
    GET_INT_PARAM(ny);
    GET_INT_PARAM(nz);

    #undef GET_INT_PARAM

    #define GET_FLOAT_PARAM(name) \
        if (document.HasMember(#name) && document[#name].IsNumber()) { \
            settings.name = document[#name].GetFloat(); \
        } else { \
            std::cerr << "error: \"" #name "\" is missing or not a number" << std::endl; \
            complete = false; \
        }

    GET_FLOAT_PARAM(thickness);
    GET_FLOAT_PARAM(E);
    GET_FLOAT_PARAM(nu);

    #undef GET_FLOAT_PARAM

    return complete;
}

// Requires a cubic grid (nx = ny = nz > 0), warns for an odd resolution, and prints the settings.
inline bool check_simulation_setting(const simulation_setting& settings)
{
    if (settings.nx <= 0 || settings.nx != settings.ny || settings.nx != settings.nz) {
        std::cerr << "error: the grid must be cubic, nx = ny = nz > 0" << std::endl;
        return false;
    }
    if (settings.nx % 2 != 0) {
        std::cerr << "warning: an odd resolution gives a non-conforming periodic tet mesh; use an even nx" << std::endl;
    }
    std::cout << "nx: " << settings.nx << std::endl;
    std::cout << "ny: " << settings.ny << std::endl;
    std::cout << "nz: " << settings.nz << std::endl;
    std::cout << "thickness: " << settings.thickness << std::endl;
    std::cout << "E: " << settings.E << std::endl;
    std::cout << "nu: " << settings.nu << std::endl;
    return true;
}
