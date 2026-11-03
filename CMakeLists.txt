cmake_minimum_required(VERSION 3.24)

# Build for the GPU(s) of this machine unless an architecture list is given,
# e.g. -DCMAKE_CUDA_ARCHITECTURES="90;120".
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(ShellHomo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

function(add_shellhomo_executable name source)
  add_executable(${name} ${source})
  set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  target_compile_options(${name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  )
  target_link_libraries(${name} PRIVATE CUDA::cudart Threads::Threads)
endfunction()

# forward homogenization of implicit fields
add_shellhomo_executable(shellhomo src/main.cu)
# inverse design in the combined-TPMS space
add_shellhomo_executable(shellhomo_inverse src/inverse_design.cu)
