cmake_minimum_required(VERSION 3.16)
project(blqh LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# oneMKL, LP64, GNU OpenMP. Works for Intel and for AMD EPYC with MKL 2025.
set(MKLROOT "/opt/intel/oneapi/mkl/2025.2" CACHE PATH "oneMKL root")

add_executable(fqhe
    src/sector.cpp
    src/hamiltonian.cpp
    src/solve.cpp
    src/observe.cpp
    src/check.cpp
    src/cli.cpp
)
target_include_directories(fqhe PRIVATE include ${MKLROOT}/include)
target_compile_definitions(fqhe PRIVATE BLQH_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_compile_options(fqhe PRIVATE -O3 -fopenmp -march=native)
target_link_directories(fqhe PRIVATE ${MKLROOT}/lib)
target_link_options(fqhe PRIVATE -fopenmp -Wl,-rpath,${MKLROOT}/lib -Wl,--no-as-needed)
target_link_libraries(fqhe PRIVATE mkl_intel_lp64 mkl_gnu_thread mkl_core gomp pthread m dl)
