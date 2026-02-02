cmake_minimum_required(VERSION 4.1)
project(colored_gauss_seidel)

set(CMAKE_CXX_STANDARD 20)

FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    MESSAGE("openmp found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

add_executable(colored_gauss_seidel main.cpp rb_ompfor.cpp rb_task.cpp)

target_compile_options(colored_gauss_seidel PRIVATE
        -Wall
        -pedantic
        -O3
        -ffast-math
        -fopenmp
        -DOPENMP
        -Wfatal-errors
)