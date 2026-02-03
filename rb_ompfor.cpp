//
// Created by shiting on 2026-01-22.
//
#include "helper.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <chrono>

void gauss_v2(const int nx, const int ny, const double hx, const double hy, const int num_iterations) {
    int idx;
    double sum;
    auto *grid_current = (double*) malloc((nx+1) * (ny+1) * sizeof(double));

    const int size_x = (int)ceil((nx + 1) / 2.0);
    const int size_y = ny + 1;

    auto *grid_red = (double*)malloc(size_x * size_y * sizeof(double));
    auto *grid_black = (double*)malloc(size_x * size_y * sizeof(double));
    auto *f_red = (double*)malloc(size_x * size_y * sizeof(double));
    auto *f_black = (double*)malloc(size_x * size_y * sizeof(double));

    // initialize the grids
    init(grid_red, grid_black, size_y, f_red, f_black, nx, ny, hx, hy);
    combine_red_black(grid_current, grid_red, grid_black, size_y, nx, ny);
    double ini_res = calculate_residual(grid_current, nx, ny, hx, hy);

    double begin_time = omp_get_wtime();

    #pragma omp parallel private(idx, sum)
    {
        // start iterative solve
        for (int it = 0; it < num_iterations; it++) {
            // do red (even)
            #pragma omp for
            for (int i = 1; i < nx; i++) {
                int first_y = (i % 2 == 0) ? 2 : 1;
                for (int j = first_y; j < ny; j += 2) {
                    idx = IDX2(i, j);

                    sum = 0;

                    if (i % 2 == 0) {
                        // top neighbor
                        sum -= grid_black[idx-size_y] / (hx*hx);
                    }
                    else {
                        // bottom neighbor
                        sum -= grid_black[idx+size_y] / (hx*hx);
                    }

                    sum -= grid_black[idx] / (hx*hx);
                    sum -= grid_black[idx-1] / (hy*hy);
                    sum -= grid_black[idx+1] / (hy*hy);

                    grid_red[idx] = (f_red[idx] - sum) /
                        (2 / (hx*hx) + 2 / (hy*hy) );
                }
            }

            #pragma omp for
            for (int i = 1; i < nx; i++) {
                int first_y = (i % 2 == 0) ? 1 : 2;
                for (int j = first_y; j < ny; j += 2) {
                    idx = IDX2(i, j);

                    sum = 0;

                    if (i % 2 == 0) {
                        // top neighbor
                        sum -= grid_red[idx-size_y] / (hx*hx);
                    }
                    else {
                        // bottom neighbor
                        sum -= grid_red[idx+size_y] / (hx*hx);
                    }

                    sum -= grid_red[idx] / (hx*hx);
                    sum -= grid_red[idx-1] / (hy*hy);
                    sum -= grid_red[idx+1] / (hy*hy);

                    grid_black[idx] = (f_black[idx] - sum) /
                        (2 / (hx*hx) + 2 / (hy*hy) );
                }
            }

        }
    }

    double end_time = omp_get_wtime();

    // Copy solution from grid_red/grid_black into grid_current
    combine_red_black(grid_current, grid_red, grid_black, size_y, nx, ny);

    // Calculate residue and output solution
    printf("Total Elapsed Time: %lf s\n", end_time-begin_time);
    printf("Relative Residual: %lf \n", calculate_residual(grid_current, nx, ny, hx, hy)/ini_res);

    free(grid_current);
    free(grid_red);
    free(grid_black);
    free(f_red);
    free(f_black);
}

void gauss_v1(const int nx, const int ny, const double hx, const double hy, const int num_iterations) {
    int it, x, y, yy;
    int size_x, size_y;
    double sum, ini_res;
  
    double *f;
    double *grid_current;

    size_x = nx+1;
    size_y = ny+1;

    grid_current = (double*) malloc(size_x * size_y * sizeof(double));
    f = (double*) malloc(size_x * size_y * sizeof(double));


    init_global(grid_current, f, nx, ny, size_y, hx, hy);
    ini_res = calculate_residual(grid_current, nx, ny, hx, hy);

    //print_matrix(f, size_x, size_y);

    std::chrono::steady_clock::time_point begin_time = std::chrono::steady_clock::now();

    #pragma omp parallel private(it, x, y, yy, sum)
    {
        for (it=0; it<num_iterations; ++it) {
#pragma omp for
            for (x=1; x<nx; ++x) {
                for (yy=1; yy<ny; yy+=2) {
                    if (x % 2 == 0)
                        y = yy + 1;
                    else
                        y = yy;

                    if (y >= ny)
                        continue;

                    sum = 0;

                    sum -= grid_current[IDX(x-1, y)] / (hx*hx);
                    sum -= grid_current[IDX(x+1, y)] / (hx*hx);
                    sum -= grid_current[IDX(x, y-1)] / (hy*hy);
                    sum -= grid_current[IDX(x, y+1)] / (hy*hy);

                    grid_current[IDX(x, y)] = (f[IDX(x, y)] - sum) /
                                        (2 / (hx*hx) + 2 / (hy*hy) );
                }
            }

            #pragma omp for
            for (x=1; x<nx; ++x) {
                for (yy=1; yy<ny; yy+=2) {
                    if (x % 2 == 0)
                        y = yy;
                    else
                        y = yy + 1;

                    if (y >= ny)
                        continue;

                    sum = 0;

                    sum -= grid_current[IDX(x-1, y)] / (hx*hx);
                    sum -= grid_current[IDX(x+1, y)] / (hx*hx);
                    sum -= grid_current[IDX(x, y-1)] / (hy*hy);
                    sum -= grid_current[IDX(x, y+1)] / (hy*hy);

                    grid_current[IDX(x, y)] = (f[IDX(x, y)] - sum) /
                                        (2 / (hx*hx) + 2 / (hy*hy) );
                }
            }


        }
    }

    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();

    //print_matrix(grid_current, size_x, size_y);

    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - begin_time).count()
                  << std::endl;
    //printf("Total Elapsed Time: %lf s\n", end_time-begin_time);
    printf("Relative Residual: %lf\n", calculate_residual(grid_current, nx, ny, hx, hy)/ini_res);

    free(grid_current);
    free(f);
}
