//
// Created by shiting on 2026-01-22.
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <iostream>

#define IDX(X, Y) (X) * size_y + (Y)
#define F(X, Y) sin(M_PI * (X) * hx) * sin(M_PI * (Y) * hy)
#define IDX2(X, Y) ((int) ((X) / 2.0)) * size_y + (Y)
#define N 10

inline double computeMedian(std::vector<long>& nums) {
    if (nums.empty()) {
        return 0.0;  // Handle empty vector case
    }

    // Make a copy to avoid modifying the original vector
    std::vector<long> sorted = nums;
    std::sort(sorted.begin(), sorted.end());

    size_t n = sorted.size();
    if (n % 2 == 0) {  // Even number of elements
        return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
    } else {  // Odd number of elements
        return sorted[n/2];
    }
}

inline void combine_red_black(double* dest, const double* red, const double* black, int size_y, int nx, int ny)
{
    int x;
    for (int xx=0; xx<nx+1; xx+=2) {
        for (int y=0; y<ny+1; ++y) {
            if (y % 2 == 0)
                x = xx;
            else
                x = xx + 1;

            if (x <= nx)
                dest[IDX(x, y)] = red[IDX2(x, y)];

            if (y % 2 == 0)
                x = xx + 1;
            else
                x = xx;

            if (x <= nx)
                dest[IDX(x, y)] = black[IDX2(x, y)];
        }
    }
}

inline void init_global(double* grid_current, double* f, int nx, int ny, int size_y, double hx, double hy)
{

#pragma omp parallel
    {
        // Initialize borders
#pragma omp for
        for (int i = 0; i < nx+1; i++) {
            grid_current[IDX(i, 0)] = 0;
            grid_current[IDX(i, ny)] = 0;
        }
#pragma omp for
        for (int i = 0; i < ny+1; i++) {
            grid_current[IDX(0, i)] = 0;
            grid_current[IDX(nx, i)] = 0;
        }

#pragma omp for
        for (int x = 1; x < nx; x++) {
            for (int y = 1; y < ny; y++) {
                int idx = IDX(x, y);
                grid_current[idx] = 0;
                f[idx] = F(x, y);
            }
        }
    }
}

inline void init(double* grid_red, double* grid_black, int size_y, double* f_red, double* f_black,
                 int nx, int ny, double hx, double hy)
{
    int idx;
#pragma omp parallel private(idx)
    {
        // Proper parallel initialization of grid and f
        // 1. BLACK SITES (i + j is odd)
#pragma omp for
        for (int i = 1; i < nx; i++) {
            int first_y = (i % 2 == 0) ? 1 : 2;
            for (int j = first_y; j < ny; j += 2) {

                // Set boundary values in the RED array (the neighbor array)
                if (i == 1)      grid_red[IDX2(i-1, j)] = 0;
                if (i == nx-1)   grid_red[IDX2(i+1, j)] = 0;
                if (j == 1)      grid_red[IDX2(i, j-1)] = 0;
                if (j == ny-1)   grid_red[IDX2(i, j+1)] = 0;

                idx = IDX2(i, j);
                grid_black[idx] = 0;
                f_black[idx] = F(i, j);
            }
        }

        // 2. RED SITES (i + j is even)
        // We update grid_red and f_red here.
#pragma omp for
        for (int i = 1; i < nx; i++) {
            int first_y = (i % 2 == 0) ? 2 : 1;
            for (int j = first_y; j < ny; j += 2) {

                // Set boundary values in the BLACK array (the neighbor array)
                if (i == 1)      grid_black[IDX2(i-1, j)] = 0;
                if (i == nx-1)   grid_black[IDX2(i+1, j)] = 0;
                if (j == 1)      grid_black[IDX2(i, j-1)] = 0;
                if (j == ny-1)   grid_black[IDX2(i, j+1)] = 0;

                idx = IDX2(i, j);
                grid_red[idx] = 0;
                f_red[idx] = F(i, j);
            }
        }
    }
}

inline void split_grid(const double* global, double* grid_red, double* grid_black, int nx, int ny) {
    int size_y = ny + 1;

#pragma omp parallel for
    for (int i = 0; i <= nx; i++) {
        for (int j = 0; j <= ny; j++) {
            if ((i + j) % 2 == 0) {
                grid_red[(i * size_y + j) / 2]   = global[i * size_y + j];
            } else {
                grid_black[(i * size_y + j) / 2] = global[i * size_y + j];
            }
        }
    }
}

inline void merge_grids(double* global, const double* grid_red, const double* grid_black, int nx, int ny) {
    int size_y = ny + 1;

    for (int i = 0; i <= nx; i++) {
        for (int j = 0; j <= ny; j++) {
            if ((i + j) % 2 == 0) {
                global[i * size_y + j] = grid_red[(i * size_y + j) / 2];
            } else {
                global[i * size_y + j] = grid_black[(i * size_y + j) / 2];
            }
        }
    }
}

inline double calculate_residual(const double* arr, int nx, int ny, double hx, double hy) {
    double global_res = 0.0;
    int x, y;

    // Use OpenMP reduction to sum the squared errors across all threads
#pragma omp parallel for private(x, y) reduction(+:global_res)
    for (x = 1; x < nx; x++) {
        for (y = 1; y < ny; y++) {
            double local_sum = 0.0;

            // Discrete Laplacian operator: (u_i-1 + u_i+1)/hx^2 + (u_j-1 + u_j+1)/hy^2
            local_sum -= (arr[(x - 1) * (ny + 1) + y] +
                          arr[(x + 1) * (ny + 1) + y]) / (hx * hx);
            local_sum -= (arr[x * (ny + 1) + (y - 1)] +
                          arr[x * (ny + 1) + (y + 1)]) / (hy * hy);

            // Center point: 2u/hx^2 + 2u/hy^2
            local_sum += arr[x * (ny + 1) + y] * (2.0 / (hx * hx) + 2.0 / (hy * hy));

            // Difference from source function
            double r = F(x, y) - local_sum;
            global_res += r * r;
        }
    }

    return sqrt(global_res);
}

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

    std::vector<long> total(N, 0.0);

    for (int i = 0 ; i < N; i++)
    {
        init_global(grid_current, f, nx, ny, size_y, hx, hy);
        //ini_res = calculate_residual(grid_current, nx, ny, hx, hy);

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

        std::cout << "Time (Parallel for) = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - begin_time).count()
                      << std::endl;
        total[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - begin_time).count();
        //printf("Total Elapsed Time: %lf s\n", end_time-begin_time);
        //printf("Relative Residual: %lf\n", calculate_residual(grid_current, nx, ny, hx, hy)/ini_res);
    }
    std::cout << "median of total time " << computeMedian(total) << std::endl;

    free(grid_current);
    free(f);
}