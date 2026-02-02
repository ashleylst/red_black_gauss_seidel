//
// Created by shiting on 2026-01-22.
//

#ifndef COLORED_GAUSS_SEIDEL_HELPER_H
#define COLORED_GAUSS_SEIDEL_HELPER_H

#include <cmath>
#include <iostream>

#define F(X, Y) sin(M_PI * (X) * hx) * sin(M_PI * (Y) * hy)

// Grid to array conversion functions
#define IDX(X, Y) (X) * size_y + (Y)
#define IDX2(X, Y) ((int) ((X) / 2.0)) * size_y + (Y)

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


inline void print_matrix(double* arr, int x, int y)
{
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            std::cout << arr[i * y + j] << " ";
        }
        std::cout << std::endl;
    }
}
#endif //COLORED_GAUSS_SEIDEL_HELPER_H