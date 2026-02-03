//
// Created by shiting on 2026-01-22.
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <nanos6.h>
#include <chrono>
#include <iostream>

#define IDX(X, Y) (X) * size_y + (Y)
#define F(X, Y) sin(M_PI * (X) * hx) * sin(M_PI * (Y) * hy)
#define N 10

enum COLOR
{
    RED,
    BLACK
};

double computeMedian(std::vector<long>& nums) {
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

inline void init_global(double* grid_current, double* f, int nx, int ny, int size_y, double hx, double hy)
{
        // Initialize borders
#pragma oss taskloop
        for (int i = 0; i < nx+1; i++) {
            grid_current[IDX(i, 0)] = 0;
            grid_current[IDX(i, ny)] = 0;
        }
#pragma oss taskloop
        for (int i = 0; i < ny+1; i++) {
            grid_current[IDX(0, i)] = 0;
            grid_current[IDX(nx, i)] = 0;
        }
#pragma oss taskloop collapse(2)
        for (int x = 1; x < nx; x++) {
            for (int y = 1; y < ny; y++) {
                int idx = IDX(x, y);
                grid_current[idx] = 0;
                f[idx] = F(x, y);
            }
        }

}

inline void split_grid(const double* global, double* grid_red, double* grid_black, int nx, int ny) {
    int size_y = ny + 1;
#pragma oss taskloop collapse(2)
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
#pragma oss taskloop collapse(2)
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
#pragma oss taskloop collapse(2) reduction(+: global_sum)
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


inline int get_idx(int loc_i, int glb_j, int size_y, bool parity)
{
    return parity ?  loc_i * size_y + glb_j / 2 : loc_i * size_y + (glb_j - 1) / 2;
}

inline int get_global_y(int loc_j, bool parity)
{
    return parity ? loc_j * 2 : loc_j * 2 + 1;
}

inline void get_y_range(int half_y, int& start_y, int& end_y, bool parity) {
    start_y = parity ? 1 : 0;
    end_y = parity ? half_y : half_y - 1;
}

inline void update_block(double* cur_self, double* cur_neighbor, double* top_neighbor, double* bottom_neighbor,
    double* f_self, int half_y, int hb, int b, double hx, double hy, COLOR cur_color)
{
    double sum = 0;
    int start_y, end_y;

    if(top_neighbor != nullptr)
    {
        int global_i = b * hb;
        bool parity = (global_i % 2 == 0) == (cur_color == RED);
        get_y_range(half_y, start_y, end_y, parity);
        for (int j = start_y; j < end_y; j++)
        {
            int global_j = get_global_y(j, parity);

            sum = 0;
            sum -= top_neighbor[get_idx(hb - 1, global_j, half_y, parity)] / (hx*hx);
            sum -= cur_neighbor[get_idx(1, global_j, half_y, parity)] / (hx*hx);
            sum -= cur_neighbor[get_idx(0, global_j - 1, half_y, !parity)] / (hy*hy);
            sum -= cur_neighbor[get_idx(0, global_j + 1, half_y, !parity)] / (hy*hy);
            cur_self[get_idx(0, global_j, half_y, parity)] =
                (f_self[get_idx(0, global_j, half_y, parity)] - sum) / (2 / (hx*hx) + 2 / (hy*hy));
        }
    }

    for (int i = 1; i < hb - 1; i++)
    {
        int global_i = b * hb + i;
        bool parity = (global_i % 2 == 0) == (cur_color == RED);
        get_y_range(half_y, start_y, end_y, parity);
        for (int j = start_y; j < end_y; j++)
        {
            int global_j = get_global_y(j, parity);

            sum = 0;
            sum -= cur_neighbor[get_idx(i - 1, global_j, half_y, parity)] / (hx*hx);
            sum -= cur_neighbor[get_idx(i + 1, global_j, half_y, parity)] / (hx*hx);
            sum -= cur_neighbor[get_idx(i, global_j - 1, half_y, !parity)] / (hy*hy);
            sum -= cur_neighbor[get_idx(i, global_j + 1, half_y, !parity)] / (hy*hy);
            cur_self[get_idx(i, global_j, half_y, parity)] =
                (f_self[get_idx(i, global_j, half_y, parity)] - sum) / (2 / (hx*hx) + 2 / (hy*hy));
        }
    }

    if (bottom_neighbor != nullptr)
    {
        int global_i = b * hb + hb - 1;
        bool parity = (global_i % 2 == 0) == (cur_color == RED);
        get_y_range(half_y, start_y, end_y, parity);
        for (int j = start_y; j < end_y; j++)
        {
            int global_j = get_global_y(j, parity);

            sum = 0;
            sum -= cur_neighbor[get_idx(hb - 2, global_j, half_y, parity)] / (hx*hx);
            sum -= bottom_neighbor[get_idx(0, global_j, half_y, parity)] / (hx*hx);
            sum -= cur_neighbor[get_idx(hb - 1, global_j - 1, half_y, !parity)] / (hy*hy);
            sum -= cur_neighbor[get_idx(hb - 1, global_j + 1, half_y, !parity)] / (hy*hy);
            cur_self[get_idx(hb - 1, global_j, half_y, parity)] =
                    (f_self[get_idx(hb - 1, global_j, half_y, parity)] - sum) / (2 / (hx*hx) + 2 / (hy*hy));
        }
    }

}

inline void update_color(double* grid_self, double* grid_neighbor, double* f,
    int hb, int half_y, int num_blocks, double hx, double hy, COLOR color)
{
#pragma oss task inout(grid_self[0]) in(grid_neighbor[0], grid_neighbor[hb * half_y])
    update_block(&grid_self[0], &grid_neighbor[0], nullptr, &grid_neighbor[hb * half_y],
        &f[0], half_y, hb, 0, hx, hy, color);

    for (int b = 1; b < num_blocks - 1; b++)
    {
#pragma oss task inout(grid_self[b * hb * half_y]) in(grid_neighbor[b * hb * half_y], \
grid_neighbor[(b - 1) * hb * half_y], grid_neighbor[(b + 1) * hb * half_y])
        update_block(&grid_self[b * hb * half_y], &grid_neighbor[b * hb * half_y],
            &grid_neighbor[(b - 1) * hb * half_y],
            &grid_neighbor[(b + 1) * hb * half_y], &f[b * hb * half_y],
            half_y, hb, b, hx, hy, color);
    }

#pragma oss task inout(grid_self[(num_blocks - 1) * hb * half_y]) in(grid_neighbor[(num_blocks - 1) * hb * half_y], \
grid_neighbor[(num_blocks - 2) * hb * half_y])
    update_block(&grid_self[(num_blocks - 1) * hb * half_y], &grid_neighbor[(num_blocks - 1) * hb * half_y],
        &grid_neighbor[(num_blocks - 2) * hb * half_y], nullptr,
        &f[(num_blocks - 1) * hb * half_y],
               half_y, hb, num_blocks - 1, hx, hy, color);
}


void rbgs_task(const int nx, const int ny, const double hx, const double hy,
               const int num_iterations, const int num_blocks)
{
    int size_x = nx + 1;
    int size_y = ny + 1;
    //int hb = (ny - 1) / num_blocks;
    int hb = size_y / num_blocks;
    int half_y = size_y / 2;

    auto *grid_current = (double*) malloc(size_x * size_y * sizeof(double));
    auto *f = (double*) malloc(size_x * size_y * sizeof(double));

    nanos6_bitmask_t bitmask;
    nanos6_bitmask_set_wildcard(&bitmask, NUMA_ALL);
    size_t numa_nodes = nanos6_count_setbits(&bitmask);
    size_t size = size_x * half_y * sizeof(double);
    size_t block_size = size/numa_nodes;

    auto *grid_red = (double *)nanos6_numa_alloc_block_interleave(size, &bitmask, block_size);
    auto *grid_black = (double *)nanos6_numa_alloc_block_interleave(size, &bitmask, block_size);

    auto *f_red = (double *)nanos6_numa_alloc_block_interleave(size, &bitmask, block_size);
    auto *f_black = (double *)nanos6_numa_alloc_block_interleave(size, &bitmask, block_size);

    std::vector<long> total(N, 0.0);

    for (int i = 0; i < N; i++)
    {
        init_global(grid_current, f, nx, ny, size_y, hx, hy);
        //double ini_res = calculate_residual(grid_current, nx, ny, hx, hy);

        split_grid(grid_current, grid_red, grid_black, nx, ny);
        split_grid(f, f_red, f_black, nx, ny);

        std::chrono::steady_clock::time_point begin_time = std::chrono::steady_clock::now();

        for (int it = 0; it < num_iterations; it++)
        {
            update_color(grid_red, grid_black, f_red, hb, half_y, num_blocks, hx, hy, RED);

            update_color(grid_black, grid_red, f_black, hb, half_y, num_blocks, hx, hy, BLACK);
        }
#pragma oss taskwait

        std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();

        //merge_grids(grid_current, grid_red, grid_black, nx, ny);
        //print_matrix(grid_current, size_x, size_y);

        std::cout << "Time (OSS Task) = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - begin_time).count()
                      << std::endl;
        total[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - begin_time).count();
        //printf("Relative Residual: %lf \n", calculate_residual(grid_current, nx, ny, hx, hy)/ini_res);
    }

    std::cout << "median of total time " << computeMedian(total) << std::endl;

    free(grid_current);
    free(f);
    nanos6_numa_free(grid_red);
    nanos6_numa_free(grid_black);
    nanos6_numa_free(f_red);
    nanos6_numa_free(f_black);
}
