//
// Created by shiting on 2026-01-22.
//
#include "helper.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>

enum COLOR
{
    RED,
    BLACK
};

inline int get_start_idx(COLOR color, int idx)
{
    if (color == RED)
        return (idx % 2 == 0) ? 2 : 1;
    return (idx % 2 == 0) ? 1 : 2;
}

inline void update(double* current, double* top, double* bottom, const double* f,
                       const int hb, const int ny, const int b,
                       const double hx, const double hy, const COLOR color)
{
    int size_y = ny + 1;
    int row_ind = b * hb + 1;
    int first_y = get_start_idx(color, row_ind);
    for (int j = first_y; j < ny; j += 2)
    {
        double sum = 0;
        if (top != nullptr)
            sum -= top[(hb-1) * size_y + j] / (hx*hx);
        sum -= current[size_y + j] / (hx*hx);
        sum -= current[j - 1] / (hy*hy);
        sum -= current[j + 1] / (hy*hy);

        current[j] = (f[j] - sum) / (2 / (hx*hx) + 2 / (hy*hy));
    }

    for (int i = 1; i < hb - 1; i++)
    {
        row_ind = b * hb + 1 + i;
        first_y = get_start_idx(color, row_ind);
        for (int j = first_y; j < ny; j += 2)
        {
            double sum = 0;
            sum -= current[IDX(i - 1, j)] / (hx*hx);
            sum -= current[IDX(i + 1, j)] / (hx*hx);
            sum -= current[IDX(i, j - 1)] / (hy*hy);
            sum -= current[IDX(i, j + 1)] / (hy*hy);

            current[IDX(i, j)] = (f[IDX(i, j)] - sum) /
                                (2 / (hx*hx) + 2 / (hy*hy) );
        }
    }

    row_ind = b * hb + hb;
    first_y = get_start_idx(color, row_ind);
    for (int j = first_y; j < ny; j += 2)
    {
        double sum = 0;
        sum -= current[(hb - 2) * size_y + j]/ (hx*hx);
        if (bottom!= nullptr)
        {
            sum -= bottom[j] / (hx*hx);
        }
        sum -= current[IDX(hb - 1, j - 1)] / (hy*hy);
        sum -= current[IDX(hb - 1, j + 1)] / (hy*hy);

        current[IDX(hb - 1, j)] = (f[IDX(hb - 1, j)] - sum) /
                                (2 / (hx*hx) + 2 / (hy*hy) );
    }
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
#pragma omp task depend(inout: grid_self[0]) depend(in: grid_neighbor[0], grid_neighbor[hb * half_y])
    update_block(&grid_self[0], &grid_neighbor[0], nullptr, &grid_neighbor[hb * half_y],
        &f[0], half_y, hb, 0, hx, hy, color);

    for (int b = 1; b < num_blocks - 1; b++)
    {
#pragma omp task depend(inout: grid_self[b * hb * half_y]) depend(in: grid_neighbor[b * hb * half_y], \
grid_neighbor[(b - 1) * hb * half_y], grid_neighbor[(b + 1) * hb * half_y])
        update_block(&grid_self[b * hb * half_y], &grid_neighbor[b * hb * half_y],
            &grid_neighbor[(b - 1) * hb * half_y],
            &grid_neighbor[(b + 1) * hb * half_y], &f[b * hb * half_y],
            half_y, hb, b, hx, hy, color);
    }

#pragma omp task depend(inout: grid_self[(num_blocks - 1) * hb * half_y]) depend(in: grid_neighbor[(num_blocks - 1) * hb * half_y], \
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

    init_global(grid_current, f, nx, ny, size_y, hx, hy);
    double ini_res = calculate_residual(grid_current, nx, ny, hx, hy);

    auto *grid_red = (double*) malloc(size_x * half_y * sizeof(double));
    auto *grid_black = (double*) malloc(size_x * half_y * sizeof(double));
    split_grid(grid_current, grid_red, grid_black, nx, ny);

    auto *f_red = (double*) malloc(size_x * half_y * sizeof(double));
    auto *f_black = (double*) malloc(size_x * half_y * sizeof(double));
    split_grid(f, f_red, f_black, nx, ny);

    double begin_time = omp_get_wtime();

#pragma omp parallel
    {
#pragma omp single
        {
            for (int it = 0; it < num_iterations; it++)
            {
                update_color(grid_red, grid_black, f_red, hb, half_y, num_blocks, hx, hy, RED);

                update_color(grid_black, grid_red, f_black, hb, half_y, num_blocks, hx, hy, BLACK);
            }
        }
    }

    double end_time = omp_get_wtime();

    /*
    *for (int it = 0; it < num_iterations; it++)
    {
        update(&grid_current[size_y], nullptr,
                &grid_current[size_y + size_y * hb], &f[size_y],
                       hb, ny, 0, hx, hy, RED);
        for (int b = 1; b < num_blocks - 1; b++)
        {
            update(&grid_current[size_y + b * size_y * hb], &grid_current[size_y + (b - 1) * size_y * hb],
                &grid_current[size_y + (b + 1) * size_y * hb], &f[size_y + b * size_y * hb],
                       hb, ny, b, hx, hy, RED);
        }
        update(&grid_current[size_y + (num_blocks - 1) * size_y * hb],
                   &grid_current[size_y + (num_blocks - 2) * size_y * hb],
                nullptr, &f[size_y + (num_blocks - 1) * size_y * hb],
                       hb, ny, num_blocks - 1, hx, hy, RED);

        update(&grid_current[size_y], nullptr,
                &grid_current[size_y + size_y * hb], &f[size_y],
                       hb, ny, 0, hx, hy, BLACK);
        for (int b = 1; b < num_blocks - 1; b++)
        {
            update(&grid_current[size_y + b * size_y * hb], &grid_current[size_y + (b - 1) * size_y * hb],
                &grid_current[size_y + (b + 1) * size_y * hb], &f[size_y + b * size_y * hb],
                       hb, ny, b, hx, hy, BLACK);
        }
        update(&grid_current[size_y + (num_blocks - 1) * size_y * hb],
                   &grid_current[size_y + (num_blocks - 2) * size_y * hb],
                nullptr, &f[size_y + (num_blocks - 1) * size_y * hb],
                       hb, ny, num_blocks - 1, hx, hy, BLACK);


    }
    for (int i = 0; i < size_x * size_y/2; i++)
    {
        std::cout << grid_red[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < size_x * size_y/2; i++)
    {
        std::cout << grid_black[i] << " ";
    }
    std::cout << std::endl;
    merge_grids(grid_current, grid_red, grid_black, nx, ny);*/
    merge_grids(grid_current, grid_red, grid_black, nx, ny);
    //print_matrix(grid_current, size_x, size_y);

    printf("Total Elapsed Time: %lf s\n", end_time-begin_time);
    printf("Relative Residual: %lf \n", calculate_residual(grid_current, nx, ny, hx, hy)/ini_res);

    free(grid_current);
    free(f);
    free(grid_red);
    free(grid_black);
    free(f_red);
    free(f_black);
}