#include <cstdio>
#include <cstdlib>
#include <cmath>
//#include <omp.h>
//#include "helper.h"

#ifndef USE_FOR
    #define USE_FOR 1
#endif

#if USE_FOR == 1
    #include "rb_ompfor.h"
#else
    #include "rb_task.h"
#endif

#define USAGE "Usage:\n\trbgs nx ny t c b\n\nWhere:\n \
 nx,ny\tnumber of discretization intervals in x and y axis, repectively\n \
 c\tnumber of iterations\n \
 b\tnumber of blocks\n"

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, USAGE);
        exit(1);
    }

    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);
    int num_iterations = atoi(argv[3]);
    int num_blocks = atoi(argv[4]);
    double hx = 1 / (double) nx;
    double hy = 1 / (double) ny;

    if ((nx < 1) || (ny < 1)) {
        fprintf(stderr, "error: %dx%d are not valid dimensions for discretization\n", nx, ny);
        exit(1);
    }

    if (num_iterations < 0) {
        fprintf(stderr, "error: %d is not a valid number of iterations\n", num_iterations);
        exit(1);
    }

    //omp_set_num_threads(num_threads);

    // calculate
#if USE_FOR == 1
    gauss_v1(nx, ny, hx, hy, num_iterations);
#else
    rbgs_task(nx, ny, hx, hy, num_iterations, num_blocks);
#endif
    exit(0);
}
