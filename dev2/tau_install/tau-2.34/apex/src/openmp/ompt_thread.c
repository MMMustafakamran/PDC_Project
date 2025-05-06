#include <unistd.h>
#include <stdio.h>
#include <omp.h>
#include "apex.h"

int main (int argc, char** argv) {
    apex_set_use_screen_output(1);
#pragma omp parallel
    {
        printf("Hello from thread %d of %d\n",
            omp_get_thread_num(),
            omp_get_num_threads());
        fflush(stdout);
    }
    return 0;
}

