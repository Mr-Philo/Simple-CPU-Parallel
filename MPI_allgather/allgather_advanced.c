#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_ITERATIONS 10   // iterations of testing
#define DATA_SIZE 1024       // define the amount of data that every thread send (unit: int)

int main(int argc, char* argv[]) {
    int rank, size, i, j, k;
    int *sendbuf, *recvbuf, *recvbuf2;
    MPI_Status status;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // malloc enough space
    sendbuf = (int*)malloc(DATA_SIZE * sizeof(int));
    recvbuf = (int*)malloc(size * DATA_SIZE * sizeof(int));
    recvbuf2 = (int*)malloc(size * DATA_SIZE * sizeof(int));
    
    // initialization
    for (i = 0; i < DATA_SIZE; ++i) {
        sendbuf[i] = rank + 1;
    }

    double custom_allgather_time = 0.0;
    double official_allgather_time = 0.0;
    double start_time;

    for (k = 0; k < TEST_ITERATIONS; ++k) {
        // Custom MPI_Allgather
        start_time = MPI_Wtime();
        for (i = 0; i < size; ++i) {
            if (i == rank) {
                for (j = 0; j < size; ++j) {
                    if (j != rank) {
                        MPI_Send(sendbuf, DATA_SIZE, MPI_INT, j, 0, MPI_COMM_WORLD);
                    }
                }
                memcpy(&recvbuf[rank * DATA_SIZE], sendbuf, DATA_SIZE * sizeof(int));
            } else {
                MPI_Recv(&recvbuf[i * DATA_SIZE], DATA_SIZE, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            }
        }
        custom_allgather_time += MPI_Wtime() - start_time;

        // Official MPI_Allgather
        start_time = MPI_Wtime();
        MPI_Allgather(sendbuf, DATA_SIZE, MPI_INT, recvbuf2, DATA_SIZE, MPI_INT, MPI_COMM_WORLD);
        official_allgather_time += MPI_Wtime() - start_time;
    }

    custom_allgather_time /= TEST_ITERATIONS;
    official_allgather_time /= TEST_ITERATIONS;

    if (rank == 0) {
        printf("Average custom MPI_Allgather time: %f seconds\n", custom_allgather_time);
        printf("Average official MPI_Allgather time: %f seconds\n", official_allgather_time);
    }

    free(sendbuf);
    free(recvbuf);
    free(recvbuf2);
    MPI_Finalize();
    return 0;
}

// mpicc -o allgather_advanced allgather_advanced.c -lm
// mpirun -np 8 ./allgather_advanced
