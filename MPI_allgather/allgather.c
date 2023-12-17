#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[]) {
    int rank, size, i, j;
    int *sendbuf, *recvbuf, *recvbuf2;
    MPI_Status status;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sendbuf = (int*)malloc(sizeof(int));
    recvbuf = (int*)malloc(size * sizeof(int));
    recvbuf2 = (int*)malloc(size * sizeof(int));
    *sendbuf = rank + 1;

    // Custom MPI_Allgather
    double start_time = MPI_Wtime();
    for (i = 0; i < size; i++) {
        if (i == rank) {
            // Broadcast own data to all other processes
            for (j = 0; j < size; j++) {
                if (j != rank) {
                    MPI_Send(sendbuf, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
                }
            }
            // Copy own data to own slot in recvbuf
            recvbuf[i] = *sendbuf;
        } else {
            // Receive data from process i
            MPI_Recv(&recvbuf[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
        }
    }
    double custom_allgather_time = MPI_Wtime() - start_time;

    // Official MPI_Allgather
    start_time = MPI_Wtime();
    MPI_Allgather(sendbuf, 1, MPI_INT, recvbuf2, 1, MPI_INT, MPI_COMM_WORLD);
    double official_allgather_time = MPI_Wtime() - start_time;

    // Print results
    if (rank == 0) {
        printf("Custom MPI_Allgather time: %f seconds\n", custom_allgather_time);
        printf("Official MPI_Allgather time: %f seconds\n", official_allgather_time);
    }

    // Verify results
    for (i = 0; i < size; i++) {
        if (recvbuf[i] != recvbuf2[i]) {
            fprintf(stderr, "Mismatch in gathered data at rank %d!\n", rank);
            break;
        }
    }
    if (rank == 0) {printf("Verify results passed!\n");}

    free(sendbuf);
    free(recvbuf);
    free(recvbuf2);
    MPI_Finalize();
    return 0;
}


// mpicc -o allgather allgather.c -lm
// mpirun -np 8 ./allgather
