#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define P 2                 // 服务器进程数量P，进而可以计算工作进程数量为总数量-P
#define ROUND 10            // 进行模拟数据交换的轮数
#define DEBUG 0

int main(int argc, char* argv[]) {

    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int Q = size - P;       // 工作进程数量
    srand(rank);            // 不设置该种子的话，每个线程send的随机数都一样，因为本质是伪随机数，不设置种子的话都默认是1
    
    // 在该程序中划分两种通信域：一是按每个参数服务器对应分组，二是把服务器线程分为一组、工作进程氛围一组
    MPI_Comm param_comm, server_comm;
    MPI_Comm_split(MPI_COMM_WORLD, (rank % P), rank, &param_comm);
    MPI_Comm_split(MPI_COMM_WORLD, ((rank < P) ? 1 : 0), rank, &server_comm);

    // 各服务器组内的rank和size
    int p_rank, p_size;
    MPI_Comm_rank(param_comm, &p_rank);
    MPI_Comm_size(param_comm, &p_size);

    // 开始模拟参数服务器过程
    int i,j;
    float value, sum, average;

    for (i=0; i<ROUND; i++){
        if(DEBUG && (rank == 0)){ printf("--------ROUND %d-----------\n", i+1); }

        if (p_rank != 0){       // 每个分组内rank为0才表示服务器进程，不为0时表示工作进程
            value = (float)rand() / RAND_MAX;       // 每个工作进程产生一个0-1的随机浮点数
            if(DEBUG){ printf("Work thread %d send value: %.4f\n", rank, value); }
        }
        else{
            value = 0.0;
        }

        MPI_Barrier(MPI_COMM_WORLD);      // 这里为了方便起见，假设所有线程发送完再执行参数服务器的聚集计算

        // 在每个服务器域内聚集各工作进程的数据
        MPI_Reduce(&value, &sum, 1, MPI_FLOAT, MPI_SUM, 0, param_comm);     // 关键参数是指定p_rank=0，即reduce至服务器线程rank0内
        if (DEBUG && (p_rank == 0)){ printf("Server thread %d got sum: %.4f\n", rank, sum);} 

        // 在参数服务器内部聚集数据
        MPI_Allreduce(&sum, &sum, 1, MPI_FLOAT, MPI_SUM, server_comm);      // 在server通信域内all_reduce数据，操作后各个参数服务器内都是sum的总和
        if (p_rank == 0){
            average = sum / Q;      // 分母是工作进程总数
        }
        if (DEBUG && (p_rank == 0)){ printf("Server thread %d computed sum %.4f, average: %.4f\n", rank, sum, average);} 

        MPI_Barrier(MPI_COMM_WORLD);

        // 广播回工作进程
        MPI_Bcast(&average, 1, MPI_FLOAT, 0, param_comm);
        if (DEBUG && (p_rank != 0)){ printf("Work thread %d got value: %.4f\n", rank, average); }
        
        if ( !DEBUG && (rank == 0)){
            printf("%d round: broadcast value %.4f\n", (i+1), average);
        }
    }

    MPI_Comm_free(&param_comm);
    MPI_Comm_free(&server_comm);
    MPI_Finalize();
    return 0;
}

// abandoned code
int custom_main(int argc, char* argv[]) {

    int rank, size;
    int Q = size - P;       // 工作进程数量
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;
    
    // 开始模拟参数服务器过程
    int i,j;
    float value, sum, average, all_average;

    for ( i = 0; i < ROUND; i++){
        if(DEBUG){ printf("--------DOUND %d-----------\n", i); }

        // 首先是数据收集过程
        if (rank >= P){       // 大于等于P时表示工作进程
            value = (float)rand() / RAND_MAX;       // 每个工作进程产生一个0-1的随机浮点数
            int server_rank = rank % P;  
            MPI_Send(&value, 1, MPI_FLOAT, server_rank, 0, MPI_COMM_WORLD); 
            if(DEBUG){ printf("Work thread %d send value: %.2f\n", rank, value); }
        }
        else{   // 以下内容是在参数服务器进程内完成的
            sum = 0.0;  
            for (j = 0; j < Q; j++) {  
                MPI_Recv(&value, 1, MPI_FLOAT, j + P, 0, MPI_COMM_WORLD, &status);  // 从rank=j+P的各工作进程收集数据
                sum += value;  
            }
            average = sum / Q;
            if(DEBUG){ printf("Server thread %d got sum: %.2f, average: %.2f\n", rank, sum, average);} 
        }

        // 接下来，同步所有参数服务器进程以获得全局平均值 
        if (rank == 0){
            all_average = 0.0 + average;    // 参数服务器的主进程0
            for (j = 0; j < P; j++){
                // MPI_Recv(&)     // TODO:
            }
        }

        // 发送最终的全局平均值回对应的工作进程  
        for (j = 0; j < Q; j++) {  
            MPI_Send(&average, 1, MPI_FLOAT, j + P, 0, MPI_COMM_WORLD);  
        }

        if (rank == 0){
            printf("%d round: broadcast value %.2f\n", (i+1), value);
        }
    }

    MPI_Finalize();
    return 0;
}

// mpicc -o param_server param_server.c
// mpirun -np 8 ./param_server
