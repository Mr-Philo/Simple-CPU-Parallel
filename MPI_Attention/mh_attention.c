#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <math.h>

#define DEBUG 0

// 全局变量
double **Wq, **Wk, **Wv;                    // 注意力矩阵声明（每一头的QKV）
double **Input, **Output;                   // 输入、输出矩阵声明（每一头的输入输出）
double **Overall_Output;                    // 最终的输出矩阵（由每一头的输出矩阵拼接而成）
int SeqLength, d_input, d_hidden;           // SeqLength:输入向量长度；d_input:输入向量维度；d_hidden:QKV隐藏层维度
int p,my_rank;                              // p:处理器个数； rank:各线程标识符
double time1, time2;

/*
 *函数名：matrix_random
 *功能：随机生成矩阵
 */
void matrix_random(double **m, int d_x, int d_y, int extra_int)
{
    int i,j;

    srand((unsigned int)time(NULL)+extra_int);     //设随机数种子

	//随机生成A,B,并初始化C
    for(i=0; i<d_x ; i++){
        for(j=0; j<d_y ; j++){
            m[i][j] = rand() / (double)(RAND_MAX) ;
	    }
    }
}


/*
 *函数名：print
 *功能：打印矩阵
 *输入：指向矩阵指针的指针，字符串
 */
void print(double **m, char *str, int d_x, int d_y)
{
   int i,j;
   printf("%s",str);
   for(i=0;i<d_x;i++)
   {
       for(j=0;j<d_y;j++)
           printf("%15.6f    ",m[i][j]);
       printf("\n");
   }
   printf("\n");
}


/*
 *函数名：allocateMatrix
 *功能：给二维矩阵分配内存
 *输入：rows:矩阵行数 cols:矩阵列数
 */
double** allocateMatrix(int rows, int cols) {  
    double** matrix = (double**)malloc(rows * sizeof(double*));  
    for (int i = 0; i < rows; i++) {  
        matrix[i] = (double*)malloc(cols * sizeof(double));  
    }  
    return matrix;  
}  

/*
 *函数名：freeMatrix
 *功能：释放二维矩阵的内存
 *输入：matrix:待释放矩阵 rows:矩阵行数
 */
void freeMatrix(double** matrix, int rows) {  
    for (int i = 0; i < rows; i++) {  
        free(matrix[i]);  
    }  
    free(matrix);  
}

/*
 *函数名：main
 *功能：主过程，计算Multi-Head Attention
 *输入：argc为命令行参数个数，argv为每个命令行参数组成的字符串数组
 */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);                  // 启动MPI计算
    MPI_Comm_size(MPI_COMM_WORLD, &p);       // 确定处理器个数
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // 确定各自的处理器标识符

    SeqLength = atoi(argv[1]);
    d_input = atoi(argv[2]);
    d_hidden = atoi(argv[3]);
    int d_head_hidden = d_hidden / p;

    // 确保多头注意力中的隐藏层能被线程数均分，否则打印错误信息，程序退出
    if (d_head_hidden*p != d_hidden)
    {
        if (my_rank == 0)
	    printf("[System Error] Number of hidden dim can not be exactly devided by num of processors!\n");
        printf(">> Usage: mpirun -np <num_of_processers> ./mh_attention <seqlength> <d_input> <d_hidden>");
        MPI_Finalize();
        exit(1);
    }

    time1 = MPI_Wtime();
    clock_t start = clock();

    // TODO: 这一部分是每一头的计算，故每个线程都要分配内存
    // 分配和初始化矩阵
    Wq = allocateMatrix(d_input, d_head_hidden);
    Wk = allocateMatrix(d_input, d_head_hidden);
    Wv = allocateMatrix(d_input, d_head_hidden);
    Input = allocateMatrix(SeqLength, d_input);
    Output = allocateMatrix(SeqLength, d_head_hidden);

    // 初始化权重矩阵和输入矩阵
    matrix_random(Wq, d_input, d_head_hidden, my_rank);
    matrix_random(Wk, d_input, d_head_hidden, my_rank);
    matrix_random(Wv, d_input, d_head_hidden, my_rank);
    matrix_random(Input, SeqLength, d_input, my_rank);
    // if(DEBUG and my_rank == 0) print(Wq,"Wq: \n", d_input, d_head_hidden);
    // if(DEBUG and my_rank == 0) print(Wk,"Wk: \n", d_input, d_head_hidden);
    // if(DEBUG and my_rank == 0) print(Wv,"Wv: \n", d_input, d_head_hidden);

    // 计算Q, K, V矩阵
    double **Q = (double **)malloc(SeqLength * sizeof(double *));
    double **K = (double **)malloc(SeqLength * sizeof(double *));
    double **V = (double **)malloc(SeqLength * sizeof(double *));
    for (int i = 0; i < SeqLength; ++i) {
        Q[i] = (double *)malloc(d_head_hidden * sizeof(double));
        K[i] = (double *)malloc(d_head_hidden * sizeof(double));
        V[i] = (double *)malloc(d_head_hidden * sizeof(double));
        for (int j = 0; j < d_head_hidden; ++j) {
            Q[i][j] = 0;
            K[i][j] = 0;
            V[i][j] = 0;
            for (int k = 0; k < d_input; ++k) {
                Q[i][j] += Input[i][k] * Wq[k][j];
                K[i][j] += Input[i][k] * Wk[k][j];
                V[i][j] += Input[i][k] * Wv[k][j];
            }
        }
    }
    // if(DEBUG and my_rank == 0) print(Q,"Q: \n", SeqLength, d_head_hidden);
    // if(DEBUG and my_rank == 0) print(K,"K: \n", SeqLength, d_head_hidden);
    // if(DEBUG and my_rank == 0) print(V,"V: \n", SeqLength, d_head_hidden);

    // 计算注意力分数并应用softmax函数
    double **attention_scores = (double **)malloc(SeqLength * sizeof(double *));
    for (int i = 0; i < SeqLength; ++i) {
        attention_scores[i] = (double *)malloc(SeqLength * sizeof(double));
        for (int j = 0; j < SeqLength; ++j) {
            attention_scores[i][j] = 0;
            for (int k = 0; k < d_head_hidden; ++k) {
                attention_scores[i][j] += Q[i][k] * K[j][k];
            }
            attention_scores[i][j] /= sqrt(d_head_hidden);
        }
    }
    // if(DEBUG and my_rank == 0) print(attention_scores,"attention_scores(stage 1): \n", SeqLength, SeqLength);
    for (int i = 0; i < SeqLength; ++i) {
        // 找到每一行中的最大分数
        double max_score = attention_scores[i][0];
        for (int j = 1; j < SeqLength; ++j) {
            if (attention_scores[i][j] > max_score) {
                max_score = attention_scores[i][j];
            }
        }
        
        // 计算分母
        double sum = 0;
        for (int j = 0; j < SeqLength; ++j) {
            sum += exp(attention_scores[i][j] - max_score); // 减去最大分数进行归一化，防止e指数溢出
        }
        
        // 计算softmax后的注意力分数
        for (int j = 0; j < SeqLength; ++j) {
            attention_scores[i][j] = exp(attention_scores[i][j] - max_score) / sum;
        }
    }
    // if(DEBUG and my_rank == 0) print(attention_scores,"attention_scores(stage 2): \n", SeqLength, SeqLength);

    // 使用注意力权重加权值矩阵V
    for (int i = 0; i < SeqLength; ++i) {
        for (int j = 0; j < d_head_hidden; ++j) {
            Output[i][j] = 0;
            for (int k = 0; k < SeqLength; ++k) {
                Output[i][j] += attention_scores[i][k] * V[k][j];
            }
        }
    }

    // 连接最终输出
    MPI_Barrier(MPI_COMM_WORLD);        // 同步所有处理器
    if (my_rank == 0) {  
        // 主进程为Overall_Output分配空间  
        Overall_Output = allocateMatrix(SeqLength, d_hidden);
    }  
    // 为拼接操作准备接收缓冲区  
    double* recv_buffer = NULL;  
    if (my_rank == 0) {  
        recv_buffer = (double*)malloc(SeqLength * d_hidden * sizeof(double));  
    }
  
    // 从每个进程的Output矩阵中拷贝数据到一维数组中以便发送  
    double* send_buffer = (double*)malloc(SeqLength * d_head_hidden * sizeof(double));  
    for (int i = 0; i < SeqLength; i++) {  
        for (int j = 0; j < d_head_hidden; j++) {  
            send_buffer[i * d_head_hidden + j] = Output[i][j];  
        }  
    }  
  
    // 使用MPI_Gather将所有进程的矩阵数据横向拼接到主进程  
    MPI_Gather(send_buffer, SeqLength * d_head_hidden, MPI_DOUBLE,  
               recv_buffer, SeqLength * d_head_hidden, MPI_DOUBLE,  
               0, MPI_COMM_WORLD);  
  
    if (my_rank == 0) {  
        // 主进程将接收到的一维数组数据转回二维矩阵格式  
        for (int i = 0; i < SeqLength; i++) {  
            for (int j = 0; j < d_hidden; j++) {  
                Overall_Output[i][j] = recv_buffer[i * d_hidden + j];  
            }  
        }  
    }

    time2 = MPI_Wtime();
    clock_t end = clock();  
    double duration = (double)(end - start) / CLOCKS_PER_SEC; 

    if (my_rank==0){
        printf("MPI multi-head attention time = %lf\n", time2 - time1);
        printf("multi-head attention time = %lf\n", duration);
        if(DEBUG) print(Input, "Multi-head Attention Input: \n", SeqLength, d_input);
        if(DEBUG) print(Overall_Output,"Multi-head Attention Output: \n", SeqLength, d_hidden);
    }


    // 释放分配的内存
    free(send_buffer);  
    if (my_rank == 0) {  
        free(recv_buffer);  
        freeMatrix(Overall_Output, SeqLength);  
    }
    freeMatrix(Wq, SeqLength);
    freeMatrix(Wk, SeqLength);
    freeMatrix(Wv, SeqLength);
    freeMatrix(Input, SeqLength);
    freeMatrix(Output, SeqLength);
    freeMatrix(Q, SeqLength);
    freeMatrix(K, SeqLength);
    freeMatrix(V, SeqLength);
    freeMatrix(attention_scores, SeqLength);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0; 
}

// mpicc mh_attention.c -o mh_attention -lm
// mpirun -np 4 ./mh_attention 4 4 8
