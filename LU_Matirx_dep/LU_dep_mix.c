#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"
#include "time.h"
#include "omp.h"

#define a(x,y) a[x*M+y]
/*A为M*M矩阵*/
#define A(x,y) A[x*M+y]
#define l(x,y) l[x*M+y]
#define u(x,y) u[x*M+y]
#define floatsize sizeof(float)
#define intsize sizeof(int)

int M,N;
int m;
float *A;
int my_rank;
int p;
MPI_Status status;

void fatal(char *message)
{
    printf("%s\n",message);
    exit(1);
}

void Environment_Finalize(float *a,float *f)
{
    free(a);
    free(f);
}

// 生成并返回一个满秩的随机方阵  
void generate_full_rank_matrix(int n, float* A) {  
    // 生成随机矩阵，但保证对角线元素较大，以保持矩阵满秩  
    for (int i = 0; i < n; ++i) {  
        for (int j = 0; j < n; ++j) {  
            if (i == j) {  
                A[i*n + j] = (float)rand() / RAND_MAX * 10.0 + n; // 对角线上元素较大，确保不为零  
            } else {  
                A[i*n + j] = (float)rand() / RAND_MAX * 10.0; // 非对角线元素为较小的随机数  
            }  
        }  
    }  
    return;  
}

int main(int argc, char **argv)
{   
    clock_t overall_start = clock();

    int i,j,k,my_rank,group_size;
    int i1,i2;
    int v,w;
    float *a,*f,*l,*u;
    FILE *fdA;

    // MPI初始化
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    // 指定OpenMP线程数
    int num_of_threads = atoi(argv[2]);     // OpenMP线程数
    omp_set_num_threads(num_of_threads);

    p=group_size;

    // 交叉划分必须保证矩阵的阶数能被MPI节点数整除
    M = atoi(argv[1]);      // 命令行指定阶数
    N = M;
    int sp = (int)(M/p);
    if (sp*p != M)
    {
        if (my_rank == 0)
        printf("The dimension of the matrix must be divisible by the number of MPI nodes!\n");
        MPI_Finalize();
        exit(1);
    }

    if (my_rank==0)
    {   
        // 按阅读文件的方式读取矩阵
        // fdA=fopen("dataIn.txt","r");
        // fscanf(fdA,"%d %d", &M, &N);
        // if(M != N)
        // {
        //     puts("The input is error!");
        //     exit(0);
        // }
        // A=(float *)malloc(floatsize*M*M);
        // for(i = 0; i < M; i ++)
        //     for(j = 0; j < M; j ++)
        //         fscanf(fdA, "%f", A+i*M+j);
        // fclose(fdA);
        
        // 生成一个随机的更大的矩阵
        // srand(time(NULL));      // 初始化随机数种子
        srand(0);
        A = (float *)malloc(floatsize*M*M);
        generate_full_rank_matrix(M, A);
    }

    /*0号进程将M广播给所有进程*/
    MPI_Bcast(&M,1,MPI_INT,0,MPI_COMM_WORLD);
    m=M/p;
    if (M%p!=0) m++;

    /*分配至各进程的子矩阵大小为m*M*/
    a=(float*)malloc(floatsize*m*M);

    /*各进程为主行元素建立发送和接收缓冲区*/
    f=(float*)malloc(floatsize*M);

    /*0号进程为l和u矩阵分配内存，以分离出经过变换后的A矩阵中的l和u矩阵*/
    if (my_rank==0)
    {
        l=(float*)malloc(floatsize*M*M);
        u=(float*)malloc(floatsize*M*M);
    }

    /*0号进程采用行交叉划分将矩阵A划分为大小m*M的p块子矩阵，依次发送给1至p-1号进程*/
    if (a==NULL) fatal("allocate error\n");

    if (my_rank==0)
    {
        for(i=0;i<m;i++)
            for(j=0;j<M;j++)
                a(i,j)=A((i*p),j);
        for(i=0;i<M;i++)
            if ((i%p)!=0)
        {
            i1=i%p;
            i2=i/p+1;
            MPI_Send(&A(i,0),M,MPI_FLOAT,i1,i2,MPI_COMM_WORLD);
        }
    }
    else
    {
        for(i=0;i<m;i++)
            MPI_Recv(&a(i,0),M,MPI_FLOAT,0,i+1,MPI_COMM_WORLD,&status);
    }

    // 进行主循环过程
    clock_t main_start = clock();
    for(i=0;i<m;i++)
        for(j=0;j<p;j++)
        {
            // j号进程负责广播主行元素
            if (my_rank==j)
            {
                v=i*p+j;

                // 使用OpenMP并行化内循环  
                // #pragma omp parallel for private(k) 
                for (k=v;k<M;k++)
                    f[k]=a(i,k);

                MPI_Bcast(f,M,MPI_FLOAT,my_rank,MPI_COMM_WORLD);
            }
            else
            {
                v=i*p+j;
                MPI_Bcast(f,M,MPI_FLOAT,j,MPI_COMM_WORLD);
            }

            // 编号小于my_rank的进程（包括my_rank本身）利用主行对其第i+1,…,m-1行数据做行变换
            if (my_rank<=j)
                #pragma omp parallel for private(k, w)
                for(k=i+1;k<m;k++)
            {
                a(k,v)=a(k,v)/f[v];
                for(w=v+1;w<M;w++)
                    a(k,w)=a(k,w)-f[w]*a(k,v);
            }

            // 编号大于my_rank的进程利用主行对其第i,…,m-1行数据做行变换
            if (my_rank>j)
                #pragma omp parallel for private(k, w)
                for(k=i;k<m;k++)
            {
                a(k,v)=a(k,v)/f[v];
                for(w=v+1;w<M;w++)
                    a(k,w)=a(k,w)-f[w]*a(k,v);
            }
        }
    clock_t main_end = clock();

    // 计算完毕后，0号进程从其余各进程中接收子矩阵a，得到经过变换的矩阵A
    if (my_rank==0)
    {
        for(i=0;i<m;i++)
            for(j=0;j<M;j++)
                A(i*p,j)=a(i,j);
    }
    if (my_rank!=0)
    {
        for(i=0;i<m;i++)
            MPI_Send(&a(i,0),M,MPI_FLOAT,0,i,MPI_COMM_WORLD);
    }
    else
    {
        for(i=1;i<p;i++)
            for(j=0;j<m;j++)
        {
            MPI_Recv(&a(j,0),M,MPI_FLOAT,i,j,MPI_COMM_WORLD,&status);
            for(k=0;k<M;k++)
                A((j*p+i),k)=a(j,k);
        }
    }

    if (my_rank==0)
    {
        for(i=0;i<M;i++)
            for(j=0;j<M;j++)
                u(i,j)=0.0;
        for(i=0;i<M;i++)
            for(j=0;j<M;j++)
                if (i==j)
                    l(i,j)=1.0;
        else
            l(i,j)=0.0;
        for(i=0;i<M;i++)
            for(j=0;j<M;j++)
                if (i>j)
                    l(i,j)=A(i,j);
        else
            u(i,j)=A(i,j);
        // printf("Input of file \"dataIn.txt\"\n");
        // printf("%d\t %d\n",M, N);
        // for(i=0;i<M;i++)
        // {
        //     for(j=0;j<N;j++)
        //         printf("%f\t",A(i,j));
        //     printf("\n");
        // }
        // printf("\nOutput of LU operation\n");
        // printf("Matrix L:\n");
        // for(i=0;i<M;i++)
        // {
        //     for(j=0;j<M;j++)
        //         printf("%f\t",l(i,j));
        //     printf("\n");
        // }
        // printf("Matrix U:\n");
        // for(i=0;i<M;i++)
        // {
        //     for(j=0;j<M;j++)
        //         printf("%f\t",u(i,j));
        //     printf("\n");
        // }
    }
    MPI_Finalize();
    Environment_Finalize(a,f);

    free(A);

    clock_t overall_end = clock(); 
    double time_spent = (double)(overall_end - overall_start) / CLOCKS_PER_SEC;  
    double main_time_spent = (double)(main_end - main_start) / CLOCKS_PER_SEC;  
    if (my_rank==0) printf("Time taken to run the whole program: %f seconds\n", time_spent); 
    if (my_rank==0) printf("Time taken to run the main loop: %f seconds\n", main_time_spent); 
    return(0);
}

// mpicc LU_dep_mix.c -o LU_dep_mix -fopenmp
// mpirun -np 4 LU_dep_mix 8 4
