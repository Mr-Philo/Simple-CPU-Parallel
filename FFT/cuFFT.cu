#include <iostream>  
#include <cmath>  
#include <vector>  
#include <complex>  
#include <cuda_runtime.h>  
#include <cufft.h>  
  
using namespace std;  
  
int main() {  
    int N = 8;  
    cufftDoubleComplex *h_data = new cufftDoubleComplex[N];  
    for (int i = 0; i < N; ++i) {  
        h_data[i].x = (i < 4) ? 1.0 : 0.0;  
        h_data[i].y = 0.0;  
    }  
  
    cufftDoubleComplex *d_data;  
    cudaMalloc((void**)&d_data, N * sizeof(cufftDoubleComplex));  
    cudaMemcpy(d_data, h_data, N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);  
  
    cufftHandle plan;  
    cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);  
    cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD);  
    cudaMemcpy(h_data, d_data, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);  
  
    cout << "FFT result:" << endl;  

    // 输出“实部+虚部”的形式
    for (int i = 0; i < N; ++i) {  
        cout << h_data[i].x << ", " << h_data[i].y << "*i" << endl;  
    }

    // 输出“幅值+相角”的形式
    /*
    for (int i = 0; i < N; ++i) {  
        double magnitude = sqrt(h_data[i].x * h_data[i].x + h_data[i].y * h_data[i].y);  
        double phase = atan2(h_data[i].y, h_data[i].x) / M_PI;  
        cout << magnitude << ", " << phase << " * PI" << endl;  
    }  
    */
  
    cufftDestroy(plan);  
    cudaFree(d_data);  
    delete[] h_data;  
    return 0;  
}  

// nvcc cuFFT.cu -lcufft -o cuFFT
// ./cuFFT
