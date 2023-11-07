#include <iostream>  
#include <complex>  
#include <vector>  
#include <cmath>  
#include <chrono>  
#include <omp.h>  
  
using namespace std;  
using namespace std::chrono;  
  
const double PI = 3.14159265358979323846;  
  
// 串行递归实现快速傅里叶变换  
void fft_recursive_serial(vector<complex<double>>& a) {  
    int n = a.size();  
    if (n == 1) {  
        return;  
    }  
  
    vector<complex<double>> a_even(n / 2), a_odd(n / 2);  
    for (int i = 0; i < n / 2; i++) {  
        a_even[i] = a[2 * i];  
        a_odd[i] = a[2 * i + 1];  
    }  
  
    fft_recursive_serial(a_even);  
    fft_recursive_serial(a_odd);  
  
    double angle = 2 * PI / n;      
    // complex<double> w(1), wn(cos(angle), sin(angle));
    // complex<double> w_i; 
    
    for (int i = 0; i < n / 2; i++) {  
        // w_i = cos(angle * i) + sin(angle * i) * 1i;
        complex<double> w_i = cos(angle * i) + sin(angle * i) * 1i; 
        a[i] = a_even[i] + w_i * a_odd[i];  
        a[i + n / 2] = a_even[i] - w_i * a_odd[i];  
    }  
}  
  
// 使用OpenMP的并行递归实现快速傅里叶变换  
void fft_recursive_parallel(vector<complex<double>>& a) {  
    int n = a.size();  
    if (n == 1) {  
        return;  
    }  
  
    vector<complex<double>> a_even(n / 2), a_odd(n / 2); 
    // 并行分割输入  
    // #pragma omp parallel for     // 这个反而会掉速
    for (int i = 0; i < n / 2; i++) {  
        a_even[i] = a[2 * i];  
        a_odd[i] = a[2 * i + 1];  
    }  
    
    // 并行化递归调用  
    #pragma omp parallel sections  
    {  
        #pragma omp section  
        fft_recursive_parallel(a_even);  
  
        #pragma omp section  
        fft_recursive_parallel(a_odd);  
    }  
  
    double angle = 2 * PI / n;  
    // complex<double> w(1), wn(cos(angle), sin(angle)); 
    // complex<double> w_i; 
    
    // 并行化蝶形运算  为什么这里会掉速555
    // #pragma omp parallel for private(w_i)
    #pragma omp parallel for
    for (int i = 0; i < n / 2; i++) { 
        // w_i = cos(angle * i) + sin(angle * i) * 1i;
        complex<double> w_i = cos(angle * i) + sin(angle * i) * 1i;
        a[i] = a_even[i] + w_i * a_odd[i];  
        a[i + n / 2] = a_even[i] - w_i * a_odd[i];  
    } 
}  
  
int main(int argc, char** argv) {  

    if (argc != 2) {  
        cerr << "Usage: " << argv[0] << " <arg:numThreads>" << endl;  
        return 1;  
    }  
  
    int numThreads = atoi(argv[1]);
    if (numThreads > omp_get_max_threads()) {
        cerr << "numThreads should be less than " << omp_get_max_threads() << endl;
        return 1;
    }
    omp_set_num_threads(numThreads);  

    // 使用较大的输入数据  
    int n = 1 << 20;        // 左移操作，等价于 int n = pow(2, 20);，结果是 n = 1048576
    vector<complex<double>> data(n, 1);  
  
    // 复制输入数据，以便在串行和并行版本中使用相同的数据  
    vector<complex<double>> data_serial(data);  
    vector<complex<double>> data_parallel(data);  
    
    cout << "Start, num of threads:" << numThreads << ", size of testing array:" << n << endl;
    // 测量串行版本的运行时间  
    auto start_serial = high_resolution_clock::now();  
    fft_recursive_serial(data_serial);  
    auto end_serial = high_resolution_clock::now();  
    auto duration_serial = duration_cast<milliseconds>(end_serial - start_serial);  
  
    // 测量并行版本的运行时间  
    auto start_parallel = high_resolution_clock::now();  
    fft_recursive_parallel(data_parallel);  
    auto end_parallel = high_resolution_clock::now();  
    auto duration_parallel = duration_cast<milliseconds>(end_parallel - start_parallel);  

    // 确保并行计算结果和串行计算相同
    for (int i = 0; i < n; i++) {  
        if (abs(data_serial[i] - data_parallel[i]) > 1e-5) {  
            cout << "Results are not identical!" << endl;  
            return 1;  
        }  
    }
  
    cout << "Serial version duration: " << duration_serial.count() << " ms" << endl;  
    cout << "Parallel version duration: " << duration_parallel.count() << " ms" << endl;  
    
    // 计算加速比
    double speedup = static_cast<double>(duration_serial.count()) / duration_parallel.count();
    cout << "Speedup: " << speedup << endl;

    // 计算并行效率
    double efficiency = speedup / numThreads;
    cout << "Efficiency: " << efficiency << endl;
    
    return 0;  
}  

// g++ -fopenmp -o compare_FFT compare_FFT.cpp
// ./compare_FFT 4
