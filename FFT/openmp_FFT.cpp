#include <iostream>  
#include <complex>  
#include <vector>  
#include <cmath>  
#include <omp.h>  
  
using namespace std;  
  
const double PI = 3.14159265358979323846;  
  
// 递归实现快速傅里叶变换  
// 在这个问题中，直接并行化蝶形运算循环可能并不是最佳选择，因为在递归调用中存在数据依赖性。为了解决这个问题，可以考虑将递归调用fft_recursive(a_even和fft_recursive(a_odd)并行化。

void fft_recursive(vector<complex<double>>& a) {  
    int n = a.size();  
    if (n == 1) {  
        return;  
    }  
  
    vector<complex<double>> a_even(n / 2), a_odd(n / 2);  
    // 并行分割输入 
    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < n / 2; i++) {  
        a_even[i] = a[2 * i];  
        a_odd[i] = a[2 * i + 1];  
    }  
  
    // 并行化递归调用  
    #pragma omp parallel sections  
    {  
        #pragma omp section  
        fft_recursive(a_even);  
  
        #pragma omp section  
        fft_recursive(a_odd);  
    }  
  
    double angle = 2 * PI / n;  
    // complex<double> w(1), wn(cos(angle), sin(angle));  
    // complex<double> w_i;
  
    // 并行化蝶形运算  
    // #pragma omp parallel for
    // for (int i = 0; i < n / 2; i++) {  
    //     a[i] = a_even[i] + w * a_odd[i];  
    //     a[i + n / 2] = a_even[i] - w * a_odd[i];  
    //     w *= wn;        // 这一步关于旋转因子W的计算依赖于前一次循环的结果，存在流依赖的数据依赖关系，因此该循环不能直接并行化，否则会导致错误的结果
    // }  

    // 并行化蝶形运算
    // #pragma omp parallel for private(w_i)       // w_i是在循环外定义的，因此这里必须声明变量w_i（旋转因子）在循环内的私有性，从而避免不同线程之间产生对w_i值的数据竞争，进而导致计算结果出错
    #pragma omp parallel for
    for (int i = 0; i < n / 2; i++) {  
        // w_i = cos(angle * i) + sin(angle * i) * 1i;
        complex<double> w_i = cos(angle * i) + sin(angle * i) * 1i;
        a[i] = a_even[i] + w_i * a_odd[i];
        a[i + n / 2] = a_even[i] - w_i * a_odd[i];
    }
}
  
int main() {  
    vector<complex<double>> data = {1, 1, 1, 1, 0, 0, 0, 0};  
    omp_set_num_threads(4);
  
    fft_recursive(data);  
  
    cout << "FFT result:" << endl;  

    // 输出“实部+虚部”的形式
    // for (const auto& d : data) {  
    //     cout << d.real() << ", " << d.imag() << "*i" << endl;  
    // }

    // 输出“幅值+相角”的形式
    
    for (const auto& d : data) {  
        double magnitude = abs(d);  
        double phase = arg(d) / PI; 
        if(phase == 0)
            cout << magnitude << ", 0" << endl;
        else
            cout << magnitude << ", " << phase << "*PI" << endl;  
    } 
    
  
    return 0;  
}  

/*修改部分：

在蝶形运算前添加#pragma omp parallel for指令，表示将循环并行化。这样，循环中的每次迭代都可以在多个线程上同时执行。

注意：在某些情况下，递归调用fft_recursive(a_even)和fft_recursive(a_odd)也可以并行化，但这取决于系统资源和问题规模。在实际应用中，您可以尝试并行化这两个调用，但需要注意线程同步和资源竞争的问题。*/

// g++ -fopenmp -o openmp_FFT openmp_FFT.cpp
// ./openmp_FFT
