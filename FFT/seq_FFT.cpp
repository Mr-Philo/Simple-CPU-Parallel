#include <iostream>  
#include <complex>  
#include <vector>  
#include <cmath>  
  
using namespace std;  
  
const double PI = 3.14159265358979323846;  
  
// 递归实现快速傅里叶变换  
void fft_recursive(vector<complex<double>>& a) {  
    int n = a.size();  
    if (n == 1) {  
        return;  
    }  
  
    vector<complex<double>> a_even(n / 2), a_odd(n / 2);  
    for (int i = 0; i < n / 2; i++) {  
        a_even[i] = a[2 * i];  
        a_odd[i] = a[2 * i + 1];  
    }  
  
    fft_recursive(a_even);  
    fft_recursive(a_odd);  
  
    double angle = 2 * PI / n;  
    // complex<double> w(1), wn(cos(angle), sin(angle));        // 不用这种在迭代过程中计算旋转因子w_i的算法
    complex<double> w_i;
    
    // 蝶形运算  
    for (int i = 0; i < n / 2; i++) {  
        // a[i] = a_even[i] + w * a_odd[i];  
        // a[i + n / 2] = a_even[i] - w * a_odd[i];  
        // w *= wn;        // 这一步计算旋转因子的算法太巧妙了，巧妙到很难使用并行对其优化。这里考虑故意在串行算法中使用直接按公式计算旋转因子以拖慢速度。

        w_i = cos(angle * i) + sin(angle * i) * 1i;
        a[i] = a_even[i] + w_i * a_odd[i];
        a[i + n / 2] = a_even[i] - w_i * a_odd[i];
    }  
}  
  
int main() {  
    vector<complex<double>> data = {1, 1, 1, 1, 0, 0, 0, 0};  
  
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

// g++ -o seq_FFT seq_FFT.cpp
// ./seq_FFT

/* correct result
FFT result:
4, 0
2.61313, 0.375*PI
0, 0
1.08239, 0.125*PI
0, 0
1.08239, -0.125*PI
0, 0
2.61313, -0.375*PI
*/
