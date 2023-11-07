#include <iostream>
#include <omp.h>

using namespace std;

int main(){
    int num_threads = 8;

    #pragma omp parallel num_threads(num_threads)
    {
        int ID = omp_get_thread_num();

        #pragma omp critical
        {
            if (ID == 0){
                cout << ">>> Message from main thread: Number of threads = " << omp_get_num_threads() << endl;
            }
            cout << "Hello World from thread " << ID << endl;
        }
    }
}

// g++ -fopenmp hello_openmp.cpp -o hello_openmp.o && ./hello_openmp.o
