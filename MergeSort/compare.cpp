#include <iostream>  
#include <vector>  
#include <random>  
#include <chrono>  
#include <cassert>
#include <omp.h>  
  
using namespace std;  
using namespace std::chrono;  
  
vector<int> generateRandomArray(int size) {  
    random_device rd;  
    mt19937 gen(rd());  
    uniform_int_distribution<> dis(1, 100);  
  
    vector<int> arr(size);  
    for (int i = 0; i < size; ++i) {  
        arr[i] = dis(gen);  
    }  
  
    return arr;  
}  
  
bool isSorted(const vector<int>& arr) {  
    for (int i = 1; i < arr.size(); ++i) {  
        if (arr[i] < arr[i-1]) {  
            return false;  
        }  
    }  
  
    return true;  
}  

void merge(vector<int>& arr, int left, int mid, int right) {  
    vector<int> temp(right-left+1);  
    int i = left, j = mid+1, k = 0;  
      
    while (i <= mid && j <= right) {  
        if (arr[i] <= arr[j]) {  
            temp[k++] = arr[i++];  
        } else {  
            temp[k++] = arr[j++];  
        }  
    }  
      
    while (i <= mid) {  
        temp[k++] = arr[i++];  
    }  
      
    while (j <= right) {  
        temp[k++] = arr[j++];  
    }  
      
    for (i = left, k = 0; i <= right; ) {  
        arr[i++] = temp[k++];  
    }  
}  
  
void mergeSort(vector<int>& arr, int left, int right) {  
    if (left < right) {  
        int mid = (left + right) / 2;  
        mergeSort(arr, left, mid);  
        mergeSort(arr, mid+1, right);  
        merge(arr, left, mid, right);  
    }  
}  
  
void parallelMergeSort(vector<int>& arr, int left, int right) {  
    if (left < right) {  
        int mid = (left + right) / 2;  
          
        #pragma omp parallel sections  
        {  
            #pragma omp section  
            {  
                mergeSort(arr, left, mid);  
            }  
            #pragma omp section  
            {  
                mergeSort(arr, mid+1, right);  
            }  
        }  
          
        merge(arr, left, mid, right);   
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
    int size = 100000;  

    double totalSpeedup = 0.0;  
    double totalEfficiency = 0.0;  
    int numRuns = 10;  
    
    cout << "Start, num of threads:" << numThreads << ", size of testing array:" << size << ", num of runs:" << numRuns << endl;
    for (int i = 0; i < numRuns; ++i) { 
  
        vector<int> arr = generateRandomArray(size);  
  
        vector<int> arr1 = arr;  
        auto start1 = high_resolution_clock::now();  
        mergeSort(arr1, 0, size-1);  
        auto stop1 = high_resolution_clock::now();  
        auto duration1 = duration_cast<microseconds>(stop1 - start1);  
        // cout << "Serial time: " << duration1.count() << " microseconds" << endl;  
        assert(isSorted(arr1));  
    
        vector<int> arr2 = arr;  
        
        omp_set_num_threads(numThreads);  
        auto start2 = high_resolution_clock::now();  
        // #pragma omp parallel  
        // {  
        //     #pragma omp single  
        //     parallelMergeSort(arr2, 0, size-1);  
        // }  
        parallelMergeSort(arr2, 0, size-1);
        auto stop2 = high_resolution_clock::now();  
        auto duration2 = duration_cast<microseconds>(stop2 - start2);  
        // cout << "Parallel time: " << duration2.count() << " microseconds" << endl;  
        assert(isSorted(arr2));  

        // Calculate speedup and efficiency  
        double speedup = (double)duration1.count() / duration2.count();  
        double efficiency = speedup / numThreads;  
        totalSpeedup += speedup;  
        totalEfficiency += efficiency;  
    }
  
    cout << "Average speedup: " << totalSpeedup / numRuns << endl;  
    cout << "Average efficiency: " << totalEfficiency / numRuns << endl;
  
    return 0;  
}
