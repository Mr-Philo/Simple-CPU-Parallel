#include<iostream>  
#include<vector>  
  
using namespace std;  
  
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
  
int main() {  
    vector<int> arr = {2, 6, 7, 1, 3, 9, 8, 4, 5};  
    mergeSort(arr, 0, arr.size()-1);  
    for (int i : arr) {  
        cout << i << " ";  
    }  
    return 0;  
}  
