# Simple-CPU-Parallel
Simple usage demo of cpu parallel program (openmp+mpi)

## OpemMP

testing openmp environment:

```sh
$ g++ -fopenmp hello_openmp.cpp -o hello_openmp.o && ./hello_openmp.o
Hello World from thread 5
Hello World from thread 2
Hello World from thread 1
Hello World from thread 4
Hello World from thread 7
Hello World from thread 3
Hello World from thread 6
>>> Message from main thread: Number of threads = 8
Hello World from thread 0
```
