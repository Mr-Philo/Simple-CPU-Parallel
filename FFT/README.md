## KEY

关于如何使用OpenMP并行化FFT的蝶形运算

关键是弄清楚蝶形运算中的数据依赖关系和数据竞争问题。蝶形运算中，第k次计算时的旋转因子e^j(2*k*PI/N)是与k有关的，**这一点非常的重要**

1. 数据依赖问题

在第一版代码里，使用的是比较高效的在迭代的同时计算旋转因子的方法。此时会产生数据依赖（计算旋转因子的这行代码关于自身的流依赖），故循环不能直接进行并行化

2. 循环变量问题

在OpenMP中，使用#pragma omp parallel for指令并行化一个for循环时，循环变量（如i）会被自动地分配给各个线程去处理。每个线程都有自己的i值，这些i值是不会相互影响的。OpenMP会负责管理这个过程，程序员不需要手动进行控制。

例如，如果你有两个线程去并行执行for (int i = 0; i < 8; i++)这个循环，OpenMP可能会将i的值0到3分配给第一个线程，将i的值4到7分配给第二个线程。这只是一种可能的分配方式，实际的分配方式取决于OpenMP的调度策略和运行时环境。

在每个线程中，i的值会按照常规的for循环方式依次递增，直到处理完被分配给该线程的所有i值。不同线程中的i值是独立的，互不影响。

3. 私有变量和数据竞争问题

第二版代码中，如果在循环中计算旋转因子w_i，但w_i是在循环外定义的话，就必须将w_i指定为私有变量，否则不同线程之间会竞争w_i的数据值。其实有其他的方法来解决这个问题：一是直接定义n个w_i（即一个数组）而不是定义一个w_i，因为不同的线程访问的i值不同，就不会产生竞争，不过当n很大是会造成很大的空间开销。二是在循环内定义w_i变量，而不是在循环外，这样w_i就自动变成了循环内的私有变量。

另：代码还附带了CUDA对FFT的实现
另：代码还附带了MPI对FFT的实现，以及MPI+OpenMP混合框架对FFT的实现（不推荐使用）
