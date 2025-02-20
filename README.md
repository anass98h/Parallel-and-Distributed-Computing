# Parallel-and-Distributed-Computing

Parallel and Distributed Computing

## Running the programs

Below is a single line command to run the task.

```sh
 ❯ g++ ./task1.cpp -o ./output/task1 -std=c++11 -O2 && ./output/task1 && python3 ./test_matmul.py
```

It consists out of 3 parts separated by `&&`:

1. using `g++` compiler with flags (`-O2` and `-std=c++11` are optional, but "optimal" for performance and succesful
   compile) in to compile the cpp task. Change the file name and output path if you want to change which task to be
   executed.
2. Running the compiled program
3. Running the tests if matmul was done correctly. This is executed/asserted against the numpy implementation of matmul.
   Change python version in order to config the running kernel.

The results should look something similar like this:

```sh
--------------RESULTS------------------
SIZE OF MATRIX = 1024
Execution time: 1091 ms
The result of the matrix multiplication is correct
Execution time of numpy matmul: 5.29 ms
----------------------------------------
```
