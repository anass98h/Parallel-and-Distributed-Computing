#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <fstream>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>
#include <omp.h>

void throw_on_cuda_error(cudaError_t code, const char *file, int line)
{
  if(code != cudaSuccess)
  {
    std::stringstream ss;
    ss << file << "(" << line << ")";
    std::string file_and_line;
    ss >> file_and_line;
    throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
  }
}

#define threads 32
#define TILE_SIZE 32

std::vector<int> generate_matrix_1d(int size, unsigned int seed)
{
    static std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    int full_size = size * size;
    std::vector<int> matrix(full_size);

    for (int i = 0; i < full_size; i++)
    {
        matrix[i] = distribution(generator);
    }

    return matrix;
}

void save_matrix_1d(std::vector<int> &matrix, std::string filename, int size)
{
    std::ofstream myFile("tests/" + filename);
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            myFile << matrix[i * size + j] << ",";
        }
        myFile << "\n";
    }
    myFile.close();
}

// _______  MAIN PROGRAM _____________


__global__ void matrixMultiplyKernel(int *A, int *B, int *C, int size)
{
    // WHOAMI for each threads
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 1. optimization: Boundary check for our matrix -> avoid control divergence
    if (row < size && col < size) {
        int sum = 0;

        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

__global__ void matrixMultiplyKernelCoalesced(int *A, int *B, int *C, int size) {
    __shared__ int A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ int B_tile[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // position
    int aRow = by * TILE_SIZE + ty;
    int aCol = tx;
    int bRow = ty;
    int bCol = bx * TILE_SIZE + tx;
    // output 
    int cRow = aRow;
    int cCol = bCol;

    float sum = 0;

    for (int t = 0; t < (size + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory in a coalesced manner
        if (aRow < size && t * TILE_SIZE + aCol < size) {
            A_tile[ty][tx] = A[aRow * size + (t * TILE_SIZE + aCol)];
        } else {
            A_tile[ty][tx] = 0;
        }

        if (t * TILE_SIZE + bRow < size && bCol < size) {
            B_tile[ty][tx] = B[(t * TILE_SIZE + bRow) * size + bCol];
        } else {
            B_tile[ty][tx] = 0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[ty][k] * B_tile[k][tx];
        }

        __syncthreads();
    }

    // Write output if within bounds
    if (cRow < size && cCol < size) {
        C[cRow * size + cCol] = sum;
    }
}

std::vector<int> cuda_matrix_multiply(
    const std::vector<int> &A, 
    const std::vector<int> &B, 
    int size) {

    int elements = size * size;
    std::vector<int> C(elements, 0);
    
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, elements * sizeof(int));
    cudaMalloc((void**)&d_B, elements * sizeof(int));
    cudaMalloc((void**)&d_C, elements * sizeof(int));
    
    cudaMemcpy(d_A, A.data(), elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), elements * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 THREADS(threads, threads); // <-- THREADS 16x16
    dim3 BLOCKS((size + TILE_SIZE - 1) / TILE_SIZE, (size + TILE_SIZE - 1) / TILE_SIZE); // <-- BLOCKS 
    
    matrixMultiplyKernelCoalesced<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, size);
    cudaDeviceSynchronize(); // sync because kernel executation is async
    
    cudaMemcpy(C.data(), d_C, elements * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return C;
}


int main()
{
    try
  {
      
      int SIZE_OF_MATRIX = 1024;
      std::vector<int> matrix1 = generate_matrix_1d(SIZE_OF_MATRIX, 1);
      std::vector<int> matrix2 = generate_matrix_1d(SIZE_OF_MATRIX, 1);
      std::vector<int> targetMatrix;
      
    double startTime = omp_get_wtime();
    // throw_on_cuda_error(cuda_matrix_multiply(matrix1, matrix2, SIZE_OF_MATRIX))
    targetMatrix = cuda_matrix_multiply(matrix1, matrix2, SIZE_OF_MATRIX);
    double endTime = omp_get_wtime();
    double duration = (endTime - startTime) * 1000;
  
    // Print results
    std::cout << "--------------RESULTS------------------" << std::endl;
    std::cout << "SIZE OF MATRIX = " << SIZE_OF_MATRIX << std::endl;
    std::cout << "Execution time: " << duration << " ms" << std::endl;

    // save_matrix_1d(matrix1, "matrix1.csv", SIZE_OF_MATRIX);
    // save_matrix_1d(matrix2, "matrix2.csv", SIZE_OF_MATRIX);
    // save_matrix_1d(targetMatrix, "result.csv", SIZE_OF_MATRIX);
}catch(thrust::system_error &e){
  std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
  // oops, recover
  cudaSetDevice(0);
}

return 0;
}


// nvcc test.cu -o test

// __host__​cudaError_t cudaEventCreate ( cudaEvent_t* event, unsigned int  flags)