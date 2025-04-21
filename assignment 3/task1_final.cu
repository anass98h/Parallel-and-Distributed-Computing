#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <omp.h>

// Error-checking macro for CUDA calls
#define CHECK_CUDA(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

const int TILE_DIM = 32; // Tile size (adjustable: 16 or 32 recommended)

__global__ void matMulTiled(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int N)
{
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Compute global row and col index for C
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float value = 0.0f;

    // Loop over tiles along the shared dimension (which is N here)
    for (int t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; ++t)
    {
        int tiledCol = t * TILE_DIM + threadIdx.x; // column index in A
        int tiledRow = t * TILE_DIM + threadIdx.y; // row index in B

        // Boundary check <- inside of our tile
        if (row < N && tiledCol < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + tiledCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile into shared memory
        if (col < N && tiledRow < N)
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product for this tile
#pragma unroll
        for (int i = 0; i < TILE_DIM; ++i)
        {
            value += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the result to global memory if within bounds
    if (row < N && col < N)
    {
        C[row * N + col] = value;
    }
}

int main()
{
    int N = 1024;

    size_t bytes = N * N * sizeof(float);

    // Allocate RAM for our matrices, which is the same size as MATRIX_SIZE * sizeof(float)
    float *h_A, *h_B, *h_C;
    CHECK_CUDA(cudaMallocHost(&h_A, bytes));
    CHECK_CUDA(cudaMallocHost(&h_B, bytes));
    CHECK_CUDA(cudaMallocHost(&h_C, bytes));

    for (int i = 0; i < N * N; ++i)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < N * N; ++i)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate GPU memory for our matrices, which is the same size as MATRIX_SIZE * sizeof(float)
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // Enable asynchronous operations
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Copy matrices A and B to the device asynchronously
    CHECK_CUDA(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, stream));

    // REST OF CODE  .....

    // Configure grid and block dimensions
    dim3 block(TILE_DIM, TILE_DIM); // threads x threads -> eg 16x16
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    double sharedMemoryForKernel = 0;
    double startTime = omp_get_wtime();
    matMulTiled<<<grid, block, sharedMemoryForKernel, stream>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaPeekAtLastError());
    double endTime = omp_get_wtime();
    double durationMS = (endTime - startTime) * 1000;

    // Copy result matrix C back to host asynchronously
    CHECK_CUDA(cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost, stream));

    // Wait for all operations in the stream to finish
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::cout << "Kernel execution time (excluding transfers): "
              << durationMS << " ms" << std::endl;

    // (Optional) Validate a few entries of the result for correctness
    bool correct = true;
    for (int i = 0; i < 20 && correct; ++i)
    {
        for (int j = 0; j < 20 && correct; ++j)
        {
            double ref = 0.0;
            for (int k = 0; k < N; ++k)
                ref += h_A[i * N + k] * h_B[k * N + j];
            if (fabs(h_C[i * N + j] - ref) > 1e-3)
            {
                correct = false;
                std::cerr << "Mismatch at (" << i << ", " << j << "): "
                          << h_C[i * N + j] << " vs " << ref << std::endl;
            }
        }
    }

    if (correct)
        std::cout << "Result verification passed!" << std::endl;
    else
        std::cerr << "Result verification failed!" << std::endl;

    // Clean up all allocated resources
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));
    CHECK_CUDA(cudaFreeHost(h_C));

    return 0;
}
