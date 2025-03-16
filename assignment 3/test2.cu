#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Error-checking macro for CUDA calls
#define CHECK_CUDA(call) do {                                    \
    cudaError_t err = (call);                                    \
    if (err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(EXIT_FAILURE);                                      \
    } } while(0)

const int TILE_DIM = 32;  // Tile size (adjustable: 16 or 32 recommended)

// CUDA kernel for tiled matrix multiplication (C = A * B)
__global__
void matMulTiled(const float* __restrict__ A, const float* __restrict__ B, 
                 float* __restrict__ C, int N, int K, int M) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float value = 0.0f;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        int tiledCol = t * TILE_DIM + threadIdx.x;   // column index for A
        int tiledRow = t * TILE_DIM + threadIdx.y;     // row index for B

        if (row < N && tiledCol < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + tiledCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < M && tiledRow < K)
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * M + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_DIM; ++i) {
            value += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < M) {
        C[row * M + col] = value;
    }
}

int main() {
    // Example matrix dimensions: A is N x K, B is K x M, C is N x M
    int N = 1024;  // rows of A and C
    int K = 1024;  // columns of A and rows of B
    int M = 1024;  // columns of B and C

    size_t bytesA = N * K * sizeof(float);
    size_t bytesB = K * M * sizeof(float);
    size_t bytesC = N * M * sizeof(float);

    // Allocate pinned host memory for A, B, and C
    float *h_A, *h_B, *h_C;
    CHECK_CUDA(cudaMallocHost(&h_A, bytesA));
    CHECK_CUDA(cudaMallocHost(&h_B, bytesB));
    CHECK_CUDA(cudaMallocHost(&h_C, bytesC));

    // Initialize matrices A and B with random values
    for (int i = 0; i < N * K; ++i)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int j = 0; j < K * M; ++j)
        h_B[j] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytesA));
    CHECK_CUDA(cudaMalloc(&d_B, bytesB));
    CHECK_CUDA(cudaMalloc(&d_C, bytesC));

    // Create a CUDA stream for asynchronous operations
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Copy matrices A and B to the device asynchronously
    CHECK_CUDA(cudaMemcpyAsync(d_A, h_A, bytesA, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_B, h_B, bytesB, cudaMemcpyHostToDevice, stream));

    // Configure grid and block dimensions
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    // Create CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    // Record start event
    CHECK_CUDA(cudaEventRecord(startEvent, stream));

    // Launch the tiled matrix multiplication kernel
    matMulTiled<<<grid, block, 0, stream>>>(d_A, d_B, d_C, N, K, M);
    CHECK_CUDA(cudaPeekAtLastError());

    // Record stop event after kernel execution and before copying result back
    CHECK_CUDA(cudaEventRecord(stopEvent, stream));

    // Copy result matrix C back to host asynchronously
    CHECK_CUDA(cudaMemcpyAsync(h_C, d_C, bytesC, cudaMemcpyDeviceToHost, stream));

    // Wait for all operations in the stream to finish
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Calculate elapsed time in milliseconds
    float elapsedTime;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));
    std::cout << "Kernel execution time (excluding transfers): " 
              << elapsedTime << " ms" << std::endl;

    // Optionally, you can print out overall runtime (including memory transfers) using CPU timers

    // (Optional) Validate correctness for a few entries
    bool correct = true;
    for (int i = 0; i < 5 && correct; ++i) {
        for (int j = 0; j < 5 && correct; ++j) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k)
                ref += h_A[i * K + k] * h_B[k * M + j];
            if (fabs(h_C[i * M + j] - ref) > 1e-3) {
                correct = false;
                std::cerr << "Mismatch at (" << i << ", " << j << "): "
                          << h_C[i * M + j] << " vs " << ref << std::endl;
            }
        }
    }
    if (correct)
        std::cout << "Result verification passed!" << std::endl;
    else
        std::cerr << "Result verification failed!" << std::endl;

    // Clean up
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));
    CHECK_CUDA(cudaFreeHost(h_C));

    return 0;
}
