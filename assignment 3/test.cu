#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <fstream>

#define TILE_WIDTH 16

std::vector<std::vector<int>> generate_matrix_2d(int size, unsigned int seed)
{
    static std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    std::vector<std::vector<int>> matrix(size, std::vector<int>(size));
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrix[i][j] = distribution(generator);
        }
    }
    return matrix;
}


void save_matrix_2d(std::vector<std::vector<int>> &matrix, std::string filename)
{
    std::ofstream myFile("tests/" + filename);
    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < matrix[i].size(); j++)
        {
            myFile << matrix[i][j] << ",";
        }
        myFile << "\n";
    }
    myFile.close();
}

// _______  MAIN PROGRAM _____________


__global__ void matrixMultiplyKernel(int *A, int *B, int *C, int size)
{
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < size && col < size) {
        int sum = 0;

        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

std::vector<std::vector<int>> cuda_matrix_multiply(
    const std::vector<std::vector<int>> &A, 
    const std::vector<std::vector<int>> &B
) {
   int size = A.size();
   int *h_A = new int[size * size];
   int *h_B = new int[size * size];
   int *h_C = new int[size * size];
  
   // 2D -> 1D
   for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
        h_A[i * size + j] = A[i][j];
        h_B[i * size + j] = B[i][j];
    }
   }

   int *d_A, *d_B, *d_C;
   cudaMalloc((void**)&d_A, size * size * sizeof(int));
   cudaMalloc((void**)&d_B, size * size * sizeof(int));
   cudaMalloc((void**)&d_C, size * size * sizeof(int));

   cudaMemcpy(d_A, h_A, size * size * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, size * size * sizeof(int), cudaMemcpyHostToDevice);

   dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
   dim3 gridSize((size + TILE_WIDTH - 1) / TILE_WIDTH, 
   (size + TILE_WIDTH - 1) / TILE_WIDTH);

   matrixMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, size);
   cudaDeviceSynchronize();

   // cpy back to host
   cudaMemcpy(h_C, d_C, size * size * sizeof(int), cudaMemcpyDeviceToHost);

   // 1D -> 2D
   std::vector<std::vector<int>> C(size, std::vector<int>(size));
   for (int i = 0; i < size; i++) {
       for (int j = 0; j < size; j++) {
           C[i][j] = h_C[i * size + j];
       }
   }

   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

   delete[] h_A;
   delete[] h_B;
   delete[] h_C;

   return C;
}


int main()
{
    int SIZE_OF_MATRIX = 1024;
    std::vector<std::vector<int>> matrix1 = generate_matrix_2d(SIZE_OF_MATRIX, 1);
    std::vector<std::vector<int>> matrix2 = generate_matrix_2d(SIZE_OF_MATRIX, 1);
    std::vector<std::vector<int>> targetMatrix(SIZE_OF_MATRIX, std::vector<int>(SIZE_OF_MATRIX));
    // <--- here 
    auto start = std::chrono::high_resolution_clock::now();
    targetMatrix = cuda_matrix_multiply(matrix1, matrix2);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Print results
    std::cout << "--------------RESULTS------------------" << std::endl;
    std::cout << "SIZE OF MATRIX = " << SIZE_OF_MATRIX << std::endl;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    save_matrix_2d(matrix1, "matrix1.csv");
    save_matrix_2d(matrix2, "matrix2.csv");
    save_matrix_2d(targetMatrix, "result.csv");
    return 0;
}