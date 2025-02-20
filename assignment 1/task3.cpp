#include <iostream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <omp.h>
#include <fstream>
#include <random>


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

// Threshold to switch to the direct (triple-nested) multiply.
// You can tune this depending on cache sizes.
static const int BLOCK_SIZE = 1024;

void matmulRecHelper(const int* A, const int* B, int* C,
                     int n, int offsetA, int offsetB, int offsetC)
{
    // IKJ version
    if (n <= BLOCK_SIZE)
    {
        #pragma omp parallel for schedule(dynamic, 4)
        for (int i = 0; i < n; ++i)
        {
            for (int k = 0; k < n; ++k)
            {
                int r = A[i * offsetA + k];
                for (int j = 0; j < n; ++j)
                {
                    C[i * offsetC + j] += r * B[k * offsetB + j];
                }
            }
        }
    }
    else
    {
        // Divide and conquer: split n×n block into four (n/2)×(n/2) blocks
        int half = n / 2;

        // Offsets in A
        const int* A11 = A;
        const int* A12 = A + half;                 // shift right by half
        const int* A21 = A + half * offsetA;       // shift down by half
        const int* A22 = A21 + half;               // shift down and right

        // Offsets in B
        const int* B11 = B;
        const int* B12 = B + half;                 // shift right by half
        const int* B21 = B + half * offsetB;       // shift down by half
        const int* B22 = B21 + half;               // shift down and right

        // Offsets in C
        int* C11 = C;
        int* C12 = C + half;                       // shift right
        int* C21 = C + half * offsetC;             // shift down
        int* C22 = C21 + half;                     // shift down and right

        // Use a taskgroup to group our eight recursive calls
        //#pragma omp taskgroup
        {

        // C11 = A11*B11 + A12*B21
        //#pragma omp task shared(A,B,C) if (n > BLOCK_SIZE)
        matmulRecHelper(A11, B11, C11, half, offsetA, offsetB, offsetC);
        //#pragma omp task shared(A,B,C) if (n > BLOCK_SIZE)
        matmulRecHelper(A12, B21, C11, half, offsetA, offsetB, offsetC);

        // C12 = A11*B12 + A12*B22
        //#pragma omp task shared(A,B,C) if (n > BLOCK_SIZE)
        matmulRecHelper(A11, B12, C12, half, offsetA, offsetB, offsetC);
        //#pragma omp task shared(A,B,C) if (n > BLOCK_SIZE)
        matmulRecHelper(A12, B22, C12, half, offsetA, offsetB, offsetC);

        // C21 = A21*B11 + A22*B21
        //#pragma omp task shared(A,B,C) if (n > BLOCK_SIZE)
        matmulRecHelper(A21, B11, C21, half, offsetA, offsetB, offsetC);
        //#pragma omp task shared(A,B,C) if (n > BLOCK_SIZE)
        matmulRecHelper(A22, B21, C21, half, offsetA, offsetB, offsetC);

        // C22 = A21*B12 + A22*B22
        //#pragma omp task shared(A,B,C) if (n > BLOCK_SIZE)
        matmulRecHelper(A21, B12, C22, half, offsetA, offsetB, offsetC);
        //#pragma omp task shared(A,B,C) if (n > BLOCK_SIZE)
        matmulRecHelper(A22, B22, C22, half, offsetA, offsetB, offsetC);
        }
    }
}

/**
 * Public interface: multiplies two n×n matrices A and B (in row-major order)
 * and returns the result in a new std::vector<int>.
 */
std::vector<int> matmulRec(const std::vector<int>& A,
                           const std::vector<int>& B, 
                           int n)
{
    // Sanity-check
    if (static_cast<int>(A.size()) < n*n || static_cast<int>(B.size()) < n*n)
    {
        throw std::runtime_error("Input matrices are too small for n×n multiplication.");
    }

    // Allocate result (C) and initialize to 0
    std::vector<int> C(n * n, 0);

    // Recursively multiply full n×n blocks

    matmulRecHelper(A.data(), B.data(), C.data(), n, n, n, n);
    return C;
}

int main()
{
    int SIZE_OF_MATRIX = 3072;

    std::vector<int> matrix1 = generate_matrix_1d(SIZE_OF_MATRIX, 1);
    std::vector<int> matrix2 = generate_matrix_1d(SIZE_OF_MATRIX, 1);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> targetMatrix = matmulRec(matrix1, matrix2, SIZE_OF_MATRIX);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Print results
    std::cout << "--------------RESULTS------------------" << std::endl;
    std::cout << "SIZE OF MATRIX = " << SIZE_OF_MATRIX << std::endl;
    std::cout << "BLOCK SIZE = " << BLOCK_SIZE << std::endl;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    save_matrix_1d(matrix1, "matrix1.csv", SIZE_OF_MATRIX);
    save_matrix_1d(matrix2, "matrix2.csv", SIZE_OF_MATRIX);
    save_matrix_1d(targetMatrix, "result.csv", SIZE_OF_MATRIX);
    return 0;
}


