#include <iostream>
#include <vector>
#include <chrono>
#include "test_helper.h"
#include "matrix_generator.h"

int main()
{
    int SIZE_OF_MATRIX = 1024;
    std::vector<std::vector<int>> matrix1 = generate_matrix_2d(SIZE_OF_MATRIX, 1);
    std::vector<std::vector<int>> matrix2 = generate_matrix_2d(SIZE_OF_MATRIX, 1);
    std::vector<std::vector<int>> targetMatrix(SIZE_OF_MATRIX, std::vector<int>(SIZE_OF_MATRIX));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < SIZE_OF_MATRIX; i++)
    {
        for (int j = 0; j < SIZE_OF_MATRIX; j++)
        {
            for (int k = 0; k < SIZE_OF_MATRIX; k++)
            {
                targetMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
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