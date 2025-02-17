#include <iostream>
#include <vector>
#include <chrono>

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    int SIZE_OF_MATRIX = 1000;
    
    std::vector<std::vector<int>> matrix1(SIZE_OF_MATRIX, std::vector<int>(SIZE_OF_MATRIX));
    std::vector<std::vector<int>> matrix2(SIZE_OF_MATRIX, std::vector<int>(SIZE_OF_MATRIX));
    // matrix1 @ matrix2 = targetMatrix
    std::vector<std::vector<int>> targetMatrix(SIZE_OF_MATRIX, std::vector<int>(SIZE_OF_MATRIX));
    
    // initialize matrix1 and matrix2 with (random) numbers
    for (int i = 0; i < SIZE_OF_MATRIX; i++) {
        for(int j = 0; j < SIZE_OF_MATRIX; j++) {
            matrix1[i][j] = 1; // random number
            matrix2[i][j] = 1; // random number
            targetMatrix[i][j] = 0;
        }
    }

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
 
    // Stop measuring time and calculate duration
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << targetMatrix[0][0] << std::endl;
    // Print execution time
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    return 0;
}