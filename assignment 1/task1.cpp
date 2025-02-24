#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <fstream>

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