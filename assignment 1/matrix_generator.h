#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include <vector>
#include <random>

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

#endif
