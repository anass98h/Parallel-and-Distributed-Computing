#ifndef TEST_HELPER_H
#define TEST_HELPER_H

#include <vector>
#include <string>
#include <fstream>

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

#endif