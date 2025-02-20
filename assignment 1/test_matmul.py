import numpy as np
import os
import time

def read_csv_to_matrix(filename):
    return np.genfromtxt(filename, delimiter=',')


current_folder = os.path.dirname(__file__)
matrix1 = read_csv_to_matrix(os.path.join(current_folder, 'output/tests/matrix1.csv'))
matrix2 = read_csv_to_matrix(os.path.join(current_folder, 'output/tests/matrix2.csv'))
result_matrix = read_csv_to_matrix(os.path.join(current_folder, 'output/tests/result.csv'))


matrix1 = matrix1[:, :-1]
matrix2 = matrix2[:, :-1]
result_matrix = result_matrix[:, :-1]

assert matrix1.shape == matrix2.shape == result_matrix.shape

# Check numpy matmul result and execution time
ms_factor = 1000
start_time = time.time()
calculated_result = np.matmul(matrix1, matrix2)
end_time = time.time()
execution_time = (end_time - start_time) * ms_factor

assert np.allclose(calculated_result, result_matrix)

print("The result of the matrix multiplication is correct (matches the result of the numpy.matmul())")
print(f"Execution time of numpy matmul: {execution_time:.2f} ms")
print("----------------------------------------")