import numpy as np

# Функция для нахождения индексов максимального элемента матрицы (не по диагонали)
def find_max_off_diagonal(matrix):
    max_val = abs(matrix[0][1])
    row_max, col_max = 0, 1
    for row in range(len(matrix)):
        for col in range(row + 1, len(matrix)):
            if abs(matrix[row][col]) > max_val:
                max_val = abs(matrix[row][col])
                row_max = row
                col_max = col
    return row_max, col_max

# Функция для вычисления нормы вне диагональных элементов матрицы
def off_diagonal_norm(matrix):
    size = matrix.shape[0]
    norm_val = 0
    for row in range(size):
        for col in range(row + 1, size):
            norm_val += matrix[row][col]**2
    return np.sqrt(norm_val)

# Функция для нахождения собственных значений и векторов методом вращений
def jacobi_rotation(matrix, tolerance):
    size = matrix.shape[0]
    
    # Проверка симметричности матрицы
    for i in range(size):
        for j in range(i, size):
            if matrix[i][j] != matrix[j][i]:
                return None, None, -1

    working_matrix = np.copy(matrix)
    eigen_vectors = np.eye(size)
    iterations = 0
    
    # Итерационный процесс метода вращений
    while off_diagonal_norm(working_matrix) > tolerance:
        i_max, j_max = find_max_off_diagonal(working_matrix)
        if working_matrix[i_max][i_max] == working_matrix[j_max][j_max]:
            angle = np.pi / 4
        else:
            angle = 0.5 * np.arctan(2 * working_matrix[i_max][j_max] / 
                                    (working_matrix[i_max][i_max] - working_matrix[j_max][j_max]))
        
        rotation_matrix = np.eye(size)
        rotation_matrix[i_max][j_max] = -np.sin(angle)
        rotation_matrix[j_max][i_max] = np.sin(angle)
        rotation_matrix[i_max][i_max] = np.cos(angle)
        rotation_matrix[j_max][j_max] = np.cos(angle)
        
        working_matrix = rotation_matrix.T @ working_matrix @ rotation_matrix
        eigen_vectors = eigen_vectors @ rotation_matrix
        iterations += 1
    
    eigen_values = np.array([working_matrix[i][i] for i in range(size)])
    return eigen_values, eigen_vectors, iterations

# Функция для проверки результатов (A * v == λ * v)
def verify_eigen(matrix, eigen_values, eigen_vectors):
    lhs = []  # A * v
    rhs = []  # λ * v
    for i in range(matrix.shape[0]):
        lhs.append(matrix @ eigen_vectors[:, i])
        rhs.append(eigen_values[i] * eigen_vectors[:, i])
    return np.array(lhs), np.array(rhs)

# Загрузка данных из файлов
data_matrix = np.loadtxt('data/4.txt', delimiter=' ', dtype=int).reshape(3, 3)
matrix_a = data_matrix[:, :]

tolerance = np.loadtxt('data/eps.txt', dtype=float)

# Нахождение собственных значений и векторов
eigen_values, eigen_vectors, num_iterations = jacobi_rotation(matrix_a, tolerance)

# Проверка результатов
verification_lhs, verification_rhs = verify_eigen(matrix_a, eigen_values, eigen_vectors)

# Запись результатов в файл
with open('4_result.txt', 'w', encoding="utf-8") as file:
    file.write("Собственные значения:\n")
    np.savetxt(file, eigen_values, fmt='%f')
    file.write("\n")

    file.write("Собственные векторы:\n")
    for i in range(matrix_a.shape[0]):
        np.savetxt(file, eigen_vectors[:, i], fmt='%f')
        file.write("\n")

    file.write("Число итераций:\n")
    file.write(str(num_iterations))
    file.write("\n")
    file.write("\n")

    file.write("Проверка:\n")
    for i in range(matrix_a.shape[0]):
        np.savetxt(file, verification_lhs[i], fmt='%f')
        file.write("\n")
        np.savetxt(file, verification_rhs[i], fmt='%f')
        file.write("\n")