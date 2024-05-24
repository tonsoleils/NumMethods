import numpy as np
import math

# Функция для вычисления нормы вектора
def compute_norm(vector):
    result = 0
    for element in vector:
        result += element ** 2
    return math.sqrt(result)

# Функция для определения знака числа
def determine_sign(value):
    if value > 0:
        return 1
    elif value == 0:
        return 0
    else:
        return -1
    
# Функция для построения матрицы Хаусхолдера
def householder_matrix(matrix, column):
    size = matrix.shape[0]
    v_vector = np.zeros(size)
    x_vector = matrix[:, column]
    v_vector[column] = x_vector[column] + determine_sign(x_vector[column]) * compute_norm(x_vector[column:])
    for i in range(column + 1, size):
        v_vector[i] = x_vector[i]
    v_vector = v_vector[:, np.newaxis]
    H_matrix = np.eye(size) - (2 / (v_vector.T @ v_vector)) * (v_vector @ v_vector.T)
    return H_matrix

# Функция для QR-разложения матрицы
def qr_decomposition(matrix):
    size = matrix.shape[0]
    Q_matrix = np.eye(size)
    working_matrix = np.copy(matrix)
    for i in range(size - 1):
        H_matrix = householder_matrix(working_matrix, i)
        Q_matrix = Q_matrix @ H_matrix
        working_matrix = H_matrix @ working_matrix
    return Q_matrix, working_matrix

# Функция для нахождения корней характеристического полинома 2x2 подматрицы
def characteristic_roots(matrix, index):
    size = matrix.shape[0]
    a11 = matrix[index][index]
    a12 = matrix[index][index + 1] if index + 1 < size else 0
    a21 = matrix[index + 1][index] if index + 1 < size else 0
    a22 = matrix[index + 1][index + 1] if index + 1 < size else 0
    return np.roots((1, -a11 - a22, a11 * a22 - a12 * a21))

# Функция для проверки, являются ли корни комплексными
def check_complex(matrix, index, tolerance):
    Q_matrix, R_matrix = qr_decomposition(matrix)
    next_matrix = np.dot(R_matrix, Q_matrix)
    lambda1 = characteristic_roots(matrix, index)
    lambda2 = characteristic_roots(next_matrix, index)
    return abs(lambda1[0] - lambda2[0]) <= tolerance and abs(lambda1[1] - lambda2[1]) <= tolerance

# Функция для получения собственных значений
def compute_eigenvalues(matrix, index, tolerance):
    working_matrix = np.copy(matrix)
    while True:
        Q_matrix, R_matrix = qr_decomposition(working_matrix)
        working_matrix = R_matrix @ Q_matrix
        if compute_norm(working_matrix[index + 1:, index]) <= tolerance:
            return working_matrix[index][index], working_matrix
        elif compute_norm(working_matrix[index + 2:, index]) <= tolerance and check_complex(working_matrix, index, tolerance):
            return characteristic_roots(working_matrix, index), working_matrix

# Функция для нахождения всех собственных значений матрицы методом QR-разложения
def all_eigenvalues_qr(matrix, tolerance):
    size = matrix.shape[0]
    working_matrix = np.copy(matrix)
    eigen_values_list = []
    index = 0
    while index < size:
        current_eigen_values, updated_matrix = compute_eigenvalues(working_matrix, index, tolerance)
        if isinstance(current_eigen_values, np.ndarray):
            eigen_values_list.extend(current_eigen_values)
            index += 2
        else:
            eigen_values_list.append(current_eigen_values)
            index += 1
        working_matrix = updated_matrix
    return eigen_values_list

# Загрузка данных из файлов
data_matrix = np.loadtxt('data/5.txt', delimiter=' ', dtype=int).reshape(3, 3)
matrix_a = data_matrix

tolerance = np.loadtxt('data/eps.txt', dtype=float)

# Выполнение QR-разложения
Q_matrix, R_matrix = qr_decomposition(matrix_a)

# Нахождение собственных значений
eigenvalues = all_eigenvalues_qr(matrix_a, tolerance)

# Запись результатов в файл
with open('5_result.txt', 'w', encoding="utf-8") as file:
    file.write("Q:\n")
    np.savetxt(file, Q_matrix, fmt='%f')
    file.write("\n")

    file.write("R:\n")
    np.savetxt(file, R_matrix, fmt='%f')
    file.write("\n")

    file.write("Собственные значения:\n")
    np.savetxt(file, eigenvalues, fmt='%f')
    file.write("\n")