import numpy as np
import math

# Функция для вычисления нормы вектора или матрицы
def norm(a):
    return np.linalg.norm(a)

# Функция для решения системы линейных уравнений методом простых итераций
def iter(a, b, eps):
    # Проверка диагонального преобладания
    diag_sum = abs(a[0][0]) + abs(a[0][1]) + abs(a[-1][-1]) + abs(a[-1][-2])
    for i in range(1, len(a)-1):
        diag_sum += abs(a[i][i-1]) + abs(a[i][i]) + abs(a[i][i+1])
    full_sum = 0
    for i in range(len(a)):
        for j in range(len(a[i])):
            full_sum += abs(a[i][j])
    if diag_sum <= (full_sum - diag_sum) * 2:
        return None, -1

    n = a.shape[0]  # Размерность матрицы
    alpha = np.zeros_like(a, dtype='float')  # Матрица коэффициентов альфа
    beta = np.zeros_like(b, dtype='float')  # Вектор свободных членов бета

    # Вычисление матриц альфа и бета
    for i in range(n):
        for j in range(n):
            if i == j:
                alpha[i][j] = 0
            else:
                alpha[i][j] = -a[i][j] / a[i][i]
        beta[i] = b[i] / a[i][i]

    iter = 0  # Счетчик итераций
    cur_x = np.copy(beta)  # Начальное приближение
    n_a = norm(alpha)  # Норма матрицы альфа
    n_b = norm(beta)  # Норма вектора бета

    # Итерационный процесс
    while iter <= 50:
        prev_x = np.copy(cur_x)
        cur_x = alpha @ prev_x + beta
        iter += 1
        if norm(cur_x - prev_x) * (n_a / (1 + n_a)) <= eps:
            return cur_x, iter  # Возврат результата и числа итераций
    return None, -1

# Функция для умножения матрицы alpha на вектор x и добавления вектора beta
def multiplication(alpha, x, beta):
    res = np.copy(x)
    for i in range(alpha.shape[0]):
        res[i] = beta[i]
        for j in range(alpha.shape[1]):
            res[i] += alpha[i][j] * res[j]
    return res

# Функция для решения системы линейных уравнений методом Зейделя
def zeidel(a, b, eps):
    n = a.shape[0]  # Размерность матрицы
    alpha = np.zeros_like(a, dtype='float')  # Матрица коэффициентов альфа
    beta = np.zeros_like(b, dtype='float')  # Вектор свободных членов бета

    # Вычисление матриц альфа и бета
    for i in range(n):
        for j in range(n):
            if i == j:
                alpha[i][j] = 0
            else:
                alpha[i][j] = -a[i][j] / a[i][i]
        beta[i] = b[i] / a[i][i]

    iter = 0  # Счетчик итераций
    cur_x = np.copy(beta)  # Начальное приближение
    C = np.triu(alpha)  # Верхняя треугольная часть матрицы альфа
    n_c = norm(C)  # Норма матрицы C
    n_a = norm(alpha)  # Норма матрицы альфа

    # Итерационный процесс
    while iter <= 50:
        prev_x = np.copy(cur_x)
        cur_x = multiplication(alpha, prev_x, beta)
        iter += 1
        if norm(cur_x - prev_x) * n_c / (1 - n_a) <= eps:
            return cur_x, iter  # Возврат результата и числа итераций
    return None, -1

# Загрузка данных из файлов
data = np.loadtxt('data/3.txt', delimiter=' ')
a = data[:, :-1]  # Матрица коэффициентов
b = data[:, -1]  # Вектор свободных членов
eps = np.loadtxt('data/eps.txt')  # Точность

# Решение системы методом простых итераций
x1, k1 = iter(a, b, eps)

# Решение системы методом Зейделя
x2, k2 = zeidel(a, b, eps)

# Запись результатов в файл
with open('3_result.txt', 'w', encoding="utf-8") as file:
    file.write("Решение СЛАУ методом итераций:\n")
    np.savetxt(file, x1, fmt='%f')
    file.write("\n")

    file.write("Число итераций:\n")
    file.write(str(k1))
    file.write("\n")

    file.write("Решение СЛАУ методом Зейделя:\n")
    np.savetxt(file, x2, fmt='%f')
    file.write("\n")

    file.write("Число итераций:\n")
    file.write(str(k2))