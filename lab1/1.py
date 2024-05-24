import numpy as np

# Функция для разложения матрицы a на матрицу LU
def decompose_to_LU(a):
    # Инициализация нулевой матрицы LU
    lu = np.zeros([a.shape[0], a.shape[1]])
    n = a.shape[0]
    # Проходим по всем элементам матрицы
    for k in range(n):
        for j in range(k, n):
            # Заполняем верхнюю треугольную матрицу U
            lu[k, j] = a[k, j] - np.dot(lu[k, :k], lu[:k, j]).item()
        for i in range(k + 1, n):
            # Заполняем нижнюю треугольную матрицу L
            lu[i, k] = (a[i, k] - np.dot(lu[i, :k], lu[:k, k]).item()) / lu[k, k].item()
    return lu

# Функция для извлечения нижней треугольной матрицы L из LU
def get_L(lu):
    L = np.tril(lu, k=-1) + np.eye(lu.shape[0])
    return L

# Функция для извлечения верхней треугольной матрицы U из LU
def get_U(lu):
    U = np.triu(lu)
    return U

# Функция для построения перестановочной матрицы P
def get_P(lu):
    n = lu.shape[0]
    P = np.eye(n)
    k = 0
    while lu[k, k] == 0:
        k += 1
    for i in range(n):
        if i != k and lu[i, k] != 0:
            P[[i, k]] = P[[k, i]]
    return P

# Функция для разложения матрицы a на матрицы L, U и P (LUP-разложение)
def decompose_to_LUP(a):
    n = a.shape[0]
    P = np.eye(n)
    L = np.zeros((n, n))
    U = np.copy(a)

    for k in range(n - 1):
        # Находим строку с максимальным элементом для выбора ведущего элемента
        pivot_index = np.argmax(np.abs(U[k:, k])) + k
        if pivot_index != k:
            # Меняем строки местами в матрицах U и P
            U[[k, pivot_index]] = U[[pivot_index, k]]
            P[[k, pivot_index]] = P[[pivot_index, k]]
            if k > 0:
                L[[k, pivot_index], :k] = L[[pivot_index, k], :k]

        # Обновляем матрицы L и U
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
    
    L += np.eye(n)
    return L, U, P

# Функция для решения системы линейных уравнений с использованием разложения LU
def solve(lu, b):
    n = lu.shape[0]
    y = np.zeros([n, 1])
    x = np.zeros([n, 1])
    # Прямой ход для нахождения y
    for i in range(n):
        y[i, 0] = b[i, 0] - np.dot(lu[i, :i], y[:i, 0]).item()
    # Обратный ход для нахождения x
    for i in range(n - 1, -1, -1):
        x[i, 0] = (y[i, 0] - np.dot(lu[i, i+1:], x[i+1:, 0]).item()) / lu[i, i].item()
    return x

# Функция для вычисления определителя матрицы с использованием разложения LU
def determinant(lu):
    det = np.prod(np.diagonal(lu))
    return det

# Функция для вычисления обратной матрицы с использованием разложения LU
def inverse(lu):
    n = lu.shape[0]
    inv = np.zeros_like(lu)
    for i in range(n):
        e = np.zeros([n, 1])
        e[i, 0] = 1
        inv[:, i] = solve(lu, e)[:, 0]
    return inv

# Функция для проверки корректности вычисления обратной матрицы
def check(a, inv):
    ch = np.matmul(a, inv)
    return ch

if __name__ == "__main__":
    # Чтение данных из файла
    data = np.loadtxt('data/1.txt', delimiter=' ')
    a = np.matrix(data[:, :-1])
    b = np.matrix(data[:, -1]).reshape(-1, 1)

    # Разложение матрицы на LU
    lu = decompose_to_LU(a)
    # Разложение матрицы на L, U и P (LUP-разложение)
    L, U, P = decompose_to_LUP(a)
    # Проверка произведения L и U
    ch1 = np.matmul(L, U)
    # Решение системы линейных уравнений
    x = solve(lu, b)
    # Вычисление определителя матрицы
    det = determinant(lu)
    # Вычисление обратной матрицы
    inv = inverse(lu)
    # Проверка произведения матрицы и ее обратной
    ch = check(a, inv)

    # Запись результатов в файл
    with open('1_result.txt', 'w', encoding="utf-8") as file:
        file.write("Разложение LUP:\n")
        np.savetxt(file, lu, fmt='%f')
        file.write("\n")

        file.write("Матрица L:\n")
        np.savetxt(file, L, fmt='%f')
        file.write("\n")

        file.write("Матрица U:\n")
        np.savetxt(file, U, fmt='%f')
        file.write("\n")

        file.write("Матрица P:\n")
        np.savetxt(file, P, fmt='%f')
        file.write("\n")

        file.write("Произведение L * U:\n")
        np.savetxt(file, ch1)
        file.write("\n")

        file.write("Решение СЛАУ:\n")
        np.savetxt(file, x, fmt='%f')
        file.write("\n")

        file.write("Определитель матрицы:\n")
        file.write(str(det))
        file.write("\n")

        file.write("Обратная матрица:\n")
        np.savetxt(file, inv)
        file.write("\n")

        file.write("Произведение матрицы и ее обратной:\n")
        np.savetxt(file, ch)