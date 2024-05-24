import numpy as np

# Функция для решения системы линейных уравнений методом прогонки
def progon(a, b):
    n = len(a)  # Количество уравнений
    P = np.zeros(n)  # Вспомогательный массив P
    Q = np.zeros(n)  # Вспомогательный массив Q

    # Начальные значения для P и Q
    P[0] = -a[0, 1] / a[0, 0]
    Q[0] = b[0] / a[0, 0]

    # Прямой ход метода прогонки
    for i in range(1, n):
        denominator = a[i, i] + a[i, i - 1] * P[i - 1]
        if i == n - 1:
            P[i] = 0
        else:
            P[i] = -a[i, i + 1] / denominator
        Q[i] = (b[i] - a[i, i - 1] * Q[i - 1]) / denominator

    x = np.zeros(n)  # Массив для хранения решения
    x[n - 1] = Q[n - 1]  # Начальное значение для обратного хода

    # Обратный ход метода прогонки
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x

# Загрузка данных из файла
data = np.loadtxt('data/2.txt', delimiter=' ')
a = data[:, :-1]  # Матрица коэффициентов
b = data[:, -1]  # Вектор свободных членов

# Решение системы методом прогонки
x = progon(a, b)

# Запись результата в файл
with open('2_result.txt', 'w', encoding="utf-8") as file:
    file.write("Решение СЛАУ методом прогонки:\n")
    np.savetxt(file, x, fmt='%d')