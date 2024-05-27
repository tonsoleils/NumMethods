import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import math

# Параметр уравнений
a = 2

# Создание массива значений x2 от 0 до 4 с 400 равными интервалами
x2 = np.linspace(0, 4, 400)

# Вычисление значений x1_1 и x1_2 для заданного уравнения
x1_1 = np.sqrt((x2 - np.power(x2, 2) + a) / (2 * a))
x1_2 = np.sqrt(x2 + a) - 1

# Создание графиков для визуализации уравнений
fig, axs = plt.subplots(2, 1, figsize=(8, 12))

# Построение графика первого уравнения
axs[0].plot(x1_1, x2, label='2x1^2 - x2 + x2^2 - a')
# Построение графика второго уравнения
axs[0].plot(x1_2, x2, label='x1 - sqrt(x2 + a) + 1')
axs[0].legend()

# Создание массива значений x1 от 0 до 2 с 400 равными интервалами
x1 = np.linspace(0, 2, 400)
# Создание массива значений x2 от 0 до 2 с 400 равными интервалами
x2 = np.linspace(0, 2, 400)

# Вычисление значений для функций fi1 и fi2
fi1_i = 0.5 * np.power((x2 - np.power(x2, 2) + a), 0.5)
fi2_i = np.sqrt(x2 + a) - 1

# Построение графика функций fi1 и fi2
axs[1].plot(x2, fi1_i, label='fi1')
axs[1].plot(x1, fi2_i, label='fi2')
axs[1].legend()

# Настройка макета и отображение графиков
plt.tight_layout()
plt.show()

# Определение функции fi1
def fi1(x1, x2):
    return (x2 - np.power(x2, 2) + a) / (2 * a)

# Определение функции fi2
def fi2(x1, x2):
    return np.sqrt(x2 + a) - 1

# Реализация метода простой итерации
def prost_iter(q, x0_1, x0_2, eps):
    k = 0
    x1 = x0_1
    x2 = x0_2
    f1 = fi1(x1, x2)
    f2 = fi2(x1, x2)
    results = [(k, x1, x2, f1, f2)]
    # Итерационный процесс для нахождения корней уравнения
    while max(abs(x1 - f1), abs(x2 - f2)) * q / (1 - q) > eps:
        x1 = f1
        f1 = fi1(x1, x2)
        x2 = f2
        f2 = fi2(x1, x2)
        k += 1
        results.append((k, x1, x2, f1, f2))
    print(tabulate(results, headers=["k", "x1", "x2", "fi1(x1, x2)", "fi2(x1, x2)"], tablefmt="grid"))

# Определение функции F1
def F1(x1, x2):
    return 2 * np.power(x1, 2) - x2 + np.power(x2, 2) - a

# Определение функции F2
def F2(x1, x2):
    return x1 - np.sqrt(x2 + a) + 1

# Производная функции F1 по x1
def F1_pr1(x1, x2):
    return 4 * x1

# Производная функции F1 по x2
def F1_pr2(x1, x2):
    return -1 + 2 * x2

# Производная функции F2 по x1
def F2_pr1(x1, x2):
    return 1

# Производная функции F2 по x2
def F2_pr2(x1, x2):
    return -1 / (2 * np.sqrt(x2 + a))

# Реализация метода Ньютона
def newton(x0_1, x0_2, eps):
    x1 = x0_1
    x2 = x0_2
    k = 0
    f1 = F1(x1, x2)
    f2 = F2(x1, x2)
    f1_pr1 = F1_pr1(x1, x2)
    f1_pr2 = F1_pr2(x1, x2)
    f2_pr1 = F2_pr1(x1, x2)
    f2_pr2 = F2_pr2(x1, x2)
    d_a1 = f1 * f2_pr2 - f2 * f1_pr2
    d_a2 = f2 * f1_pr1 - f1 * f2_pr1
    d_j = f1_pr1 * f2_pr2 - f1_pr2 * f2_pr1
    results = [(k, x1, x2, f1, f2, f1_pr1, f1_pr2, f2_pr1, f2_pr2, d_a1, d_a2, d_j)]
    # Итерационный процесс для нахождения корней уравнения
    while max(abs(d_a1 / d_j), abs(d_a2 / d_j)) > eps:
        x1 = x1 - d_a1 / d_j
        x2 = x2 - d_a2 / d_j
        f1 = F1(x1, x2)
        f2 = F2(x1, x2)
        f1_pr1 = F1_pr1(x1, x2)
        f1_pr2 = F1_pr2(x1, x2)
        f2_pr1 = F2_pr1(x1, x2)
        f2_pr2 = F2_pr2(x1, x2)
        d_a1 = f1 * f2_pr2 - f2 * f1_pr2
        d_a2 = f2 * f1_pr1 - f1 * f2_pr1
        d_j = f1_pr1 * f2_pr2 - f1_pr2 * f2_pr1
        k += 1
        results.append((k, x1, x2, f1, f2, f1_pr1, f1_pr2, f2_pr1, f2_pr2, d_a1, d_a2, d_j))
    print(tabulate(results, headers=["k", "x1", "x2", "f1", "f2", "f1_pr1", "f1_pr2", "f2_pr1", "f2_pr2", "d_a1", "d_a2", "d_j"], tablefmt="grid"))

# Параметры метода простой итерации и метода Ньютона
q = 0.8
eps = 0.000001
x0_1 = 0.4
x0_2 = 1

# Выполнение метода простой итерации
print("Prost Iteration:")
prost_iter(q, x0_1, x0_2, eps)
print("\nNewton's Method:")
# Выполнение метода Ньютона
newton(x0_1, x0_2, eps)
