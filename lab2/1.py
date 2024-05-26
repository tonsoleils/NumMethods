import math

# Метод простой итерации
def simple_iteration_method(a, b, tolerance):
    # Определяем функцию g(x), используемую в методе простой итерации
    def g(x):
        return math.log10(x + 1) + 0.5
    
    # Начальное приближение для метода простой итерации
    initial_guess = (a + b) / 2
    x_prev = initial_guess
    k = 0
    print("Простая итерация:")
    print("k\t x\t\t\t f(x)")  # Заголовок для вывода значений итерации
    while True:
        k += 1
        # Вычисляем следующее приближение x и значение функции f(x)
        x_next = g(x_prev)
        f_x = math.log10(x_next + 1) - x_next + 0.5
        # Выводим значения итерации
        print(f"{k}\t {x_next:.10f}\t {f_x:.10f}")
        # Проверяем достижение критерия останова
        if abs(x_next - x_prev) < tolerance:
            return x_next
        x_prev = x_next

# Метод Ньютона
def newton_method(a, b, tolerance):
    # Определяем функцию f(x) и ее производную f'(x), используемые в методе Ньютона
    def f(x):
        return math.log10(x + 1) - x + 0.5
    
    def f_prime(x):
        return 1 / ((x + 1) * math.log(10)) - 1
    
    # Начальное приближение для метода Ньютона
    initial_guess = (a + b) / 2
    x_prev = initial_guess
    k = 0
    print("Метод Ньютона:")
    print("k\t x\t\t f(x)\t\t f'(x)\t -f(x)/f'(x)")  # Заголовок для вывода значений итерации
    while True:
        k += 1
        # Вычисляем значение функции f(x), ее производной f'(x) и следующее приближение x
        f_x = f(x_prev)
        f_prime_x = f_prime(x_prev)
        if f_prime_x == 0:
            print("Деление на ноль.")
            return None
        x_next = x_prev - f_x / f_prime_x
        # Выводим значения итерации
        print(f"{k}\t {x_next:.10f}\t {f_x:.10f}\t {f_prime_x:.10f}\t {-f_x / f_prime_x:.10f}")
        # Проверяем достижение критерия останова
        if abs(x_next - x_prev) < tolerance:
            return x_next
        x_prev = x_next

# Задаем отрезок [a, b] и точность вычислений
a = 0.1  # Начало отрезка
b = 1.0  # Конец отрезка
tolerance = 1e-6     # Точность вычислений

# Вызываем метод простой итерации
result_simple_iteration = simple_iteration_method(a, b, tolerance)
print("Решение методом простой итерации:", result_simple_iteration)

# Вызываем метод Ньютона
result_newton = newton_method(a, b, tolerance)
print("Решение методом Ньютона:", result_newton)
