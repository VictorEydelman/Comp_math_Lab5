from tabulate import tabulate
import os.path
from sympy import diff, symbols, cos, sin, sqrt, lambdify, sympify, tan, exp, pi, solve, ln
from check import typeofread, readabe, is_number
import matplotlib.pyplot as plt
import numpy as np

x, y, xv, yv = symbols('x y xv yv')


def vidyravnenia(num):
    if num == 1:
        return cos(x)
    if num == 2:
        return x ** 2 - 5.41 * x - 11.76


u = typeofread(False, "", "Выберете откуда вводите информацию: для ввода с клавиатуры напишите 'клавиатура',"
                          " для ввода из файла напишите 'файл', для выбора функции напишите 'функция':\n",
               ["файл", "клавиатура", "функция"])
fi = False
if u == "файл":
    fi = True
    file = input("Название файла из которого считываются данные\n")
    while not os.path.exists(file):
        print("Файла не существует")
        file = input("Название файла из которого считываются данные\n")
if fi:
    fil = open(file, 'r')
if fi:
    x1 = list(map(float, fil.readline().split(" ")))
    y1 = list(map(float, fil.readline().split(" ")))
else:
    if u == "функция":
        typeoffunction = int(typeofread(False, "", "Выберете функцию (введите номер):\n" +
                                        "1." + str(vidyravnenia(1)) + "\n2." + str(vidyravnenia(2)) + "\n", ["1", "2"]))
        fun = vidyravnenia(typeoffunction)
        func = lambdify(x, fun)
        a, b, n = readabe(False, "", "Введите крайне левое значение интервала:\n",
                          "Введите крайне правое значение интервала:\n")
        x1 = [0] * n
        y1 = [0] * n
        for i in range(n):
            x1[i] = a + (b - a) / (n - 1) * i
            y1[i] = func(x1[i])
    else:
        print("Введите x:")
        x1 = list(map(float, input().split()))
        print("Введите y:")
        y1 = list(map(float, input().split()))

L = 0
n = len(x1)

for i in range(n):
    l = 1
    for j in range(n):
        if j != i:
            l *= (x - x1[j]) / (x1[i] - x1[j])
    L += y1[i] * l
Ln = lambdify(x, L)


def f(i, x1, y1):
    if len(i) == 1:
        return y1[i[0]]
    else:
        return (f(i[1:], x1, y1) - f(i[:-1], x1, y1)) / (x1[i[-1]] - x1[i[0]])


N = y1[0]
xn = [0]
l = 1
for i in range(1, n):
    xn.append(i)
    l *= (x - x1[i - 1])
    N += f(xn, x1, y1) * l
N3 = lambdify(x, N)
dy = [y1]
for i in range(n - 1):
    d = [0] * n
    for j in range(n):
        if j < n - i - 1:
            if i == 0:
                d[j] = y1[j + 1] - y1[j]
            else:
                d[j] = dy[i][j + 1] - dy[i][j]
        else:
            d[j] = 0
    dy.append(d)
t = True
for i in range(n - 2):
    if round(x1[i + 1] - x1[i], 4) != round(x1[i + 2] - x1[i + 1], 4):
        t = False
        break

NL=0
tb = (x - x1[0]) / (x1[1] - x1[0])
tn = 1
for j in range(n):
    NL += dy[j][0] * tn
    tn *= (tb - j) / (j + 1)
NR = 0
tb = (x - x1[n-1]) / (x1[n-1] - x1[n-2])
tn = 1
for j in range(n):
    if n-2- j + 1 >= 0:
        NR += dy[j][n-2- j + 1] * tn
        tn *= (tb+j) / (j + 1)

if fi:
    znach = float(fil.read())
else:
    znach = float(input("Введите значение аргумента:"))
print(f'Значение многочелена Лагранжа в задной точке: {Ln(znach)}')
if not t:
    print(f'Значение многочелена Ньютона с разделенными разностями в задной точке: {N3(znach)}')
head = [""] * n
head[0] = "y"
head[1] = "dy"
for i in range(2, n):
    head[i] = f'd{i}y'
if t:
    NRf = lambdify(x, NR)
    NLf = lambdify(x, NL)
    if znach<=(x1[0]+x1[n-1])/2:
        print(f'Значение многочелена Ньютона с конечными разностями первая формула в задной точке: {NLf(znach)}')
    else:
        print(f'Значение многочелена Ньютона с конечными разностями вторая формулав задной точке: {NRf(znach)}')
    print(tabulate(np.transpose(dy), head))
else:
    print(tabulate(np.transpose(dy), head))
    print("Значения x не являются равноотстоящими")
x2132 = np.arange(min(x1) * 0.9, max(x1) * 1.1, 0.0001)
plt.plot(x2132, Ln(x2132), label="Многочелена Лагранжа")
plt.scatter(znach, Ln(znach))
if t:
    NLf = lambdify(x, NL)
    NRf = lambdify(x, NR)
    if znach<=(x1[0]+x1[n-1])/2:
        plt.plot(x2132, NLf(x2132), color="green",label="Многочелена Ньютона с конечными разностями первая формула")
        plt.scatter(znach, NLf(znach))
    else:
        plt.plot(x2132, NRf(x2132), color="red",label="Многочелена Ньютона с конечными разностями вторая формула")
        plt.scatter(znach, NRf(znach))
else:
    plt.plot(x2132, N3(x2132), label="Многочелена Ньютона с разделенными разностями")
    plt.scatter(znach, N3(znach))
plt.scatter(x1, y1)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()