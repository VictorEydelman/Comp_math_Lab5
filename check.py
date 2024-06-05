def typeofread(fi, file, message, name):
    if fi:
        q = file.read()
    else:
        q = input(message)
        t = True
        while True:
            for i in range(len(name)):
                if name[i] == q:
                    t = False
                    break
            if not t:
                break
            print("Не верный ввод")
            q = input(message)
    return q


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def readabe(fi, f, ta, tb):
    if fi:
        a = float(f.read())
        b = float(f.read())
        e = float(f.read())
    else:
        a = float(input(ta))
        while not is_number(a):
            print("Ошибка вводе\n")
            a = float(input(ta))
        b = float(input(tb))
        while not is_number(b):
            print("Ошибка вводе\n")
            b = float(input(b))
        e = int(input("Введите число точек:\n"))
        while not is_number(e):
            print("Ошибка вводе\n")
            e = int(input("Введите число точек:\n"))
    return a, b, e