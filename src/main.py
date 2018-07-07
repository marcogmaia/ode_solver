# see: https://faculty.washington.edu/heathml/2018spring307/euler_method.pdf
# the Inverse Euler method is wrong because the book solves the implicit equation
# I should fix this...
import matplotlib.pyplot as plt
import parser
import numpy
from math import *


class Methods:
    def __init__(self, x0, y0, xf, h, edoFunc):
        self.method_name = None
        self.x0 = x0
        self.y0 = y0
        self.xf = xf
        self.n = int((self.xf - self.x0)/h) + 1
        self.x = numpy.linspace(self.x0, self.xf, self.n)
        self.y = numpy.zeros(self.n)
        self.yf = None       # this variable will receive the final results
        self.y[0] = self.y0
        self.h = h
        self.func = parser.expr(edoFunc).compile()

    def f(self, x, y):
        return eval(self.func)

    def euler(self):
        self.method_name = 'Euler'
        y = self.y
        for i in range(1, self.n):
            y[i] = y[i - 1] + self.h*self.f(self.x[i - 1], y[i - 1])
        self.yf = y
        # print(self.yf[-1])
        # return y

    # the implicit equation needs to be solved.
    def eulerInverse(self):
        y = self.y
        self.euler() # yf contains the return of the euler method.
        self.method_name = 'Backwards Euler'
        yp = self.yf
        for i in range(1, self.n):
            y[i] = y[i - 1] + self.h*self.f(self.x[i], yp[i])
        self.yf = y
        # print(y[self.n - 1])

    def eulerModified(self):
        self.method_name = 'Improved Euler'
        y = self.y
        for i in range(1, self.n):
            k1 = self.h*self.f(self.x[i - 1], y[i - 1])
            k2 = self.h*self.f(self.x[i], y[i - 1] + k1)
            y[i] = y[i - 1] + (k1 + k2)/2
        self.yf = y
        # print(self.x[-1], y[-1])

    def rungeKutta(self):
        self.method_name = 'Runge Kutta'
        y = self.y
        for i in range(1, self.n):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 1] + 0.5*self.h, k1*self.h*0.5 + y[i - 1])
            k3 = self.f(self.x[i - 1] + 0.5*self.h, k2*self.h*0.5 + y[i - 1])
            k4 = self.f(self.x[i - 1] + self.h, k3*self.h + y[i - 1])
            y[i] = y[i - 1] + self.h*(k1 + 2*k2 + 2*k3 + k4)/6
        self.yf = y
        # print(y[-1])

    def rungeKutta4(self):
        self.method_name = 'Runge Kutta 4'
        y = self.y
        for i in range(1, self.n):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 1] + self.h*0.5, y[i - 1] + self.h*0.5*k1)
            k3 = self.f(self.x[i - 1] + self.h*0.5, y[i - 1] + self.h*0.5*k2)
            k4 = self.f(self.x[i - 1] + self.h, y[i - 1] + self.h*k3)
            y[i] = y[i - 1] + self.h*(k1 + 2*k2 + 2*k3 + k4)/6
        self.yf = y
        # print(y[self.n - 1])

    def rungeKutta5(self):
        self.method_name = 'Runge Kutta 5'
        y = self.y
        for i in range(1, self.n):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 1] + self.h/4, y[i - 1] + self.h/4*k1)
            k3 = self.f(
                self.x[i - 1] + self.h/4, y[i - 1] + self.h/8*k1 + self.h/8*k2
            )
            k4 = self.f(
                self.x[i - 1] + self.h/2, y[i - 1] - self.h/2*k2 + self.h*k3
            )
            k5 = self.f(
                self.x[i - 1] + self.h*3/4,
                y[i - 1] + self.h*3/16*k1 + self.h*9/16*k4
            )
            k6 = self.f(
                self.x[i - 1] + self.h, y[i - 1] +
                self.h*(-3/7*k1 + 2/7*k2 + 12/7*k3 - 12/7*k4 + 8/7*k5)
            )
            y[i] = y[i - 1] + self.h*(7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6)/90
        self.yf = y
        # print(y[self.n - 1])

    def rungeKutta6(self):
        self.method_name = 'Runge Kutta 6'
        y = self.y
        for i in range(1, self.n):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 1] + self.h, y[i - 1] + k1*self.h)
            k3 = self.f(
                self.x[i - 1] + self.h/2, y[i - 1] + ((3*k1 + k2)/8)*self.h
            )
            k4 = self.f(
                self.x[i - 1] + self.h*2/3,
                y[i - 1] + ((8*k1 + 2*k2 + 8*k3)/27)*self.h
            )
            k5 = self.f(
                self.x[i - 1] + self.h*(7 - 21**(1/2))/14, y[i - 1] + (
                    (
                        (3*(3*21**(1/2) - 7))*k1 - (8*(7 - 21**(1/2)))*k2 +
                        (48*(7 - 21**(1/2)))*k3 - (3*(21 - 21**(1/2)))*k4
                    )/392
                )*self.h
            )
            k6 = self.f(
                self.x[i - 1] + self.h*(7 + 21**(1/2))/14, y[i - 1] + (
                    (
                        -(5*(231 + 51*21**(1/2)))*k1 - (40*(7 + 21**
                                                            (1/2)))*k2 -
                        (320*21**
                         (1/2))*k3 + (3*(21 + 121*21**
                                         (1/2)))*k4 + (392*(6 + 21**(1/2)))*k5
                    )/1960
                )*self.h
            )
            k7 = self.f(
                self.x[i - 1] + self.h, y[i - 1] + (
                    (
                        (15*(22 + 7*21**(1/2)))*k1 + (120)*k2 +
                        (40*(7*21**(1/2) - 5))*k3 - (63*(3*21**(1/2) - 2))*k4 -
                        (14*(49 + 9*21**(1/2)))*k5 + (70*(7 - 21**(1/2)))*k6
                    )/180
                )*self.h
            )
            y[i] = y[i - 1] + self.h*(9*k1 + 64*k3 + 49*k5 + 49*k6 + 9*k7)/180
        self.yf = y
        # print(y[self.n - 1])

    def adamsBash2(self):
        self.method_name = 'Adams Bashforth 2'
        y = self.y
        c1 = self.h*self.f(self.x[0], y[0])
        c2 = self.h*self.f(self.x[1], y[0] + c1)
        y[1] = y[0] + (c1 + c2)/2
        for i in range(2, self.n):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 2], y[i - 2])
            y[i] = y[i - 1] + self.h*(3*k1 - k2)/2
        self.yf = y
        # print(y[self.n - 1])

    def adamsBash3(self):
        self.method_name = 'Adams Bashforth 3'
        y = self.y
        for i in range(1, 3):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 1]*self.h*0.5, y[i - 1]*self.h*0.5*k1)
            k3 = self.f(
                self.x[i - 1]*self.h, y[i - 1] - k1*self.h* +2*k2*self.h
            )
            y[i] = y[i - 1] + self.h*(k1 + 4*k2 + k3)/6
        for i in range(3, self.n):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 2], y[i - 2])
            k3 = self.f(self.x[i - 3], y[i - 3])
            y[i] = y[i - 1] + self.h*(23/12*k1 - 4/3*k2 + 5/12*k3)
        self.yf = y
        # print(y[self.n - 1])

    def adamsBash4(self):
        self.method_name = 'Adams Bashforth 4'
        y = self.y
        for i in range(1, 4):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 1] + self.h*0.5, y[i - 1] + self.h*0.5*k1)
            k3 = self.f(self.x[i - 1] + self.h*0.5, y[i - 1] + self.h*0.5*k2)
            k4 = self.f(self.x[i - 1] + self.h, y[i - 1] + self.h*k3)
            y[i] = y[i - 1] + self.h*(k1 + 2*k2 + 2*k3 + k4)/6
        for i in range(4, self.n):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 2], y[i - 2])
            k3 = self.f(self.x[i - 3], y[i - 3])
            k4 = self.f(self.x[i - 4], y[i - 4])
            y[i] = y[i - 1] + self.h*(55/24*k1 - 59/24*k2 + 37/24*k3 - 3/8*k4)
        self.yf = y
        # print(y[self.n - 1])

    def adamsBash5(self):
        self.method_name = 'Adams Bashforth 5'
        y = self.y
        for i in range(1, 5):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 1] + self.h/4, y[i - 1] + self.h/4*k1)
            k3 = self.f(
                self.x[i - 1] + self.h/4, y[i - 1] + self.h/8*k1 + self.h/8*k2
            )
            k4 = self.f(
                self.x[i - 1] + self.h/2, y[i - 1] - self.h/2*k2 + self.h*k3
            )
            k5 = self.f(
                self.x[i - 1] + self.h*3/4,
                y[i - 1] + self.h*3/16 + self.h*9/16*k4
            )
            k6 = self.f(
                self.x[i - 1] + self.h,
                y[i - 1] + self.h*(-3/7*k1 + 2/7 + 12/7*k3 - 12/7*k4 + 8/7*k5)
            )
            y[i] = y[i - 1] + self.h*(7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6)/90
        for i in range(5, self.n):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 2], y[i - 2])
            k3 = self.f(self.x[i - 3], y[i - 3])
            k4 = self.f(self.x[i - 4], y[i - 4])
            k5 = self.f(self.x[i - 5], y[i - 5])
            y[i] = y[i - 1] + self.h*(
                1901/720*k1 - 1387/360*k2 + 109/30*k3 - 637/360*k4 + 251/720*k5
            )
        self.yf = y
        # print(y[self.n - 1])

    def adamsBash6(self):
        self.method_name = 'Adams Bashforth 6'
        y = self.y
        for i in range(1, 6):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 1] + self.h, y[i - 1] + k1*self.h)
            k3 = self.f(
                self.x[i - 1] + self.h/2, y[i - 1] + ((3*k1 + k2)/8)*self.h
            )
            k4 = self.f(
                self.x[i - 1] + self.h*2/3,
                y[i - 1] + ((8*k1 + 2*k2 + 8*k3)/27)*self.h
            )
            k5 = self.f(
                self.x[i - 1] + self.h*(7 - 21**(1/2))/14, y[i - 1] + (
                    (
                        (3*(3*21**(1/2) - 7))*k1 - (8*(7 - 21**(1/2)))*k2 +
                        (48*(7 - 21**(1/2)))*k3 - (3*(21 - 21**(1/2)))*k4
                    )/392
                )*self.h
            )
            k6 = self.f(
                self.x[i - 1] + self.h*(7 + 21**(1/2))/14, y[i - 1] + (
                    (
                        -(5*(231 + 51*21**(1/2)))*k1 - (40*(7 + 21**
                                                            (1/2)))*k2 -
                        (320*21**
                         (1/2))*k3 + (3*(21 + 121*21**
                                         (1/2)))*k4 + (392*(6 + 21**(1/2)))*k5
                    )/1960
                )*self.h
            )
            k7 = self.f(
                self.x[i - 1] + self.h, y[i - 1] + (
                    (
                        (15*(22 + 7*21**(1/2)))*k1 + (120)*k2 +
                        (40*(7*21**(1/2) - 5))*k3 - (63*(3*21**(1/2) - 2))*k4 -
                        (14*(49 + 9*21**(1/2)))*k5 + (70*(7 - 21**(1/2)))*k6
                    )/180
                )*self.h
            )
            y[i] = y[i - 1] + self.h*(9*k1 + 64*k3 + 49*k5 + 49*k6 + 9*k7)/180
        for i in range(6, self.n):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 2], y[i - 2])
            k3 = self.f(self.x[i - 3], y[i - 3])
            k4 = self.f(self.x[i - 4], y[i - 4])
            k5 = self.f(self.x[i - 5], y[i - 5])
            k6 = self.f(self.x[i - 6], y[i - 6])
            y[i] = y[i - 1] + self.h*(
                4277/1440*k1 - 2641/480*k2 + 4991/720*k3 - 3649/720*k4 +
                959/480*k5 - 95/288*k6
            )
        self.yf = y
        # print(y[self.n - 1])

    def adamsMoulton3(self):
        self.method_name = 'Adams Moulton 3'
        y = self.y
        for i in range(1, self.n):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 1] + self.h*0.5, y[i - 1] + self.h*0.5*k1)
            k3 = self.f(
                self.x[i - 1] + self.h, y[i - 1] - k1*self.h + 2*k2*self.h
            )
            y[i] = y[i - 1] + self.h*(k1 + 4*k2 + k3)/6
        for i in range(2, self.n):
            k0 = self.f(self.x[i], y[i])
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 2], y[i - 2])
            y[i] = y[i - 1] + self.h*(5/12*k0 + 2/3*k1 - 1/12*k2)
        self.yf = y
        # print(y[self.n - 1])

    def adamsMoulton4(self):
        self.method_name = 'Adams Moulton 4'
        y = self.y
        for i in range(1, self.n):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 1] + self.h*0.5, y[i - 1] + self.h*0.5*k1)
            k3 = self.f(self.x[i - 1] + self.h*0.5, y[i - 1] + self.h*0.5*k2)
            k4 = self.f(self.x[i - 1] + self.h, y[i - 1] + self.h*k3)
            y[i] = y[i - 1] + self.h*(k1 + 2*k2 + 2*k3 + k4)/6
        for i in range(3, self.n):
            k0 = self.f(self.x[i], y[i])
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 2], y[i - 2])
            k3 = self.f(self.x[i - 3], y[i - 3])
            y[i] = y[i - 1] + self.h*(3/8*k0 + 19/24*k1 - 5/24*k2 + 1/24*k3)
        self.yf = y
        # print(y[self.n - 1])

    def adamsMoulton5(self):
        self.method_name = 'Adams Moulton 5'
        y = self.y
        for i in range(1, self.n):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 1] + self.h/4, y[i - 1] + self.h/4*k1)
            k3 = self.f(
                self.x[i - 1] + self.h/4, y[i - 1] + self.h/8*k1 + self.h/8*k2
            )
            k4 = self.f(
                self.x[i - 1] + self.h/2, y[i - 1] - self.h/2*k2 + self.h*k3
            )
            k5 = self.f(
                self.x[i - 1] + self.h*3/4,
                y[i - 1] + self.h*3/16*k1 + self.h*9/16*k4
            )
            k6 = self.f(
                self.x[i - 1] + self.h, y[i - 1] +
                self.h*(-3/7*k1 + 2/7*k2 + 12/7*k3 - 12/7*k4 + 8/7*k5)
            )
            y[i] = y[i - 1] + self.h*(7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6)/90
        for i in range(4, self.n):
            k0 = self.f(self.x[i], y[i])
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 2], y[i - 2])
            k3 = self.f(self.x[i - 3], y[i - 3])
            k4 = self.f(self.x[i - 4], y[i - 4])
            y[i] = y[i - 1] + self.h*(
                251/720*k0 + 323/360*k1 - 11/30*k2 + 53/360*k3 - 19/720*k4
            )
        self.yf = y
        # print(y[self.n - 1])

    def adamsMoulton6(self):
        self.method_name = 'Adams Moulton 6'
        y = self.y
        for i in range(1, self.n):
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 1] + self.h, y[i - 1] + k1*self.h)
            k3 = self.f(
                self.x[i - 1] + self.h/2, y[i - 1] + ((3*k1 + k2)/8)*self.h
            )
            k4 = self.f(
                self.x[i - 1] + self.h*2/3,
                y[i - 1] + ((8*k1 + 2*k2 + 8*k3)/27)*self.h
            )
            k5 = self.f(
                self.x[i - 1] + self.h*(7 - 21**(1/2))/14, y[i - 1] + (
                    (
                        (3*(3*21**(1/2) - 7))*k1 - (8*(7 - 21**(1/2)))*k2 +
                        (48*(7 - 21**(1/2)))*k3 - (3*(21 - 21**(1/2)))*k4
                    )/392
                )*self.h
            )
            k6 = self.f(
                self.x[i - 1] + self.h*(7 + 21**(1/2))/14, y[i - 1] + (
                    (
                        -(5*(231 + 51*21**(1/2)))*k1 - (40*(7 + 21**
                                                            (1/2)))*k2 -
                        (320*21**
                         (1/2))*k3 + (3*(21 + 121*21**
                                         (1/2)))*k4 + (392*(6 + 21**(1/2)))*k5
                    )/1960
                )*self.h
            )
            k7 = self.f(
                self.x[i - 1] + self.h, y[i - 1] + (
                    (
                        (15*(22 + 7*21**(1/2)))*k1 + (120)*k2 +
                        (40*(7*21**(1/2) - 5))*k3 - (63*(3*21**(1/2) - 2))*k4 -
                        (14*(49 + 9*21**(1/2)))*k5 + (70*(7 - 21**(1/2)))*k6
                    )/180
                )*self.h
            )
            y[i] = y[i - 1] + self.h*(9*k1 + 64*k3 + 49*k5 + 49*k6 + 9*k7)/180
        for i in range(5, self.n):
            k0 = self.f(self.x[i], y[i])
            k1 = self.f(self.x[i - 1], y[i - 1])
            k2 = self.f(self.x[i - 2], y[i - 2])
            k3 = self.f(self.x[i - 3], y[i - 3])
            k4 = self.f(self.x[i - 4], y[i - 4])
            k5 = self.f(self.x[i - 5], y[i - 5])
            y[i] = y[i - 1] + self.h*(
                95/288*k0 + 1427/1440*k1 - 133/240*k2 + 241/720*k3 -
                173/1440*k4 + 3/160*k5
            )
        self.yf = y
        # print(y[self.n - 1])

    # fazer o programa plotar agora
    def plot(self):
        print(f'{self.method_name+":":<22} f({self.x[-1]}) = {self.yf[-1]}')
        plt.plot(self.x, self.yf, label=self.method_name)
        plt.legend()
        # plt.show()


if __name__ == "__main__":
    print('Entre na seguinte ordem em uma única linha:')
    print('ex: x0, y0, xf, h, mathExpr')
    s = input()
    s = s.replace(' ', '').split(',')
    entrada = [float(i) for i in s[:4]] + [s[4]]
    print('opções:')
    print('0: Euler')
    print('1: Euler Inverso')
    print('2: Euler Modificado')
    print('3: runge kutta')
    print('4: runge kutta 4 ordem')
    print('5: runge kutta 5 ordem')
    print('6: runge kutta 6 ordem')
    print('7: adams bashforth 2 ordem')
    print('8: adams bashforth 3 ordem')
    print('9: adams bashforth 4 ordem')
    print('10: adams bashforth 5 ordem')
    print('11: adams bashforth 6 ordem')
    print('12: adams moulton 3 ordem')
    print('13: adams moulton 4 ordem')
    print('14: adams moulton 5 ordem')
    print('15: adams moulton 6 ordem')
    print('entre com a lista das funções desejadas: ex: 1, 2, 3')
    # print('q: sair')
    # while opFunc != 'q':
    func = Methods(*entrada)
    print(f'y({func.x0}) = {func.y0}; h = {func.h}; expr = {entrada[4]}')
    lista_metodos = input().replace(' ', '').split(',')
    for opFunc in lista_metodos:
        # opFunc = input("Qual método numérico você quer visualizar?\n")
        if opFunc == '0':
            func.euler()
            func.plot()
        elif opFunc == '1':
            func.eulerInverse()
            func.plot()
        elif opFunc == '2':
            func.eulerModified()
            func.plot()
        elif opFunc == '3':
            func.rungeKutta()
            func.plot()
        elif opFunc == '4':
            func.rungeKutta4()
            func.plot()
        elif opFunc == '5':
            func.rungeKutta5()
            func.plot()
        elif opFunc == '6':
            func.rungeKutta6()
            func.plot()
        elif opFunc == '7':
            func.adamsBash2()
            func.plot()
        elif opFunc == '8':
            func.adamsBash3()
            func.plot()
        elif opFunc == '9':
            func.adamsBash4()
            func.plot()
        elif opFunc == '10':
            func.adamsBash5()
            func.plot()
        elif opFunc == '11':
            func.adamsBash6()
            func.plot()
        elif opFunc == '12':
            func.adamsMoulton3()
            func.plot()
        elif opFunc == '13':
            func.adamsMoulton4()
            func.plot()
        elif opFunc == '14':
            func.adamsMoulton5()
            func.plot()
        elif opFunc == '15':
            func.adamsMoulton6()
            func.plot()
        elif opFunc == 'q':
            print('saindo...')
        else:
            print(f'{opFunc}: Operação inválida.')
    plt.show()
