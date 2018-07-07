# Métodos Numéricos
Implementação de alguns metódos numéricos

## Requisitos
- matplotlib: para plotar o gráfico de cada método
- numpy: para cálculo numérico rápido 

## Formato de entrada
Assim que iniciar o programa, será dado a forma da entrada
e uma lista com as opções dos métodos
A entrada é dada na forma "x0 y0 xf nSteps expressão"
- x0: valor de x inicial
- y0: valor de y inicial
- xf: valor final de x0
- nSteps: quantidade de divisoes do intervalo
- expressão: função matemática onde dy/dx = expressão
A expressão pode ser escrita da mesma forma que python entende as funções matemáticas

Exemplo de entrada:
0 1 1 100 exp(x)+y**2

## Índice de métodos:
0 : Euler
1 : Euler Inverso
2 : Euler Modificado
3 : runge kutta
4 : runge kutta 4 ordem
5 : runge kutta 5 ordem
6 : runge kutta 6 ordem
7 : adams bashforth 2 ordem
8 : adams bashforth 3 ordem
9 : adams bashforth 4 ordem
10: adams bashforth 5 ordem
11: adams bashforth 6 ordem
12: adams moulton 3 ordem
13: adams moulton 4 ordem
14: adams moulton 5 ordem
15: adams moulton 6 ordem
