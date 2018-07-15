# Métodos Numéricos
Implementação de alguns metódos numéricos em python

## Requisitos
- **matplotlib**: para plotar o gráfico de cada método
- **numpy**: para cálculo numérico rápido 

## Formato de entrada
Assim que iniciar o programa, será dado a forma da entrada
e uma lista com as opções dos métodos
A entrada é dada na forma 
>x0, y0, xf, step_size, mathExpr, [exactExpr]  
>lista dos métodos separados por vírgula
- **x0**: valor de x inicial
- **y0**: valor de y inicial
- **xf**: valor final de x0
- **step_size**: tamando do passo entre xi e xi+1; valor do `h`
- **mathExpr**: função matemática onde dy/dx = mathExpr
- **exactExpr**: solução analítica da equação diferencial (parâmentro opcional)  
A expressão pode ser escrita da mesma forma que python entende as funções matemáticas

### Exemplo de entrada:
>0, 1, 2, 0.01, 1-x+4*y  
>0, 1, 2, 3, 4

----------

## Índice de métodos:
|Entrada|Método|
|--|--|
|0. |Euler|
|1. |Euler Inverso|
|2. |Euler Modificado|
|3. |runge kutta|
|4. |runge kutta 4 ordem|
|5. |runge kutta 5 ordem|
|6. |runge kutta 6 ordem|
|7. |adams bashforth 2 ordem|
|8. |adams bashforth 3 ordem|
|9. |adams bashforth 4 ordem|
|10.| adams bashforth 5 ordem|
|11.| adams bashforth 6 ordem|
|12.| adams moulton 3 ordem|
|13.| adams moulton 4 ordem|
|14.| adams moulton 5 ordem|
|15.| adams moulton 6 ordem|

