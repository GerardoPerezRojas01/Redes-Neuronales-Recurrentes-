import numpy as np
import src.activaciones as act

def codificador(X, parametros, activaciones):
    medio = len(activaciones) // 2
    A = X

    for i, activacion in enumerate(activaciones[:medio]):
        W = parametros[f'W{i+1}']
        b = parametros[f'b{i+1}']
        g, _ = act.mapa_activaciones[activacion]

        Z = np.dot(A, W) + b
        A = g(Z)

    return A

def decodificador(Z, parametros, activaciones):
    medio = len(activaciones) // 2
    A = Z

    for i, activacion in enumerate(activaciones[medio:]):
        W = parametros[f'W{i+medio+1}']
        b = parametros[f'b{i+medio+1}']
        g, _ = act.mapa_activaciones[activacion]

        Z = np.dot(A, W) + b
        A = g(Z)

    return A