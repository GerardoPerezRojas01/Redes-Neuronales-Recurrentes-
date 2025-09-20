import numpy as np

def gradiente_descendente(parametros, derivadas, lr):
    parametros_actualizados = {}
    for llave in parametros.keys():
        parametros_actualizados[llave] = parametros[llave] - lr * derivadas[f'd{llave}']
    return parametros_actualizados


mapa_optimizadores = {
    'gd': gradiente_descendente
}