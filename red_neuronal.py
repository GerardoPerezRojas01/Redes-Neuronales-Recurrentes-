import numpy as np
import src.activaciones as act
import src.costos as c
import src.optimizadores as op


def inicializar_RNA(capas):

    parametros = {}

    for i in range(len(capas)-1):
        neuronas_capa_actual = capas[i]
        neuronas_capa_siguiente = capas[i+1]

        parametros[f'W{i+1}'] = np.random.randn(neuronas_capa_actual, neuronas_capa_siguiente) / np.sqrt(neuronas_capa_actual)
        parametros[f'b{i+1}'] = np.zeros((1, neuronas_capa_siguiente))

    return parametros


def propagacion_adelante(X, parametros, activaciones):
    valor_capas = {'a0': X}
    A = X

    for i, activacion in enumerate(activaciones):
        W = parametros[f'W{i+1}']
        b = parametros[f'b{i+1}']
        g, dg_dz = act.mapa_activaciones[activacion]

        Z = np.dot(A, W) + b
        A = g(Z)

        valor_capas[f'a{i+1}'] = A
        valor_capas[f'da/dz{i+1}'] = dg_dz(Z)

    return A, valor_capas

def retropropagacion(P, T, parametros, valor_capas, derivada_costo):
    L = len(parametros) // 2
    derivadas = {}

    delta = derivada_costo(T, P) * valor_capas[f'da/dz{L}']
    derivadas[f'dW{L}'] = np.matmul(valor_capas[f'a{L-1}'].transpose(), delta)
    derivadas[f'db{L}'] = np.sum(delta, axis=0, keepdims=True)

    for l in range(L-1, 0, -1):
        W = parametros[f'W{l+1}']
        delta = np.matmul(delta, W.transpose()) * valor_capas[f'da/dz{l}']
        derivadas[f'dW{l}'] = np.matmul(valor_capas[f'a{l-1}'].transpose(), delta)
        derivadas[f'db{l}'] = np.sum(delta, axis=0, keepdims=True)

    return derivadas

def generar_batches(X, T, tamano_lote):
    n_muestras = X.shape[0]
    indices = np.random.permutation(n_muestras)

    for i in range(0, n_muestras, tamano_lote):
        batch_indices = indices[i:i+tamano_lote]
        yield X[batch_indices], T[batch_indices]


def actualizacion_parametros(X, T, parametros, config):
    activaciones = config['activaciones']
    costo_tipo = config['costo']
    optimizador_tipo = config['optimizador']
    tamano_lote = config['tamano_lote']
    lr = config['lr']

    # Obtener funciones de costo
    costo_fn, costo_derivada = c.mapa_costos[costo_tipo]
    optimizador_fn = op.mapa_optimizadores[optimizador_tipo]

    costo_total = 0
    n_lotes = 0

    for X_batch, T_batch in generar_batches(X, T, tamano_lote):
        pred, val_capas = propagacion_adelante(X_batch, parametros, activaciones)
        derivadas = retropropagacion(pred, T_batch, parametros, val_capas, costo_derivada)
        parametros = optimizador_fn(parametros, derivadas, lr)

        costo_total += costo_fn(T_batch, pred)
        n_lotes += 1

    costo_promedio = costo_total / n_lotes

    return parametros, costo_promedio

def entrenar_red(X, T, config):
    input = config['input']
    capas_ocultas = config['capas_ocultas']
    epocas = config['epocas']
    costo_tipo = config['costo']

    arquitectura = [input] + capas_ocultas
    parametros = inicializar_RNA(arquitectura)

    historial = np.zeros((2, epocas))

    # Separa el conjunto en datos entrenamiento y validacion
    n = int(len(X) * 0.8)
    idx = np.random.permutation(len(X))
    X, T = X[idx], T[idx]
    X_ent, T_ent = X[:n], T[:n]
    X_val, T_val = X[n:], T[n:]

    fn_costo, _ = c.mapa_costos[costo_tipo]

    for epoca in range(epocas):
        parametros, costo_epoca = actualizacion_parametros(X_ent, T_ent, parametros, config)
        historial[0, epoca] = costo_epoca
        pred_val = predecir(X_val, parametros, config['activaciones'])
        costo_val = fn_costo(T_val, pred_val)
        historial[1, epoca] = costo_val

        print(f'Epoca {epoca:4d}, costo entrenamiento: {costo_epoca:.4f}, costo validacion: {costo_val:.4f}')

    return parametros, historial

def predecir(X, parametros, activaciones):
    A = X

    for i, activacion in enumerate(activaciones):
        W = parametros[f'W{i+1}']
        b = parametros[f'b{i+1}']
        g, _ = act.mapa_activaciones[activacion]

        Z = np.dot(A, W) + b
        A = g(Z)

    return A