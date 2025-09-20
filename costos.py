import numpy as np

def mse(T, P):
    return np.mean((T - P)**2)

def mse_derivada(T, P):
    return -2 * (T - P) / T.size

def bce(T, P):
    epsilon = 1e-15
    P = np.clip(P, epsilon, 1 - epsilon)
    return -np.mean(T * np.log(P) + (1-T) * np.log(1-P))

def bce_derivada(T, P):
    epsilon = 1e-15
    P = np.clip(P, epsilon, 1 - epsilon)
    return (-T / P + (1-T)/ (1 - P)) / T.size

mapa_costos = {
    'mse': (mse, mse_derivada),
    'bce': (bce, bce_derivada)
}