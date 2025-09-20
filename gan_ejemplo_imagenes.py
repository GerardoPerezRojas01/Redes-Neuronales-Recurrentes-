r"""Práctica final: generación de dígitos (8x8) con una GAN totalmente densa.

Los píxeles se normalizan usando

.. math::
   \tilde{x} = 2\, (x - x_{\min}) / (x_{\max} - x_{\min}) - 1

para que el generador con activación ``tanh`` produzca valores acordes.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from gan_entrenamiento_adversarial import (
    entrenar_gan,
    graficar_historial,
    generar_muestras,
)
from gan_modelos import ConfiguracionModelo


def cargar_digitos() -> np.ndarray:
    """Carga ``sklearn`` digits y los normaliza a ``[-1, 1]``."""

    dataset = load_digits()
    datos = dataset.data.astype(np.float64)
    minimo = datos.min()
    maximo = datos.max()
    datos = 2 * (datos - minimo) / (maximo - minimo) - 1
    return datos


def mostrar_imagenes(arreglo: np.ndarray, n: int = 16, titulo: str = '') -> None:
    """Muestra ``n`` imágenes 8x8 en una cuadrícula."""

    lado = int(np.sqrt(n))
    figuras = arreglo[: n].reshape(-1, 8, 8)

    fig, axes = plt.subplots(lado, lado, figsize=(6, 6))
    for ax, imagen in zip(axes.flatten(), figuras):
        ax.imshow(imagen, cmap='gray', vmin=-1, vmax=1)
        ax.axis('off')
    if titulo:
        fig.suptitle(titulo)
    plt.show()


def ejecutar_practica(epocas: int = 400) -> None:
    datos = cargar_digitos()

    config_gen = ConfiguracionModelo(
        dimensiones=[16, 128, 128, 64],
        activaciones=['relu', 'relu', 'tanh'],
        lr=2e-3,
    )
    config_disc = ConfiguracionModelo(
        dimensiones=[64, 128, 64, 1],
        activaciones=['relu', 'relu', 'sigmoide'],
        lr=2e-3,
    )

    generador, discriminador, historial = entrenar_gan(
        datos,
        config_gen,
        config_disc,
        epocas=epocas,
        tamano_lote=128,
        dimension_ruido=16,
        semilla=21,
    )

    graficar_historial(historial)

    reales = datos[:16]
    sinteticos = generar_muestras(generador, 16, dimension_ruido=16)

    mostrar_imagenes(reales, titulo='Ejemplos reales')
    mostrar_imagenes(sinteticos, titulo='Ejemplos generados')


if __name__ == '__main__':
    ejecutar_practica()
