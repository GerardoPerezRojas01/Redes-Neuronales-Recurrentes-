r"""Práctica 3D: aproximar un ``swiss roll`` mediante GAN.

El conjunto real se define por las ecuaciones paramétricas

.. math::
   x(t) = t \cos t,\quad y(t) = h,\quad z(t) = t \sin t

con ``t`` uniforme y ``h`` ruido vertical. ``make_swiss_roll`` de ``sklearn`` se
usa para sintetizar el dataset real.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.datasets import make_swiss_roll

from gan_entrenamiento_adversarial import entrenar_gan, graficar_historial
from gan_entrenamiento_adversarial import generar_muestras as generar_sinteticos
from gan_modelos import ConfiguracionModelo


def datos_reales_3d(n_muestras: int = 3000, ruido: float = 0.2, semilla: int = 0) -> np.ndarray:
    """Swiss roll normalizado a ``[-1, 1]`` en cada eje."""

    datos, _ = make_swiss_roll(n_samples=n_muestras, noise=ruido, random_state=semilla)
    datos = datos.astype(np.float64)
    minimo = datos.min(axis=0, keepdims=True)
    maximo = datos.max(axis=0, keepdims=True)
    return 2 * (datos - minimo) / (maximo - minimo) - 1


def ejecutar_practica(epocas: int = 400) -> None:
    datos = datos_reales_3d()

    config_gen = ConfiguracionModelo(
        dimensiones=[3, 64, 64, 3],
        activaciones=['relu', 'relu', 'tanh'],
        lr=1e-3,
    )
    config_disc = ConfiguracionModelo(
        dimensiones=[3, 128, 128, 1],
        activaciones=['relu', 'relu', 'sigmoide'],
        lr=1e-3,
    )

    generador, discriminador, historial = entrenar_gan(
        datos,
        config_gen,
        config_disc,
        epocas=epocas,
        tamano_lote=256,
        dimension_ruido=3,
        semilla=7,
    )

    graficar_historial(historial)

    muestras = generar_sinteticos(generador, 3000, dimension_ruido=3)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.scatter(datos[:, 0], datos[:, 1], datos[:, 2], s=4, alpha=0.5)
    ax1.set_title('Datos reales')

    ax2.scatter(muestras[:, 0], muestras[:, 1], muestras[:, 2], s=4, alpha=0.5)
    ax2.set_title('Datos sintéticos')

    plt.show()


if __name__ == '__main__':
    ejecutar_practica()
