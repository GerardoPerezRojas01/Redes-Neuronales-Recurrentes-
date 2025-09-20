r"""Práctica 2D con ``make_moons`` para visualizar el aprendizaje adversarial.

La distribución objetivo corresponde a dos semicírculos desplazados. Puede
interpretarse como el conjunto

.. math::
   \{(x, y) \in \mathbb{R}^2 : (x \pm 0.5)^2 + y^2 = 1 \}

corrompido con ruido gaussiano.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from gan_entrenamiento_adversarial import entrenar_gan, graficar_historial
from gan_entrenamiento_adversarial import generar_muestras as generar_sinteticos
from gan_modelos import ConfiguracionModelo


def datos_reales_2d(n_muestras: int = 2000, ruido: float = 0.1, semilla: int = 42) -> np.ndarray:
    """Genera ``make_moons`` escalado a ``[-1, 1]``."""

    datos, _ = make_moons(n_samples=n_muestras, noise=ruido, random_state=semilla)
    datos = datos.astype(np.float64)
    minimo = datos.min(axis=0, keepdims=True)
    maximo = datos.max(axis=0, keepdims=True)
    return 2 * (datos - minimo) / (maximo - minimo) - 1


def ejecutar_practica(epocas: int = 400) -> None:
    datos = datos_reales_2d()

    config_gen = ConfiguracionModelo(
        dimensiones=[2, 32, 32, 2],
        activaciones=['relu', 'relu', 'tanh'],
        lr=2e-3,
    )
    config_disc = ConfiguracionModelo(
        dimensiones=[2, 64, 64, 1],
        activaciones=['relu', 'relu', 'sigmoide'],
        lr=2e-3,
    )

    generador, discriminador, historial = entrenar_gan(
        datos,
        config_gen,
        config_disc,
        epocas=epocas,
        tamano_lote=128,
        dimension_ruido=2,
        semilla=123,
    )

    graficar_historial(historial)

    muestras = generar_sinteticos(generador, 2000, dimension_ruido=2)

    plt.figure(figsize=(6, 6))
    plt.scatter(datos[:, 0], datos[:, 1], s=10, alpha=0.5, label='Reales')
    plt.scatter(muestras[:, 0], muestras[:, 1], s=10, alpha=0.5, label='Sintéticas')
    plt.title('GAN 2D - Make Moons')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    ejecutar_practica()
