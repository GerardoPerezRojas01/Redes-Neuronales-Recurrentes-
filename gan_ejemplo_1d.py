r"""Práctica: GAN unidimensional para aproximar una mezcla gaussiana.

Distribución objetivo:

.. math::
   p_{\text{real}}(x) = 0.5\,\mathcal{N}(-2, 0.2^2) + 0.5\,\mathcal{N}(2, 0.2^2).

El objetivo es que ``G`` aprenda a producir muestras 1D que sigan dicha mezcla
utilizando capas densas pequeñas.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from gan_entrenamiento_adversarial import (
    entrenar_gan,
    graficar_historial,
    generar_muestras,
)
from gan_modelos import ConfiguracionModelo


def datos_reales_1d(n_muestras: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Muestra valores de la mezcla gaussiana propuesta."""

    if rng is None:
        rng = np.random.default_rng()
    componentes = rng.integers(0, 2, size=n_muestras)
    medias = np.where(componentes == 0, -2.0, 2.0)
    muestras = rng.normal(loc=medias, scale=0.2)
    return muestras.reshape(-1, 1)


def ejecutar_practica(epocas: int = 400) -> None:
    """Entrena una GAN simple y visualiza resultados."""

    rng = np.random.default_rng(42)
    datos = datos_reales_1d(2000, rng)

    config_gen = ConfiguracionModelo(
        dimensiones=[1, 16, 16, 1],
        activaciones=['relu', 'relu', 'tanh'],
        lr=1e-3,
    )
    config_disc = ConfiguracionModelo(
        dimensiones=[1, 32, 32, 1],
        activaciones=['relu', 'relu', 'sigmoide'],
        lr=1e-3,
    )

    generador, discriminador, historial = entrenar_gan(
        datos,
        config_gen,
        config_disc,
        epocas=epocas,
        tamano_lote=128,
        dimension_ruido=1,
        semilla=42,
    )

    graficar_historial(historial)

    muestras = generar_muestras(generador, 1000, dimension_ruido=1, rng=rng)

    plt.figure(figsize=(8, 4))
    plt.hist(datos, bins=50, density=True, alpha=0.6, label='Reales')
    plt.hist(muestras, bins=50, density=True, alpha=0.6, label='Sintéticas')
    plt.title('Comparación distribución real vs generada')
    plt.xlabel('x')
    plt.ylabel('densidad')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    ejecutar_practica()
