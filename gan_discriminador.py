r"""Entrenamiento supervisado del discriminador antes del ciclo adversarial.

Se utiliza la pérdida de entropía cruzada binaria

.. math::
   \mathcal{L}_D(\theta_D) = -\frac{1}{N} \sum_{i=1}^N 
   \left[y_i \log D(x_i) + (1-y_i) \log (1 - D(x_i))\right]

para separar ejemplos reales (:math:`y=1`) de ejemplos sintéticos (:math:`y=0`).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

import costos
from gan_modelos import ConfiguracionModelo, ModeloDenso, mezclar_batches


def preparar_conjunto_discriminador(
    reales: np.ndarray,
    falsos: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatena datos reales/falsos y asigna etiquetas binaras."""

    y_reales = np.ones((reales.shape[0], 1), dtype=np.float64)
    y_falsos = np.zeros((falsos.shape[0], 1), dtype=np.float64)

    X = np.concatenate([reales, falsos], axis=0)
    y = np.concatenate([y_reales, y_falsos], axis=0)
    return X, y


def entrenar_discriminador(
    datos_reales: np.ndarray,
    datos_falsos: np.ndarray,
    config: ConfiguracionModelo,
    epocas: int = 100,
    tamano_lote: int = 128,
    semilla: int | None = None,
) -> Tuple[ModeloDenso, List[float]]:
    """Ajusta un discriminador ``D`` previo al entrenamiento adversarial."""

    if semilla is not None:
        np.random.seed(semilla)

    discriminador = ModeloDenso(config)
    _, derivada_bce = costos.mapa_costos['bce']
    historial: List[float] = []

    X, y = preparar_conjunto_discriminador(datos_reales, datos_falsos)
    n = X.shape[0]

    for epoca in range(epocas):
        X_bar, y_bar = mezclar_batches(X, y)

        for inicio in range(0, n, tamano_lote):
            fin = inicio + tamano_lote
            batch_X = X_bar[inicio:fin]
            batch_y = y_bar[inicio:fin]

            pred, cache = discriminador.forward(batch_X)
            derivadas = discriminador.retropropagar(
                pred, batch_y, cache, derivada_bce
            )
            discriminador.actualizar(derivadas)

        pred_epoca, _ = discriminador.forward(X)
        perdidas = costos.bce(y, pred_epoca)
        historial.append(float(perdidas))

    return discriminador, historial
