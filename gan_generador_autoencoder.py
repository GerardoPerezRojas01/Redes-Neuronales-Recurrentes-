r"""Pre-entrenamiento del generador a partir de un autoencoder denso.

La idea pedagógica es reutilizar el decodificador de un autoencoder ya
entrenado como punto de partida del generador :math:`G`. Al minimizar la
pérdida de reconstrucción

.. math::
   \mathcal{L}_{\text{rec}} = \frac{1}{N} \sum_{i=1}^N \lVert x_i - \hat x_i \rVert_2^2,

aprendemos un mapeo suave de un espacio latente a la distribución de datos, lo
cual proporciona pesos iniciales informados para el generador de la GAN.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

import costos
from gan_modelos import ConfiguracionModelo, ModeloDenso, mezclar_batches


def entrenar_autoencoder(
    datos: np.ndarray,
    config: ConfiguracionModelo,
    epocas: int = 200,
    tamano_lote: int = 128,
    semilla: int | None = None,
) -> Tuple[ModeloDenso, List[float]]:
    r"""Ajusta un autoencoder simétrico sobre ``datos``.

    Parameters
    ----------
    datos:
        Matriz ``(N, d)`` que servirá simultáneamente como entrada y objetivo.
    config:
        Configuración de la red (debe ser simétrica para separar encoder y
        decoder).
    epocas, tamano_lote, semilla:
        Hiperparámetros del descenso de gradiente mini-batch.

    Returns
    -------
    autoencoder, historial
        Modelo entrenado y lista con ``\mathcal{L}_{\text{rec}}`` por época.
    """

    if semilla is not None:
        np.random.seed(semilla)

    autoencoder = ModeloDenso(config)
    costo_mse, derivada_mse = costos.mapa_costos['mse']

    historial: List[float] = []
    n = datos.shape[0]

    for epoca in range(epocas):
        (datos_barajados,) = mezclar_batches(datos)

        for inicio in range(0, n, tamano_lote):
            fin = inicio + tamano_lote
            batch = datos_barajados[inicio:fin]

            salida, cache = autoencoder.forward(batch)
            derivadas = autoencoder.retropropagar(
                salida, batch, cache, derivada_mse
            )
            autoencoder.actualizar(derivadas)

        reconstruccion, _ = autoencoder.forward(datos)
        perdida = float(costo_mse(datos, reconstruccion))
        historial.append(perdida)

    return autoencoder, historial


def extraer_generador_desde_autoencoder(
    autoencoder: ModeloDenso,
) -> Tuple[ModeloDenso, Dict[str, np.ndarray]]:
    """Copia el decodificador del autoencoder como generador ``G``.

    Returns
    -------
    generador, parametros
        Instancia lista para usarse junto con sus pesos.
    """

    config_auto = autoencoder.config
    mitad = len(config_auto.activaciones) // 2

    dimensiones_generador = config_auto.dimensiones[mitad:]
    activaciones_generador = config_auto.activaciones[mitad:]

    config_gen = ConfiguracionModelo(
        dimensiones=dimensiones_generador,
        activaciones=activaciones_generador,
        lr=config_auto.lr,
        optimizador=config_auto.optimizador,
    )
    generador = ModeloDenso(config_gen)

    for i in range(1, len(dimensiones_generador)):
        generador.parametros[f'W{i}'] = autoencoder.parametros[f'W{mitad + i}'].copy()
        generador.parametros[f'b{i}'] = autoencoder.parametros[f'b{mitad + i}'].copy()

    return generador, generador.parametros


def inicializar_generador_con_autoencoder(
    datos: np.ndarray,
    config_autoencoder: ConfiguracionModelo,
    epocas: int = 200,
    tamano_lote: int = 128,
    semilla: int | None = None,
) -> Tuple[ModeloDenso, List[float]]:
    """Conveniencia: entrena el autoencoder y devuelve sólo el generador.

    La salida es el par ``(generador, historial_rec)`` con el que se puede
    iniciar el bloque de entrenamiento adversarial.
    """

    autoencoder, historial = entrenar_autoencoder(
        datos, config_autoencoder, epocas=epocas, tamano_lote=tamano_lote, semilla=semilla
    )
    generador, _ = extraer_generador_desde_autoencoder(autoencoder)
    return generador, historial
