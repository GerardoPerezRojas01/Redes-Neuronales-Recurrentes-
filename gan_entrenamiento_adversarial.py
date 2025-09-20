"""Rutinas de entrenamiento adversarial para unir ``G`` y ``D``.

El lazo básico sigue el esquema:

1. Actualizar :math:`\theta_D` maximizando ``log D(x) + log(1 - D(G(z)))``.
2. Actualizar :math:`\theta_G` minimizando ``-log D(G(z))``.

Cada paso se implementa como funciones independientes para facilitar la
experimentación en clase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import costos
from gan_modelos import (
    ConfiguracionModelo,
    ModeloDenso,
    evaluar_bce,
    gradiente_bce,
    mezclar_batches,
    muestrear_ruido,
)


@dataclass
class HistorialAdversarial:
    """Almacena pérdidas y métricas de una sesión de entrenamiento."""

    costo_generador: List[float]
    costo_discriminador: List[float]
    prob_real: List[float]
    prob_fake: List[float]


def paso_discriminador(
    discriminador: ModeloDenso,
    generador: ModeloDenso,
    reales: np.ndarray,
    dimension_ruido: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """Ejecuta un paso de gradiente sobre ``D``."""

    _, derivada_bce = costos.mapa_costos['bce']

    ruido = muestrear_ruido(reales.shape[0], dimension_ruido, rng=rng)
    falsos, _ = generador.forward(ruido)

    entradas = np.concatenate([reales, falsos], axis=0)
    etiquetas = np.concatenate(
        [np.ones((reales.shape[0], 1)), np.zeros((falsos.shape[0], 1))], axis=0
    )

    entradas, etiquetas = mezclar_batches(entradas, etiquetas)

    pred, cache = discriminador.forward(entradas)
    derivadas = discriminador.retropropagar(pred, etiquetas, cache, derivada_bce)
    discriminador.actualizar(derivadas)

    perdida = evaluar_bce(etiquetas, pred)
    prob_real = float(discriminador.forward(reales)[0].mean())
    prob_fake = float(discriminador.forward(falsos)[0].mean())
    return perdida, prob_real, prob_fake


def paso_generador(
    discriminador: ModeloDenso,
    generador: ModeloDenso,
    tamano_lote: int,
    dimension_ruido: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Actualiza ``G`` buscando ``D(G(z)) -> 1``."""

    _, derivada_bce = costos.mapa_costos['bce']

    ruido = muestrear_ruido(tamano_lote, dimension_ruido, rng=rng)
    falsos, cache_g = generador.forward(ruido)

    etiquetas_objetivo = np.ones((tamano_lote, 1))
    pred_fake, cache_d = discriminador.forward(falsos)

    gradiente_salida = gradiente_bce(etiquetas_objetivo, pred_fake)
    _, gradiente_entrada = discriminador.retropropagar(
        pred_fake,
        etiquetas_objetivo,
        cache_d,
        derivada_bce,
        gradiente_salida=gradiente_salida,
        retornar_gradiente_entrada=True,
    )

    derivadas_g = generador.retropropagar(
        falsos,
        np.zeros_like(falsos),
        cache_g,
        gradiente_salida=gradiente_entrada,
    )
    generador.actualizar(derivadas_g)

    perdida = evaluar_bce(etiquetas_objetivo, pred_fake)
    return perdida, float(pred_fake.mean())


def entrenar_gan(
    datos_reales: np.ndarray,
    config_generador: ConfiguracionModelo,
    config_discriminador: ConfiguracionModelo,
    epocas: int = 200,
    tamano_lote: int = 128,
    dimension_ruido: int = 2,
    semilla: int | None = None,
) -> Tuple[ModeloDenso, ModeloDenso, HistorialAdversarial]:
    """Entrenamiento adversarial clásico con NumPy puro."""

    if semilla is not None:
        np.random.seed(semilla)
    rng = np.random.default_rng(semilla)

    generador = ModeloDenso(config_generador)
    discriminador = ModeloDenso(config_discriminador)

    historial = HistorialAdversarial([], [], [], [])
    n = datos_reales.shape[0]

    for epoca in range(epocas):
        indices = np.random.permutation(n)
        datos_reales = datos_reales[indices]

        perdidas_g: List[float] = []
        perdidas_d: List[float] = []
        prob_reales: List[float] = []
        prob_falsos: List[float] = []

        for inicio in range(0, n, tamano_lote):
            fin = inicio + tamano_lote
            reales_batch = datos_reales[inicio:fin]

            if reales_batch.shape[0] == 0:
                continue

            perdida_d, prob_real, _ = paso_discriminador(
                discriminador, generador, reales_batch, dimension_ruido, rng
            )
            perdidas_d.append(perdida_d)
            prob_reales.append(prob_real)

            perdida_g, prob_fake = paso_generador(
                discriminador,
                generador,
                reales_batch.shape[0],
                dimension_ruido,
                rng,
            )
            perdidas_g.append(perdida_g)
            prob_falsos.append(prob_fake)

        historial.costo_generador.append(float(np.mean(perdidas_g)))
        historial.costo_discriminador.append(float(np.mean(perdidas_d)))
        historial.prob_real.append(float(np.mean(prob_reales)))
        historial.prob_fake.append(float(np.mean(prob_falsos)))

    return generador, discriminador, historial


def graficar_historial(historial: HistorialAdversarial) -> None:
    """Genera figuras para analizar el entrenamiento."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(historial.costo_generador, label='Generador')
    axes[0].plot(historial.costo_discriminador, label='Discriminador')
    axes[0].set_title('Pérdidas adversariales')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('BCE')
    axes[0].legend()

    axes[1].plot(historial.prob_real, label='D(x_real)')
    axes[1].plot(historial.prob_fake, label='D(G(z))')
    axes[1].set_title('Confianza del discriminador')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Probabilidad')
    axes[1].legend()

    fig.tight_layout()
    plt.show()


def generar_muestras(
    generador: ModeloDenso,
    n_muestras: int,
    dimension_ruido: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Conveniencia para sintetizar datos con ``G``."""

    if rng is None:
        rng = np.random.default_rng()
    ruido = muestrear_ruido(n_muestras, dimension_ruido, rng=rng)
    muestras, _ = generador.forward(ruido)
    return muestras
