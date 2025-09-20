r"""Guion paso a paso: generador → discriminador → entrenamiento adversarial.

Este script enlaza los módulos especializados en tres etapas:

1. **Generador**: se obtiene a partir de un autoencoder, minimizando
   :math:`\mathcal{L}_{\text{rec}}`.
2. **Discriminador**: se entrena con ``BCE`` sobre datos reales y sintéticos
   fijos.
3. **Unión adversarial**: se optimizan simultáneamente ``G`` y ``D`` usando las
   pérdidas descritas en ``gan_entrenamiento_adversarial``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from gan_discriminador import entrenar_discriminador
from gan_entrenamiento_adversarial import HistorialAdversarial, entrenar_gan
from gan_generador_autoencoder import inicializar_generador_con_autoencoder
from gan_modelos import ConfiguracionModelo, ModeloDenso, muestrear_ruido


@dataclass
class PasoGenerador:
    generador: ModeloDenso
    historial_reconstruccion: List[float]


@dataclass
class PasoDiscriminador:
    discriminador: ModeloDenso
    historial_discriminador: List[float]


@dataclass
class PasoAdversarial:
    generador: ModeloDenso
    discriminador: ModeloDenso
    historial: HistorialAdversarial


def etapa_generador(
    datos: np.ndarray,
    config_autoencoder: ConfiguracionModelo,
    epocas: int = 200,
    tamano_lote: int = 128,
    semilla: int | None = None,
) -> PasoGenerador:
    """Entrena un autoencoder y devuelve el generador inicial."""

    generador, historial = inicializar_generador_con_autoencoder(
        datos,
        config_autoencoder,
        epocas=epocas,
        tamano_lote=tamano_lote,
        semilla=semilla,
    )
    return PasoGenerador(generador, historial)


def etapa_discriminador(
    datos_reales: np.ndarray,
    generador: ModeloDenso,
    config_discriminador: ConfiguracionModelo,
    epocas: int = 100,
    tamano_lote: int = 128,
    dimension_ruido: int = 2,
    semilla: int | None = None,
) -> PasoDiscriminador:
    """Congela ``G`` y entrena ``D`` con muestras fijas."""

    rng = np.random.default_rng(semilla)
    ruido = muestrear_ruido(datos_reales.shape[0], dimension_ruido, rng=rng)
    datos_sinteticos, _ = generador.forward(ruido)

    discriminador, historial = entrenar_discriminador(
        datos_reales,
        datos_sinteticos,
        config_discriminador,
        epocas=epocas,
        tamano_lote=tamano_lote,
        semilla=semilla,
    )
    return PasoDiscriminador(discriminador, historial)


def etapa_adversarial(
    datos_reales: np.ndarray,
    config_generador: ConfiguracionModelo,
    config_discriminador: ConfiguracionModelo,
    epocas: int = 200,
    tamano_lote: int = 128,
    dimension_ruido: int = 2,
    semilla: int | None = None,
) -> PasoAdversarial:
    """Realiza el ciclo de entrenamiento adversarial completo."""

    generador, discriminador, historial = entrenar_gan(
        datos_reales,
        config_generador,
        config_discriminador,
        epocas=epocas,
        tamano_lote=tamano_lote,
        dimension_ruido=dimension_ruido,
        semilla=semilla,
    )
    return PasoAdversarial(generador, discriminador, historial)
