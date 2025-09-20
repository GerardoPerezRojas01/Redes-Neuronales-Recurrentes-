r"""Utilidades fundamentales para construir generadores y discriminadores densos.

El objetivo de este módulo es concentrar las piezas matemáticas mínimas
necesarias para reutilizarlas en las distintas prácticas de la ruta de 28 horas
sobre GAN. En particular se documentan las funciones de costo y los gradientes
que se emplearán de forma repetida:

* **Pérdida de reconstrucción** (autoencoder):
  :math:`\mathcal{L}_{\text{rec}}(x, \hat x) = \lVert x - \hat x \rVert_2^2`.
* **Pérdida adversarial del discriminador**:
  :math:`\mathcal{L}_D = -\mathbb{E}_{x\sim p_{\text{real}}}\log D(x)
  - \mathbb{E}_{z\sim p_z}\log (1 - D(G(z)))`.
* **Pérdida adversarial del generador** (versión no saturada):
  :math:`\mathcal{L}_G = -\mathbb{E}_{z\sim p_z} \log D(G(z))`.

Cada función y clase expone los tensores que intervienen en las fórmulas para
facilitar la discusión durante la clase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

import costos
import optimizadores
import red_neuronal


@dataclass
class ConfiguracionModelo:
    r"""Define la arquitectura de una red densa y sus hiperparámetros.

    Parameters
    ----------
    dimensiones:
        Lista con el número de neuronas por capa (incluye entrada y salida).
    activaciones:
        Nombre de la activación por capa oculta/salida. Debe haber
        ``len(dimensiones) - 1`` activaciones.
    lr:
        Tasa de aprendizaje :math:`\eta` para el optimizador seleccionado.
    optimizador:
        Clave del optimizador presente en ``optimizadores.mapa_optimizadores``.
    """

    dimensiones: List[int]
    activaciones: List[str]
    lr: float
    optimizador: str = 'gd'

    def __post_init__(self) -> None:
        if len(self.dimensiones) - 1 != len(self.activaciones):
            raise ValueError(
                'El número de activaciones debe coincidir con el de capas '
                'densas (salvo la de entrada).'
            )


class ModeloDenso:
    r"""`Wrapper` ligero sobre ``red_neuronal`` para redes densas.

    La propagación hacia delante calcula :math:`a^{(l)} = g(z^{(l)})` con
    :math:`z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}`. El método ``retropropagar``
    devuelve gradientes compatibles con el algoritmo de retropropagación
    tradicional.
    """

    def __init__(self, config: ConfiguracionModelo):
        self.config = config
        self.parametros = red_neuronal.inicializar_RNA(config.dimensiones)
        self._optimizador = optimizadores.mapa_optimizadores[config.optimizador]

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Propagación hacia delante.

        Parameters
        ----------
        X:
            Matriz ``(n_muestras, n_features)`` con las activaciones de entrada.

        Returns
        -------
        tuple
            Par ``(A_L, cache)`` con la salida y los valores intermedios que
            requiere la retropropagación.
        """

        return red_neuronal.propagacion_adelante(
            X, self.parametros, self.config.activaciones
        )

    def retropropagar(
        self,
        salida: np.ndarray,
        objetivo: np.ndarray,
        cache: Dict[str, np.ndarray],
        derivada_costo=None,
        gradiente_salida: np.ndarray | None = None,
        retornar_gradiente_entrada: bool = False,
    ):
        r"""Ejecuta retropropagación sobre la red.

        Parameters
        ----------
        salida:
            Vector ``P`` de predicciones de la capa de salida.
        objetivo:
            Vector ``T`` con las etiquetas/targets.
        cache:
            Valores intermedios ``a`` y ``da/dz`` guardados en ``forward``.
        derivada_costo:
            Función :math:`\partial \mathcal{L} / \partial P`.
        gradiente_salida:
            Gradiente externo ya evaluado respecto a ``P``.
        retornar_gradiente_entrada:
            Si es ``True`` también se devuelve :math:`\partial \mathcal{L} /
            \partial X`.
        """

        return red_neuronal.retropropagacion(
            salida,
            objetivo,
            self.parametros,
            cache,
            derivada_costo=derivada_costo,
            gradiente_salida=gradiente_salida,
            retornar_gradiente_entrada=retornar_gradiente_entrada,
        )

    def actualizar(self, derivadas: Dict[str, np.ndarray]) -> None:
        r"""Aplica una actualización de parámetros.

        Se ejecuta ``\theta \leftarrow \theta - \eta \, \nabla_\theta`` a
        través del optimizador seleccionado en la configuración.
        """

        self.parametros = self._optimizador(
            self.parametros, derivadas, self.config.lr
        )


def muestrear_ruido(
    n_muestras: int,
    dimension: int,
    escala: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Genera vectores de ruido gaussiano :math:`z \sim \mathcal{N}(0, \sigma^2 I)`."""

    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(loc=0.0, scale=escala, size=(n_muestras, dimension))


def mezclar_batches(*arreglos: Iterable[np.ndarray]) -> Tuple[np.ndarray, ...]:
    """Baraja de forma consistente varios arreglos de igual longitud."""

    if not arreglos:
        raise ValueError('Se requiere al menos un arreglo para mezclar.')

    n = arreglos[0].shape[0]
    indices = np.random.permutation(n)
    return tuple(arreglo[indices] for arreglo in arreglos)


def evaluar_bce(etiquetas: np.ndarray, probabilidades: np.ndarray) -> float:
    """Calcula la entropía cruzada binaria media."""

    costo_bce, _ = costos.mapa_costos['bce']
    return float(costo_bce(etiquetas, probabilidades))


def gradiente_bce(etiquetas: np.ndarray, probabilidades: np.ndarray) -> np.ndarray:
    """Gradiente de la entropía cruzada binaria."""

    _, derivada_bce = costos.mapa_costos['bce']
    return derivada_bce(etiquetas, probabilidades)
