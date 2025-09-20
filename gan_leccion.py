"""Lecciones prácticas de GAN usando únicamente NumPy y la librería local.

Este módulo está pensado para utilizarse dentro de PyCharm durante un curso de
28 horas sobre redes generativas adversarias (GAN). El contenido se divide en
los tres temas clave del programa:

1. **Generador y discriminador**: se definen como redes densas utilizando las
   funciones de `red_neuronal.py`.
2. **Entrenamiento adversarial**: se implementa el lazo de entrenamiento donde
   ambos modelos compiten.
3. **Generación de datos sintéticos**: se muestran rutinas para muestrear datos
   artificiales y visualizar resultados.

Las funciones están ampliamente comentadas para que el profesorado pueda
explicar cada paso directamente desde PyCharm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

import costos
import optimizadores
import red_neuronal


@dataclass
class ConfiguracionModelo:
    """Agrupa la arquitectura de una red densa y sus hiperparámetros."""

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
    """Pequeño *wrapper* sobre las utilidades de ``red_neuronal``.

    Permite inicializar, propagar hacia delante y actualizar una red densa
    empleando los optimizadores definidos en el repositorio.
    """

    def __init__(self, config: ConfiguracionModelo):
        self.config = config
        self.parametros = red_neuronal.inicializar_RNA(config.dimensiones)
        self._optimizador = optimizadores.mapa_optimizadores[config.optimizador]

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Propagación hacia delante devolviendo activaciones intermedias."""

        return red_neuronal.propagacion_adelante(
            X, self.parametros, self.config.activaciones
        )

    def actualizar(self, derivadas: Dict[str, np.ndarray]) -> None:
        """Actualiza los parámetros utilizando el optimizador configurado."""

        self.parametros = self._optimizador(
            self.parametros, derivadas, self.config.lr
        )


def preparar_datos_reales(
    n_muestras: int = 2000,
    ruido: float = 0.1,
    semilla: int | None = None,
) -> np.ndarray:
    """Genera un conjunto de datos 2D no lineal (``make_moons``).

    Los datos se reescalan al rango ``[-1, 1]`` para favorecer el entrenamiento
    con un generador que utilice activación ``tanh`` en la última capa.
    """

    datos, _ = make_moons(n_samples=n_muestras, noise=ruido, random_state=semilla)
    datos = datos.astype(np.float64)

    minimo = datos.min(axis=0, keepdims=True)
    maximo = datos.max(axis=0, keepdims=True)
    datos = 2 * (datos - minimo) / (maximo - minimo) - 1
    return datos


def muestrear_ruido(
    n_muestras: int,
    dimension: int,
    escala: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Extrae ruido gaussiano estándar para el generador."""

    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(loc=0.0, scale=escala, size=(n_muestras, dimension))


def entrenar_gan(
    datos_reales: np.ndarray,
    config_generador: ConfiguracionModelo,
    config_discriminador: ConfiguracionModelo,
    epocas: int = 200,
    tamano_lote: int = 128,
    dimension_ruido: int = 2,
    semilla: int | None = None,
):
    """Entrena una GAN mínima utilizando solamente NumPy.

    Parameters
    ----------
    datos_reales:
        Matriz ``(n_muestras, n_features)`` con los ejemplos reales.
    config_generador, config_discriminador:
        Configuraciones de ambas redes.
    epocas:
        Número de iteraciones completas sobre los datos reales.
    tamano_lote:
        Cantidad de muestras reales empleadas en cada actualización.
    dimension_ruido:
        Tamaño del vector de ruido de entrada al generador.
    semilla:
        Semilla opcional para reproducibilidad.
    """

    if semilla is not None:
        np.random.seed(semilla)

    generador = ModeloDenso(config_generador)
    discriminador = ModeloDenso(config_discriminador)

    costo_bce, derivada_bce = costos.mapa_costos['bce']

    historial = {
        'costo_generador': [],
        'costo_discriminador': [],
        'prob_real': [],
        'prob_fake': [],
    }

    n_muestras = datos_reales.shape[0]

    for epoca in range(epocas):
        indices = np.random.permutation(n_muestras)

        perdidas_generador: List[float] = []
        perdidas_discriminador: List[float] = []
        confianza_real: List[float] = []
        confianza_fake: List[float] = []

        for inicio in range(0, n_muestras, tamano_lote):
            fin = min(inicio + tamano_lote, n_muestras)
            batch_indices = indices[inicio:fin]
            lote_real = datos_reales[batch_indices]
            tamano_actual = lote_real.shape[0]

            ruido = muestrear_ruido(tamano_actual, dimension_ruido)
            sinteticos, cache_generador = generador.forward(ruido)

            entradas_discriminador = np.vstack((lote_real, sinteticos))
            etiquetas = np.vstack(
                (
                    np.ones((tamano_actual, 1), dtype=np.float64),
                    np.zeros((tamano_actual, 1), dtype=np.float64),
                )
            )

            pred_disc, cache_disc = discriminador.forward(entradas_discriminador)
            derivadas_disc = red_neuronal.retropropagacion(
                pred_disc,
                etiquetas,
                discriminador.parametros,
                cache_disc,
                derivada_bce,
            )
            discriminador.actualizar(derivadas_disc)
            perdidas_discriminador.append(costo_bce(etiquetas, pred_disc))

            pred_fake, cache_disc_fake = discriminador.forward(sinteticos)
            _, gradiente_entrada = red_neuronal.retropropagacion(
                pred_fake,
                np.ones((tamano_actual, 1), dtype=np.float64),
                discriminador.parametros,
                cache_disc_fake,
                derivada_bce,
                retornar_gradiente_entrada=True,
            )

            derivadas_gen = red_neuronal.retropropagacion(
                sinteticos,
                None,
                generador.parametros,
                cache_generador,
                gradiente_salida=gradiente_entrada,
            )
            generador.actualizar(derivadas_gen)
            perdidas_generador.append(
                costo_bce(np.ones((tamano_actual, 1), dtype=np.float64), pred_fake)
            )

            confianza_real.append(float(pred_disc[:tamano_actual].mean()))
            confianza_fake.append(float(pred_fake.mean()))

        historial['costo_generador'].append(float(np.mean(perdidas_generador)))
        historial['costo_discriminador'].append(float(np.mean(perdidas_discriminador)))
        historial['prob_real'].append(float(np.mean(confianza_real)))
        historial['prob_fake'].append(float(np.mean(confianza_fake)))

        print(
            f"Epoca {epoca + 1:03d} | Costo D: {historial['costo_discriminador'][-1]:.4f} "
            f"| Costo G: {historial['costo_generador'][-1]:.4f} "
            f"| D(x): {historial['prob_real'][-1]:.3f} "
            f"| D(G(z)): {historial['prob_fake'][-1]:.3f}"
        )

    return generador, discriminador, historial


def generar_datos_sinteticos(
    modelo_generador: ModeloDenso, cantidad: int, dimension_ruido: int
) -> np.ndarray:
    """Muestrea nuevos ejemplos a partir del generador entrenado."""

    ruido = muestrear_ruido(cantidad, dimension_ruido)
    muestras, _ = modelo_generador.forward(ruido)
    return muestras


def graficar_historial(historial: Dict[str, Iterable[float]]) -> None:
    """Grafica las curvas de costo y confianza del discriminador."""

    epocas = range(1, len(historial['costo_generador']) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epocas, historial['costo_generador'], label='Generador')
    plt.plot(epocas, historial['costo_discriminador'], label='Discriminador')
    plt.title('Evolución del costo')
    plt.xlabel('Época')
    plt.ylabel('Costo BCE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epocas, historial['prob_real'], label='D(x real)')
    plt.plot(epocas, historial['prob_fake'], label='D(G(z))')
    plt.title('Confianza del discriminador')
    plt.xlabel('Época')
    plt.ylabel('Probabilidad promedio')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()


def graficar_datos(reales: np.ndarray, sinteticos: np.ndarray) -> None:
    """Compara visualmente los datos reales frente a los generados."""

    plt.figure(figsize=(6, 6))
    plt.scatter(reales[:, 0], reales[:, 1], alpha=0.3, label='Reales')
    plt.scatter(sinteticos[:, 0], sinteticos[:, 1], alpha=0.3, label='Sintéticos')
    plt.title('Comparación de distribuciones')
    plt.legend()
    plt.tight_layout()


if __name__ == '__main__':
    datos = preparar_datos_reales(n_muestras=1500, ruido=0.15, semilla=7)

    config_gen = ConfiguracionModelo(
        dimensiones=[2, 16, 16, 2],
        activaciones=['relu', 'relu', 'tanh'],
        lr=0.01,
    )
    config_disc = ConfiguracionModelo(
        dimensiones=[2, 16, 16, 1],
        activaciones=['relu', 'relu', 'sigmoide'],
        lr=0.01,
    )

    generador, discriminador, historial = entrenar_gan(
        datos,
        config_generador=config_gen,
        config_discriminador=config_disc,
        epocas=200,
        tamano_lote=128,
        dimension_ruido=2,
        semilla=42,
    )

    muestras_sinteticas = generar_datos_sinteticos(generador, 500, dimension_ruido=2)
    graficar_historial(historial)
    graficar_datos(datos, muestras_sinteticas)
    plt.show()

