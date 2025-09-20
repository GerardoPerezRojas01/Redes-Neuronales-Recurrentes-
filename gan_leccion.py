"""Mapa curricular de 28 horas para el curso de GAN con NumPy puro.

El contenido ahora está dividido en scripts especializados para que el profesorado
pueda ir activándolos sesión tras sesión dentro de PyCharm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Bloque:
    titulo: str
    horas: int
    scripts: List[str]
    temas: List[str]
    matematicas: List[str] = field(default_factory=list)


ITINERARIO_28_HORAS: List[Bloque] = [
    Bloque(
        titulo='Bloque 1 (Horas 1-4): Fundamentos de redes densas',
        horas=4,
        scripts=['gan_modelos.py'],
        temas=[
            'Repaso de perceptrón multicapa y activaciones (relu, sigmoide, tanh).',
            'Inicialización de pesos y propagación hacia delante/atrás.',
            'Definición de pérdidas MSE y BCE con sus gradientes analíticos.',
        ],
        matematicas=[
            r'a^{(l)} = g(z^{(l)}),\; z^{(l)} = a^{(l-1)}W^{(l)} + b^{(l)}',
            r'\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum y \log p + (1-y) \log(1-p)',
        ],
    ),
    Bloque(
        titulo='Bloque 2 (Horas 5-10): Generador desde autoencoder',
        horas=6,
        scripts=['gan_generador_autoencoder.py', 'gan_inicio_proyecto.py'],
        temas=[
            'Entrenamiento no supervisado de autoencoders simétricos.',
            'Transferencia del decodificador como generador inicial.',
            'Exploración del historial de reconstrucción y ajuste de hiperparámetros.',
        ],
        matematicas=[
            r'\mathcal{L}_{\text{rec}} = \lVert x - \hat x \rVert_2^2',
            r'z \sim \mathcal{N}(0, I_d)',
        ],
    ),
    Bloque(
        titulo='Bloque 3 (Horas 11-16): Discriminador supervisado',
        horas=6,
        scripts=['gan_discriminador.py'],
        temas=[
            'Construcción del dataset etiquetado real/falso.',
            'Descenso de gradiente mini-batch para maximizar la verosimilitud.',
            'Métricas de desempeño: precisión y pérdida BCE.',
        ],
        matematicas=[
            r'\nabla_{\theta_D} \mathcal{L}_D = \frac{1}{N} \sum (D(x)-y) \nabla_{\theta_D} f(x)',
        ],
    ),
    Bloque(
        titulo='Bloque 4 (Horas 17-20): Entrenamiento adversarial',
        horas=4,
        scripts=['gan_entrenamiento_adversarial.py', 'gan_inicio_proyecto.py'],
        temas=[
            'Implementación del lazo alternado G ↔ D.',
            'Interpretación de las curvas de pérdida y saturación.',
            'Discusión sobre estabilidad y trucos (label smoothing, clipping).',
        ],
        matematicas=[
            r'\mathcal{L}_D = -\mathbb{E}_{x \sim p_{\text{real}}}[\log D(x)] - \mathbb{E}_{z}[\log(1-D(G(z)))]',
            r'\mathcal{L}_G = -\mathbb{E}_{z}[\log D(G(z))]',
        ],
    ),
    Bloque(
        titulo='Bloque 5 (Horas 21-22): Caso 1D',
        horas=2,
        scripts=['gan_ejemplo_1d.py'],
        temas=[
            'Definición de mezclas gaussianas y muestreo de ruido 1D.',
            'Comparación de histogramas y evaluación visual.',
        ],
        matematicas=[
            r'p(x) = \sum_k \pi_k \mathcal{N}(\mu_k, \sigma_k^2)',
        ],
    ),
    Bloque(
        titulo='Bloque 6 (Horas 23-24): Caso 2D',
        horas=2,
        scripts=['gan_ejemplo_2d.py'],
        temas=[
            'Dataset ``make_moons`` como ejemplo de soporte no lineal.',
            'Visualización de nubes de puntos reales vs generadas.',
        ],
        matematicas=[
            r'(x \pm 0.5)^2 + y^2 = 1',
        ],
    ),
    Bloque(
        titulo='Bloque 7 (Horas 25-26): Caso 3D',
        horas=2,
        scripts=['gan_ejemplo_3d.py'],
        temas=[
            'Parametrización del ``swiss roll`` y normalización por ejes.',
            'Uso de proyecciones 3D para evaluar calidad de las muestras.',
        ],
        matematicas=[
            r'(x, y, z) = (t \cos t, h, t \sin t)',
        ],
    ),
    Bloque(
        titulo='Bloque 8 (Horas 27-28): Caso imágenes 8x8',
        horas=2,
        scripts=['gan_ejemplo_imagenes.py'],
        temas=[
            'Preparación de datos de dígitos y normalización a [-1, 1].',
            'Análisis cualitativo de muestras generadas en cuadrículas.',
        ],
        matematicas=[
            r'\tilde{x} = 2 (x - x_{\min})/(x_{\max}-x_{\min}) - 1',
        ],
    ),
]


def mostrar_plan() -> None:
    """Imprime el plan de trabajo en formato legible."""

    for bloque in ITINERARIO_28_HORAS:
        print(bloque.titulo)
        print(f'  Horas estimadas: {bloque.horas}')
        print(f'  Scripts: {", ".join(bloque.scripts)}')
        print('  Temas:')
        for tema in bloque.temas:
            print(f'    - {tema}')
        if bloque.matematicas:
            print('  Matemáticas clave:')
            for formula in bloque.matematicas:
                print(f'    * {formula}')
        print()


if __name__ == '__main__':
    mostrar_plan()
