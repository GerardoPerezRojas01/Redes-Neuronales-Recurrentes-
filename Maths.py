"""Resumen de los fundamentos matemáticos para el curso de GANs.

Este módulo puede ejecutarse como script para mostrar de forma ordenada
las expresiones y conceptos que el profesorado debe repasar con el grupo
antes y durante las prácticas. El objetivo es ofrecer referencias
compactas sin depender de frameworks externos, en concordancia con el
resto del repositorio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TemaMatematico:
    """Agrupa fórmulas y recordatorios conceptuales."""

    titulo: str
    ideas_clave: List[str]
    formulas: List[str]


TEMARIO_MATEMATICO: List[TemaMatematico] = [
    TemaMatematico(
        titulo="Probabilidad y variables aleatorias",
        ideas_clave=[
            "Definir distribuciones objetivo (reales) y latentes (ruido).",
            "Interpretar la generación como transformación de ruido: G(z).",
            "Recordar propiedades de mezclas gaussianas y ley de los grandes números.",
        ],
        formulas=[
            r"z \sim \mathcal{N}(0, I_d)",
            r"p_{\text{datos}}(x) = \sum_k \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)",
        ],
    ),
    TemaMatematico(
        titulo="Redes neuronales feedforward",
        ideas_clave=[
            "Revisar propagación hacia delante con pesos y sesgos densos.",
            "Explicar el papel de las activaciones no lineales y su derivada.",
            "Introducir la retropropagación como aplicación de la regla de la cadena.",
        ],
        formulas=[
            r"z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}",
            r"a^{(l)} = g(z^{(l)})",
            r"\frac{\partial \mathcal{L}}{\partial W^{(l)}} = (a^{(l-1)})^\top \delta^{(l)}",
            r"\delta^{(l-1)} = \delta^{(l)} (W^{(l)})^\top \odot g'(z^{(l-1)})",
        ],
    ),
    TemaMatematico(
        titulo="Pérdidas y divergencias",
        ideas_clave=[
            "Contrastar MSE para autoencoders con BCE para clasificación.",
            "Mostrar la conexión entre BCE y máxima verosimilitud.",
            "Relacionar el objetivo adversarial con la divergencia de Jensen-Shannon.",
        ],
        formulas=[
            r"\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N \|x_i - \hat{x}_i\|_2^2",
            r"\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum [y_i \log p_i + (1-y_i) \log(1-p_i)]",
            r"\text{JSD}(P \| Q) = \frac{1}{2} \text{KL}(P \| M) + \frac{1}{2} \text{KL}(Q \| M)",
        ],
    ),
    TemaMatematico(
        titulo="Entrenamiento adversarial",
        ideas_clave=[
            "Descomponer el juego minimax entre generador (G) y discriminador (D).",
            "Justificar las actualizaciones alternadas y el gradiente adversarial.",
            "Analizar saturación del gradiente y trucos: label smoothing, clipping.",
        ],
        formulas=[
            r"\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{datos}}}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]",
            r"\nabla_{\theta_D} \mathcal{L}_D = -\nabla_{\theta_D} \mathbb{E}_{x}[\log D(x)] - \nabla_{\theta_D} \mathbb{E}_{z}[\log(1 - D(G(z)))]",
            r"\nabla_{\theta_G} \mathcal{L}_G = -\nabla_{\theta_G} \mathbb{E}_{z}[\log D(G(z))]",
        ],
    ),
    TemaMatematico(
        titulo="Evaluación de modelos generativos",
        ideas_clave=[
            "Comparar distribuciones mediante histogramas, proyecciones y métricas simples.",
            "Explicar la noción de sobreajuste en el generador vs colapso de modo.",
            "Discutir métricas opcionales (distancia Wasserstein, FID) aunque no se implementen.",
        ],
        formulas=[
            r"\text{EMD}(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]",
            r"\text{FID}(X, Y) = \|\mu_X - \mu_Y\|_2^2 + \text{Tr}(\Sigma_X + \Sigma_Y - 2(\Sigma_X \Sigma_Y)^{1/2})",
        ],
    ),
]


def mostrar_matematicas() -> None:
    """Imprime el temario matemático en formato amigable."""

    for tema in TEMARIO_MATEMATICO:
        print(tema.titulo)
        print("  Ideas clave:")
        for idea in tema.ideas_clave:
            print(f"    - {idea}")
        if tema.formulas:
            print("  Fórmulas a destacar:")
            for formula in tema.formulas:
                print(f"    * {formula}")
        print()


if __name__ == "__main__":
    mostrar_matematicas()
