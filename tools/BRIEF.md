# Guía para escribir un tema del libro

Proyecto: libro MyST de aprendizaje automático (curso de la Universidad Ejemplo). Directorio raíz: el repositorio actual.
Referencia completa ya hecha: `book/01-intro/intro-ml/` (slides.html, index.md, notebook.ipynb) y `tools/src/intro_ml.py`. LEA esos tres archivos antes de empezar y copie su estructura.

## Entregables por tema (carpeta `book/<capitulo>/<slug>/`)
1. `slides.html`: 14 a 18 diapositivas, 1280x720, usando `book/assets/slides.css` y `slides.js` (rutas relativas `../../assets/`). Copie el esqueleto de `intro-ml/slides.html` (KaTeX por CDN, cabecera, pie, barra).
2. `index.md`: página de teoría en MyST (front matter `title`, admonitions con ```{admonition}, matemáticas con `$...$` y `$$...$$`, enlace a `slides.html` y `notebook.ipynb`). Debe ser autosuficiente y didáctica, con ejemplo numérico resuelto a mano cuando aplique.
3. `notebook.ipynb`: se genera desde `tools/src/<slug>.py` (formato jupytext percent) con `tools/build_nb.sh tools/src/<slug>.py book/<cap>/<slug>/notebook.ipynb`, que además lo ejecuta y guarda salidas. Debe ejecutar sin errores.

## Idioma y estilo
- Español de Costa Rica, trato de usted, tono académico claro.
- Siga la skill de redacción: sin rayas largas ni medias (— –) ni `--` como pausa; sin "además", "crucial", "fundamental" como relleno, "panorama", "sumergirse", "potente", "robusto"; nada de "no solo X sino Y" ni "no es X, es Y"; sin cierres tipo "en resumen/en conclusión"; títulos en minúscula salvo la primera palabra y nombres propios; negritas solo en términos clave; contracciones no aplican en español. Frases directas con "es/tiene". Evite las listas con "**Término:** explicación" en exceso. Emojis solo en diapositivas y con moderación, ninguno en index.md ni notebooks.
- Fórmulas correctas y con notación consistente: $\mathbf{x}$ vectores, $y$ salida, $\hat y$ predicción, $\theta$ parámetros, $n$ ejemplos, $d$ características.

## Diapositivas: contenido y animación
- Estructura sugerida: portada, motivación con ejemplo concreto, idea intuitiva (dibujo), formalización (fórmulas), paso a paso con ejemplo numérico, ilustración animada del algoritmo, hiperparámetros/variantes, ventajas y límites, errores comunes, mini-resumen de ideas para recordar como pregunta o ejercicio, puente al cuaderno. Sin cierre redundante.
- Muchas ilustraciones: SVG en línea dibujado a mano (ejes, puntos, curvas, flechas, diagramas de bloques, redes, matrices). Cada diapositiva con contenido debe tener al menos un dibujo o fórmula en caja `.formula`.
- Animaciones: clases de `slides.css` (`a-draw` con `--len`, `a-pop`, `a-flow`, `a-pulse`, `a-float`, `a-spin`, `a-blink`, `a-grow-y`, `a-grow-x`, `a-slide-up`, `--d` para escalonar), revelado paso a paso con `.fragment` (variantes `zoom`, `left`) y al menos 3 animaciones interactivas con JavaScript propio por deck (botones "▶ Paso", deslizadores `<input type=range>` que cambian un parámetro y redibujan un SVG o `<canvas>`, clic para añadir puntos, etc.). Los botones e inputs deben llevar clase `interactive` o ser `button/input` para no avanzar la diapositiva al hacer clic. Use animaciones que expliquen (ver cómo se mueve un centroide, cómo crece un árbol, cómo se propaga un gradiente), no adorno.
- Texto legible: mínimo 19px, máximo ~8 líneas de texto por diapositiva, contenido dentro de 1280x720 (área útil: x 72 a 1208, y 96 a 650). Las diapositivas con `.cover` (portada y cierre) no muestran cabecera.
- Verifique visualmente: `tools/shoot.sh book/<cap>/<slug>/slides.html <N> <dir_temporal>` genera PNG de la diapositiva N (los `.fragment` salen ocultos en la captura, es normal) y use Read sobre el PNG. Revise como mínimo 6 diapositivas variadas por tema, corrija desbordes, textos montados y SVG cortados. Pruebe que el JavaScript no da errores (sintaxis revisada con `node --check` sobre el script extraído, si hace falta).

## Cuadernos: código limpio
- Estructura: título y objetivos (markdown), imports y constantes (`RANDOM_STATE = 42`), secciones que alternan explicación en markdown (con LaTeX) y código, visualizaciones claras con matplotlib, implementación DESDE CERO con numpy del algoritmo central y comparación con scikit-learn, experimento sobre un hiperparámetro, y una sección final "Ejercicios" con 3 propuestas. Mismo formato en todos los temas.
- Código limpio: nombres que revelan intención (en inglés técnico o español, pero consistente con intro_ml.py: identificadores en inglés), funciones pequeñas de una sola responsabilidad con type hints y docstring corto, sin números mágicos (constantes con nombre), sin código comentado, sin comentarios que repitan lo que hace el código, sin duplicar bloques (extraiga funciones de graficado).
- Solo numpy, scipy, pandas, matplotlib, seaborn, scikit-learn (instalados en `.venv`). NO torch ni tensorflow: las redes profundas se implementan en numpy. Ejecución total menor a 90 segundos, sin descargas de internet (use `sklearn.datasets` incluidos como load_iris, load_digits, make_*, o datos sintéticos).
- Use `/Users/.../.venv/bin/python` desde la raíz del repo; el script `tools/build_nb.sh` ya lo hace.

## Reglas de trabajo
- Escriba SOLO dentro de su carpeta de temas `book/<cap>/<slug>/` y `tools/src/<slug>.py`. No toque `myst.yml`, `index.md` raíz, índices de capítulo ni `book/assets/` (si necesita CSS extra, póngalo en un `<style>` del deck).
- Materiales de apoyo del profesor en `materiales/*.pdf` (puede leer los relacionados al tema para alinear enfoque y nomenclatura).
- Al terminar, responda con una lista breve: temas hechos, número de diapositivas por tema, y cualquier problema pendiente.
