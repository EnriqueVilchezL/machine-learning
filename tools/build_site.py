"""Arma el sitio MyST en book/_stage (copia de book/) y lo compila a HTML.

Pasos: copia el libro, hace autosuficientes las diapositivas, da nombres únicos a las páginas
de cada tema, genera los índices de capítulo y el myst.yml, y ejecuta `myst build --html`.
Uso: python tools/build_site.py [--no-build]
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

from inline_slides import inline_deck

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book"
STAGE = ROOT / "_stage"

CHAPTER_TITLES = {
    "01-intro": "1. Introducción",
    "02-data-preparation": "2. Preparación de datos",
    "03-supervised": "3. Aprendizaje supervisado",
    "04-unsupervised": "4. Aprendizaje no supervisado",
    "05-reinforcement": "5. Aprendizaje por refuerzo",
    "06-deep-learning": "6. Introducción al aprendizaje profundo",
}
CHAPTER_BLURBS = {
    "01-intro": "Qué significa que una máquina aprenda y cómo se organiza un proyecto.",
    "02-data-preparation": "Del dato crudo al dato listo para modelar, y cómo medir sin hacer trampa.",
    "03-supervised": "Modelos que aprenden de ejemplos con etiqueta.",
    "04-unsupervised": "Estructura escondida en datos sin etiqueta.",
    "05-reinforcement": "Agentes que aprenden por prueba, error y recompensa.",
    "06-deep-learning": "Redes neuronales y las arquitecturas que más se usan hoy.",
}
TOPIC_ORDER = [
    "intro-ml", "data-cleaning", "feature-engineering", "normalization", "train-test-split",
    "cross-validation", "knn", "regression", "decision-trees", "logistic-regression", "svm",
    "naive-bayes", "ensembles", "kmeans", "hierarchical", "pca", "mdp", "q-learning",
    "neural-networks", "cnn", "rnn", "autoencoders", "gan", "transfer-learning", "deep-rl",
    "transformers",
]
FRONT_MATTER_TITLE = re.compile(r"^title:\s*(.+)$", re.MULTILINE)


def topic_title(page: Path) -> str:
    match = FRONT_MATTER_TITLE.search(page.read_text(encoding="utf-8"))
    return match.group(1).strip() if match else page.parent.name


def list_topics(chapter: Path) -> list[Path]:
    topics = [p for p in chapter.iterdir() if (p / "index.md").exists()]
    return sorted(topics, key=lambda p: TOPIC_ORDER.index(p.name))


def copy_book() -> None:
    shutil.rmtree(STAGE, ignore_errors=True)
    shutil.copytree(BOOK, STAGE, ignore=shutil.ignore_patterns("_build", ".ipynb_checkpoints"))


def prepare_topic(topic: Path) -> tuple[str, str, str]:
    """Renombra los archivos del tema con nombres únicos y devuelve (título, página, cuaderno)."""
    slug = topic.name
    page = topic / f"{slug}.md"
    notebook = topic / f"{slug}-cuaderno.ipynb"
    (topic / "index.md").rename(page)
    (topic / "notebook.ipynb").rename(notebook)
    text = page.read_text(encoding="utf-8").replace("(notebook.ipynb)", f"({notebook.name})")
    page.write_text(text, encoding="utf-8")
    inline_deck(topic / "slides.html", STAGE / "assets")
    return topic_title(page), page.name, notebook.name


def write_chapter_index(chapter: Path, topics: list[tuple[str, str]]) -> str:
    name = f"{chapter.name}.md"
    items = "\n".join(f"* [{title}]({slug}/{slug}.md)" for slug, title in topics)
    body = f"# {CHAPTER_TITLES[chapter.name]}\n\n{CHAPTER_BLURBS[chapter.name]}\n\nTemas:\n\n{items}\n"
    (chapter / name).write_text(body, encoding="utf-8")
    return name


def build_toc() -> str:
    lines = ["  toc:", "    - file: index.md"]
    for chapter in sorted(p for p in STAGE.iterdir() if p.name in CHAPTER_TITLES):
        topics = list_topics(chapter)
        prepared = [(t.name, *prepare_topic(t)) for t in topics]
        index_name = write_chapter_index(chapter, [(slug, title) for slug, title, _, _ in prepared])
        lines += [f"    - title: {CHAPTER_TITLES[chapter.name]}", "      children:",
                  f"        - file: {chapter.name}/{index_name}"]
        for slug, title, page, notebook in prepared:
            lines += [f"        - file: {chapter.name}/{slug}/{page}",
                      f"        - file: {chapter.name}/{slug}/{notebook}",
                      f"          title: 'Cuaderno: {title}'"]
    return "\n".join(lines) + "\n"


def write_config(toc: str) -> None:
    config = (BOOK / "myst.yml").read_text(encoding="utf-8")
    head = config.split("  toc:")[0]
    tail = "site:" + config.split("\nsite:")[1]
    (STAGE / "myst.yml").write_text(head + toc + tail, encoding="utf-8")
    (STAGE / "index.md").write_text((BOOK / "index.md").read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    copy_book()
    write_config(build_toc())
    if "--no-build" not in sys.argv:
        subprocess.run(["myst", "build", "--html"], cwd=STAGE, check=True)


if __name__ == "__main__":
    main()
