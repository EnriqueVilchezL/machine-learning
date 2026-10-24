"""Compila el libro MyST de book/ a HTML y copia las diapositivas al sitio generado.

MyST no copia los archivos que usa la directiva iframe ni ajusta su ruta, así que cada slides.html
y la carpeta assets/ (con la animación de portada hero.html) se copian a book/_build/html con la misma
estructura, y en cada página la ruta local del iframe se cambia por la ruta absoluta en el sitio
(con BASE_URL si está definida).
Uso: python tools/build_book.py
"""
import os
import shutil
import subprocess
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent / "book"
SITE = BOOK / "_build" / "html"
DECK_NAME = "slides.html"
HERO = "assets/hero.html"


def copy_decks() -> list[Path]:
    decks = sorted(BOOK.glob(f"*/*/{DECK_NAME}"))
    for deck in decks:
        target = SITE / deck.relative_to(BOOK)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(deck, target)
    shutil.copytree(BOOK / "assets", SITE / "assets", dirs_exist_ok=True)
    return decks


def rewrite_iframe(page_slug: str, local_src: str, site_path: str) -> None:
    """Cambia la ruta local del iframe por la ruta absoluta en el sitio, en el HTML y el JSON de la página."""
    url = f"{os.environ.get('BASE_URL', '')}/{site_path}"
    page_html = SITE / "index.html" if page_slug == "index" else SITE / page_slug / "index.html"
    for page in (SITE / f"{page_slug}.json", page_html):
        if page.exists():
            text = page.read_text(encoding="utf-8")
            text = text.replace(f'"src":"{local_src}"', f'"src":"{url}"').replace(f'src="{local_src}"', f'src="{url}"')
            page.write_text(text, encoding="utf-8")


def main() -> None:
    subprocess.run(["myst", "build", "--html"], cwd=BOOK, check=True)
    for deck in copy_decks():
        rewrite_iframe(deck.parent.name, DECK_NAME, deck.relative_to(BOOK).as_posix())
    rewrite_iframe("index", HERO, HERO)


if __name__ == "__main__":
    main()
