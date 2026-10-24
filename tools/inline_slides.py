"""Hace autosuficiente un slides.html: incrusta slides.css, slides.js e imágenes como data URI.
Uso: python tools/inline_slides.py ruta/slides.html (se sobrescribe; úselo sobre una copia de staging)."""
import base64
import mimetypes
import re
import sys
from pathlib import Path

ASSET_PATTERN = re.compile(r'(?:\.\./)+assets/([\w.-]+)')


def read_asset(assets_dir: Path, name: str) -> bytes:
    return (assets_dir / name).read_bytes()


def to_data_uri(assets_dir: Path, name: str) -> str:
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    encoded = base64.b64encode(read_asset(assets_dir, name)).decode()
    return f"data:{mime};base64,{encoded}"


def inline_css(html: str, assets_dir: Path) -> str:
    css = read_asset(assets_dir, "slides.css").decode()
    css = re.sub(r"url\(([\w.-]+\.jpg)\)", lambda m: f"url({to_data_uri(assets_dir, m.group(1))})", css)
    return re.sub(r'<link rel="stylesheet" href="[^"]*slides\.css">', lambda m: f"<style>{css}</style>", html)


def inline_js(html: str, assets_dir: Path) -> str:
    js = read_asset(assets_dir, "slides.js").decode()
    return re.sub(r'<script defer src="[^"]*slides\.js"></script>', lambda m: f"<script defer>{js}</script>", html)


def inline_images(html: str, assets_dir: Path) -> str:
    return ASSET_PATTERN.sub(lambda m: to_data_uri(assets_dir, m.group(1)), html)


def inline_deck(deck: Path, assets_dir: Path) -> None:
    html = deck.read_text(encoding="utf-8")
    html = inline_css(html, assets_dir)
    html = inline_js(html, assets_dir)
    html = inline_images(html, assets_dir)
    deck.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    deck_path = Path(sys.argv[1]).resolve()
    inline_deck(deck_path, Path(sys.argv[2]).resolve())
