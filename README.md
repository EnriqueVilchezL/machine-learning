# machine-learning

Course material (slides and notebooks) for an introductory machine learning course, published as a [MyST](https://mystmd.org) book.

## Requirements

- [Node.js](https://nodejs.org) 18 or newer
- MyST CLI: `npm install -g mystmd` (or `brew install mystmd`)
- Python 3 (only needed to build the full site and to rebuild notebooks)

## Run the site locally

### Option 1: live preview (while editing)

```bash
cd book
myst start
```

Open http://localhost:3000. Pages reload when you save a file. The embedded slide decks may not load in this mode, because MyST does not copy the `slides.html` files used by the iframes.

### Option 2: full site, including slides

```bash
python tools/build_book.py
python -m http.server 8000 --directory book/_build/html
```

Open http://localhost:8000. `build_book.py` runs `myst build --html`, copies every `slides.html` and `book/assets/` into `book/_build/html`, and fixes the iframe paths. Run it again after each change.

To deploy under a subpath (for example GitHub Pages at `/machine-learning`), set `BASE_URL` before building:

```bash
BASE_URL=/machine-learning python tools/build_book.py
```

## Rebuild a notebook

Notebooks are generated from jupytext scripts in `tools/src/` and executed with the project virtual environment (`.venv`):

```bash
tools/build_nb.sh tools/src/<topic>.py book/<chapter>/<topic>/<topic>-notebook.ipynb
```

For example:

```bash
tools/build_nb.sh tools/src/knn.py book/03-supervised/knn/knn-notebook.ipynb
```
