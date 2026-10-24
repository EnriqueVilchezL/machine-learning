# Article Slides — design system

Slides that look like a MyST article. The visual language is lifted from the **MyST article theme** (`jupyter-book/myst-theme`, `themes/article`) and translated to a 1920×1080 canvas: the article title block, the body + margin column grid, prose typography, admonitions, `{prf:}` theorem blocks, numbered figures and equations, and the bibliography.

**Sources** (public, read at build time, Oct 2026):
- https://github.com/jupyter-book/myst-theme — `styles/theme-colors.css` (all color tokens, verbatim), `styles/index.js` (prose overrides, article grid), `styles/typography.css`, `styles/button.css`, `styles/figures.css`, `themes/article/app/components/*`, `packages/frontmatter/src/*`, `packages/myst-to-react/src/{admonitions,proof,aside,code}.tsx`, `packages/site/src/components/{Abstract,Bibliography,Keywords,FooterLinksBlock}.tsx`
- https://github.com/tailwindlabs/heroicons — icon SVGs (MIT)
- https://github.com/continuous-foundation/scienceicons — ORCID, ROR, open-access, email, GitHub icons

There is **no logo**. MyST's own mark is not used; wherever a mark would go, set the deck or venue name in plain type.

## Index
- `styles.css` — entry point (imports only) → `tokens/colors.css`, `tokens/typography.css`, `tokens/spacing.css`, `tokens/base.css` (link styles + `.ats-prose` slide prose)
- `guidelines/` — foundation cards (Colors, Type, Spacing)
- `components/` — React primitives (see below), one `@dsCard` per folder
- `templates/article-slides/ArticleSlides.dc.html` — 14-slide starter deck (title, abstract, outline, section, content+margin, figure, two figures, equation+definition, theorem+proof, code, table, takeaway admonitions, pull-quote, references). Tweaks: `dark`, `showMargin`.
- `assets/icons/` — `outline/`, `solid/` (Heroicons 24), `science/` (scienceicons)
- `thumbnail.html`, `SKILL.md`

## Components
- **slide/** — `Slide` (1920×1080 frame; layouts `body`, `body-margin`, `page`, `center`, `screen`; `dark`), `SlideHeader` (subject | venue · volume running line + slide number), `FooterLink` (prev/next card)
- **frontmatter/** — `FrontmatterBlock` (title slide block: header line, h1, subtitle, Authors × Affiliations grid, date · DOI), `Abstract`
- **blocks/** — `Admonition` (10 kinds, `simple`), `Proof` (15 `{prf:}` kinds), `Aside` (margin / topic)
- **media/** — `Figure`, `Equation` (KaTeX if loaded), `CodeBlock`, `Blockquote`
- **backmatter/** — `References`
- **core/** — `Button` (primary / chip), `Icon` (+ `ICON_NAMES`)

Intentional additions (no direct MyST counterpart): `Slide` and `SlideHeader` — slides need a fixed frame and a running header; `SlideHeader` reuses the frontmatter header-line styling. `Icon` wraps the copied SVG sets.

## Content fundamentals
- Voice is the voice of a paper: third person or "we" for the authors; the audience is not addressed as "you" except in tips.
- Sentence case everywhere: slide headings, figure captions, admonition titles ("Take-home", "Limitation"). Title Case only for the talk title if the author wants it.
- Slide headings state the finding ("Error rate falls once export is automated"), not the topic ("Results").
- Captions start with the bold enumerator: **Figure 1:** then a full sentence. Tables, programs and equations are numbered too; refer to them by number ("see Theorem 1", "Eq. (1)").
- Small caps for subject and venue on the header line. Thin uppercase only for the "Authors / Affiliations" labels.
- No emoji. No exclamation marks. Numbers stay precise (n = 12, 95% intervals). Inline code in `pink-700` for identifiers and labels.
- Keep one idea per slide: a heading, a lead sentence, at most ~5 bullets, one callout.

## Visual foundations
- **Color** — white page, gray-900 headings, stone-700 body copy (`--myst-color-prose-body`, softer for reading). The only accent is the blue family: blue-700 links underlined in blue-400 at 0.1em thickness / 0.15em offset, blue-500 primary buttons. Other hues appear only as semantic signals: admonitions (info blue, tip green, warning amber, danger red) and proofs (theorem purple, example orange, proof gray). Dark scheme (`.dark`): stone-900 page, slate-800 surfaces, blue-400 links, admonition backgrounds slate-900.
- **Type** — Tailwind's system stacks: `ui-sans-serif, system-ui…` for everything, `ui-monospace…` for code. No webfonts (the source theme ships none). Prose ratios are kept exactly and scaled to a 32px base: h1 2.25em/800, h2 1.5em/700, h3 1.25em/600, lead 1.25em secondary gray, small 0.875em, labels 0.75em. Body line-height 1.75. Title slides may push h1 to 88px.
- **Layout** — the article grid: 120px side padding, a 1160px body column, 96px gutter, 424px right margin column for asides and captions; 96px top/bottom padding. Headings and text are left-aligned; only section/quote slides center vertically. Figures may break out to the full page width (`page`/`screen`), as in MyST.
- **Backgrounds** — flat white (or gray-50 for section dividers). No gradients, textures, patterns or full-bleed photography unless the figure itself is a photo.
- **Borders & rules** — 1px gray-200 hairlines for header separators, tables, proof headers. The colored 4px left border is reserved for admonitions, proofs and executable code cells (2px for `simple`).
- **Shadows** — Tailwind `shadow` (proofs, code, topic asides) and `shadow-md` (admonitions). Nothing heavier except `shadow-2xl` in dark mode.
- **Radii** — one radius: 0.25rem (`rounded`). Admonitions, buttons and cards use it; proofs, figures, images and simple admonitions are square.
- **Cards** — there are none in the marketing sense. The closest is the footer link: 1px border, rounded, shadow-sm; on hover the border and text turn blue-600 and the shadow grows to `shadow-lg`.
- **Hover/press** — links lighten to blue-500; buttons darken to blue-700; chips invert to slate-800 with white text; arrows nudge 0.25em. No press-shrink.
- **Motion** — minimal: 150ms color/transform transitions, a 1s ease-out fade-in. Slides change without animated transitions.
- **Transparency/blur** — only the translucent sticky header in the web theme (bg at 80%); not used on slides.
- **Imagery** — scientific figures: plots, diagrams, microscopy. Shown flush, unframed, at full column width, always with a numbered caption.

## Iconography
- Heroicons 24 (outline + solid), exactly as MyST maps them: note → information-circle, important → bolt, seealso → arrow-right-circle, tip → pencil-square, hint → light-bulb, attention → megaphone, warning → exclamation-triangle (solid), caution → exclamation-circle, danger → exclamation-circle (solid), error → x-circle. Navigation: arrow-left / arrow-right, document-arrow-down.
- scienceicons for scholarly metadata: ORCID, ROR, open access, email, GitHub; shown at 60% opacity, full on hover.
- Icons are never decorative and never set in colored circles. No emoji, no unicode pictographs (except ∎ ending a proof and · between date and DOI).
- Use `<Icon name="outline/…">`; the raw SVGs live in `assets/icons/`.
