Numbered display equation (MyST `{math}` with label).

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
```
```jsx
<Equation tex="E = mc^2" number={1} />
```

Without KaTeX the TeX string shows in italic serif as a fallback.
