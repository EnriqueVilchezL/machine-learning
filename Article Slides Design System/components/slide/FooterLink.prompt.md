Prev/next card from the MyST footer links block; use on closing or section slides to point to what comes next.

```jsx
<div style={{ display: 'flex', gap: '1em' }}>
  <FooterLink direction="prev" group="Previous" title="Methods" />
  <FooterLink group="Next" title="Results" />
</div>
```

Hover: border and text turn `--myst-color-active`, shadow grows, arrow nudges 0.25em.
