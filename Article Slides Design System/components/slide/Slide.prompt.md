A 1920×1080 article-style slide frame; use it as the root of every slide.

```jsx
<Slide layout="body-margin" header={<SlideHeader subject="Methods" number={3} total={12} />}
  margin={<Aside title="Note">Sample size was fixed in advance.</Aside>}>
  <h2>Study design</h2>
  <p>Plain prose children pick up the article typography.</p>
</Slide>
```

- `layout`: `body` (default, 1160px reading column), `body-margin` (column + 424px margin), `page`, `center` (title/section slides), `screen` (full bleed).
- `dark` switches to the MyST dark scheme (stone-900 background).
- Children use plain HTML: `h1` (title slides only), `h2` (slide heading), `h3`, `p`, `p.lead`, `ul/ol`, `table`, `dl`.
