The article title block (MyST `FrontmatterBlock`) for the first slide of a deck.

```jsx
<Slide layout="body">
  <FrontmatterBlock subject="Research Article" venue="Conference 2026" title="Title of the talk"
    subtitle="One-line subtitle" date="2026-10-22" doi="10.0000/example"
    authors={[{ name: 'A. Author', affiliations: ['u1'], orcid: '0000-0000-0000-0000' }]}
    affiliations={[{ id: 'u1', name: 'University' }]} />
</Slide>
```

- With affiliations → two-column grid with thin uppercase "Authors / Affiliations" labels (labels only when >1 author).
- Without affiliations → comma-separated author list.
- `openAccess` / `github` show 60%-opacity science icons on the header line.
