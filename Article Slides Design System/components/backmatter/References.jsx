import React from 'react';

export function References({ title = 'References', items = [], start = 1, style }) {
  return (
    <section style={style}>
      <header style={{ fontSize: '1.125em', fontWeight: 600, color: 'var(--myst-color-text)', lineHeight: 1.6 }}>{title}</header>
      <div style={{ paddingLeft: '0.75em', fontSize: 'var(--slide-fs-label)', lineHeight: 1.6, color: 'var(--myst-color-text-tertiary)' }}>
        <ol start={start} style={{ margin: '0.5em 0 0', paddingLeft: '1.625em' }}>
          {items.map((it, i) => <li key={i} style={{ margin: '0.4em 0', overflowWrap: 'anywhere' }}>{it}</li>)}
        </ol>
      </div>
    </section>
  );
}
