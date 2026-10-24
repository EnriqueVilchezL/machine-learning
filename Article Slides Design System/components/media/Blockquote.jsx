import React from 'react';

export function Blockquote({ children, attribution, pull = false, style }) {
  return (
    <figure style={{ margin: '1.6em 0', ...style }}>
      <blockquote style={{ margin: 0, fontWeight: 500, fontStyle: 'italic', color: 'var(--myst-color-text)', borderLeft: '0.25rem solid var(--myst-color-border)', paddingLeft: '1em', fontSize: pull ? '1.4em' : '1em', lineHeight: pull ? 1.5 : 1.75 }}>{children}</blockquote>
      {attribution && <figcaption style={{ textAlign: 'right', marginTop: '0.75em', fontSize: '0.875em', color: 'var(--myst-color-text-tertiary)' }}><span style={{ paddingRight: '0.5em' }}>—</span>{attribution}</figcaption>}
    </figure>
  );
}
