import React from 'react';

const SCHEME = { proof: 'proof', algorithm: 'proof', lemma: 'theorem', conjecture: 'theorem', theorem: 'theorem', observation: 'warning', assumption: 'warning', axiom: 'warning', criterion: 'example', corollary: 'example', property: 'example', example: 'success', remark: 'danger', definition: 'info', proposition: 'info' };

export function Proof({ kind = 'theorem', number, title, children, style }) {
  const s = SCHEME[kind] || 'proof';
  const label = kind.slice(0, 1).toUpperCase() + kind.slice(1);
  return (
    <aside style={{ margin: '1.25em 0', boxShadow: 'var(--shadow)', overflow: 'hidden', borderLeft: '0.25em solid var(--myst-color-' + s + ')', background: 'var(--myst-color-bg)', ...style }}>
      <div style={{ margin: 0, fontWeight: 500, padding: '0.5rem 0', display: 'flex', borderTop: '1px solid var(--myst-color-border)', borderBottom: '1px solid var(--myst-color-border)', background: 'var(--myst-color-' + s + '-bg)' }}>
        <div style={{ marginLeft: '1em', color: 'var(--myst-color-' + s + '-text)' }}>{label}{number != null && ' ' + number}{title && <> ({title})</>}</div>
      </div>
      <div style={{ padding: '0.5em 1em' }}>{children}</div>
    </aside>
  );
}
