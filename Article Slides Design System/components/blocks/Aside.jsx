import React from 'react';

export function Aside({ kind = 'margin', title, children, style }) {
  if (kind === 'topic') {
    return (
      <aside style={{ margin: '1.25em 0', boxShadow: 'var(--shadow)', overflow: 'hidden', background: 'var(--myst-color-bg)', ...style }}>
        {title && <div style={{ margin: 0, fontWeight: 500, padding: '0.5em 1em', borderTop: '1px solid var(--myst-color-border)', borderBottom: '1px solid var(--myst-color-border)', background: 'var(--myst-color-surface)', color: 'var(--myst-color-text)' }}>{title}</div>}
        <div style={{ padding: '0.5em 1em' }}>{children}</div>
      </aside>
    );
  }
  return (
    <aside style={{ fontSize: 'var(--slide-fs-small)', lineHeight: 'var(--slide-lh-small)', ...style }}>
      {title && <div style={{ fontSize: 'var(--slide-fs-body)', lineHeight: 1.5, fontWeight: 600, color: 'var(--myst-color-text)', marginBottom: '0.25em' }}>{title}</div>}
      <div>{children}</div>
    </aside>
  );
}
