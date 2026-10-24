import React from 'react';

export function Abstract({ title = 'Abstract', keywords, children, style }) {
  return (
    <div style={style}>
      <h2 style={{ marginBottom: '0.75em', fontSize: '1em', lineHeight: 1.5, fontWeight: 600 }}>{title}</h2>
      <div style={{ padding: '0.25em 1.5em', marginBottom: '0.75em', borderRadius: '0.125rem', background: 'var(--myst-color-bg-secondary)' }}>
        <div style={{ padding: '0.75em 0' }}>{children}</div>
      </div>
      {keywords && keywords.length > 0 && (
        <div style={{ marginTop: '1em' }}>
          <span style={{ marginRight: '0.5em', fontWeight: 600, color: 'var(--myst-color-text)' }}>Keywords:</span>
          <span>{keywords.join(', ')}</span>
        </div>
      )}
    </div>
  );
}
