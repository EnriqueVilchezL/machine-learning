import React from 'react';
import { Icon } from '../core/Icon.jsx';

export function CodeBlock({ code = '', filename, executable = false, showLineNumbers = false, startingLineNumber = 1, emphasizeLines = [], style }) {
  const lines = code.replace(/\n$/, '').split('\n');
  const hl = new Set(emphasizeLines);
  return (
    <div style={{ position: 'relative', margin: '1.25em 0', fontSize: '0.875em', boxShadow: 'var(--shadow)', background: executable ? 'var(--myst-color-bg)' : 'rgb(231 229 228 / 0.1)', border: executable ? '1px solid var(--myst-color-border)' : 'none', borderLeft: executable ? '4px solid var(--myst-color-link-underline)' : undefined, ...style }}>
      {filename && (
        <div style={{ display: 'flex', alignItems: 'center', paddingLeft: '0.5em', background: 'var(--myst-color-bg-secondary)', borderBottom: '1px solid var(--myst-color-border)' }}>
          <Icon name="outline/document" size="1em" style={{ color: 'var(--myst-color-text-tertiary)' }} />
          <div style={{ padding: '0.5em', fontSize: '0.875em', lineHeight: 1, color: 'var(--myst-color-text-secondary)', fontFamily: 'var(--font-sans)' }}>{filename}</div>
        </div>
      )}
      <pre style={{ margin: 0, padding: '0.75em', overflow: 'auto', fontFamily: 'var(--font-mono)', lineHeight: 1.6, color: 'var(--myst-color-text)' }}>
        {lines.map((l, i) => (
          <div key={i} style={{ background: hl.has(i + startingLineNumber) ? 'var(--myst-color-active-bg)' : undefined, borderLeft: hl.has(i + startingLineNumber) ? '4px solid var(--myst-color-active)' : '4px solid transparent', marginLeft: '-0.75em', paddingLeft: 'calc(0.75em - 4px)' }}>
            {showLineNumbers && <span style={{ display: 'inline-block', minWidth: '1.25em', paddingRight: '1em', textAlign: 'right', userSelect: 'none', color: 'var(--myst-color-text-tertiary)' }}>{i + startingLineNumber}</span>}
            {l || '\u00a0'}
          </div>
        ))}
      </pre>
    </div>
  );
}
