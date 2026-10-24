import React from 'react';
import { Icon } from '../core/Icon.jsx';

export function FooterLink({ title, group, href = '#', direction = 'next', style }) {
  const [h, setH] = React.useState(false);
  const prev = direction === 'prev';
  return (
    <a href={href} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)}
      style={{ flex: 1, display: 'block', padding: '1em', fontWeight: 400, textDecoration: 'none', color: h ? 'var(--myst-color-active)' : 'var(--myst-color-text-secondary)', border: '1px solid', borderColor: h ? 'var(--myst-color-active)' : 'var(--myst-color-border)', borderRadius: 'var(--radius)', boxShadow: h ? '0 10px 15px -3px rgb(0 0 0 / 0.1),0 4px 6px -4px rgb(0 0 0 / 0.1)' : '0 1px 2px 0 rgb(0 0 0 / 0.05)', transition: 'all 150ms', ...style }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5em' }}>
        {prev && <Icon name="outline/arrow-left" size="1.5em" style={{ transform: h ? 'translateX(-0.25em)' : 'none', transition: 'transform 150ms' }} />}
        <div style={{ flexGrow: 1, textAlign: prev ? 'right' : 'left' }}>
          <div style={{ fontSize: '0.75em', color: 'var(--myst-color-text-tertiary)' }}>{group || '\u00a0'}</div>
          {title}
        </div>
        {!prev && <Icon name="outline/arrow-right" size="1.5em" style={{ transform: h ? 'translateX(0.25em)' : 'none', transition: 'transform 150ms' }} />}
      </div>
    </a>
  );
}
