import React from 'react';
import { Icon } from './Icon.jsx';

export function Button({ variant = 'primary', href, icon, children, onClick, style }) {
  const [h, setH] = React.useState(false);
  const base = { display: 'inline-flex', alignItems: 'center', gap: '0.25em', textDecoration: 'none', cursor: 'pointer', whiteSpace: 'nowrap', fontFamily: 'var(--font-sans)', fontSize: 'inherit', lineHeight: 1.5, transition: 'background-color 150ms, color 150ms' };
  const v = variant === 'chip'
    ? { padding: '0.25em 0.5em', fontWeight: 400, borderRadius: 'var(--radius)', border: '1px solid ' + (h ? 'transparent' : 'var(--ats-color-chip-border)'), background: h ? 'var(--ats-color-chip-hover)' : 'var(--ats-color-chip-bg)', color: h ? '#fff' : 'var(--myst-color-text)' }
    : { marginTop: '0.25rem', padding: '0.25em 1em', fontWeight: 700, borderRadius: 'var(--radius)', border: 0, background: h ? 'var(--myst-color-primary-hover)' : 'var(--myst-color-primary)', color: '#fff' };
  const Tag = href ? 'a' : 'button';
  return (
    <Tag href={href} onClick={onClick} onMouseEnter={() => setH(true)} onMouseLeave={() => setH(false)} style={{ ...base, ...v, ...style }}>
      {icon && <Icon name={icon} size="1em" />}
      <span>{children}</span>
    </Tag>
  );
}
