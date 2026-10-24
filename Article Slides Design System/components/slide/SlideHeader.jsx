import React from 'react';

export function SlideHeader({ subject, venue, volume, issue, number, total, right, style }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', height: '1.5em', fontSize: 'var(--slide-fs-label)', fontWeight: 300, color: 'var(--myst-color-prose-body)', lineHeight: 1.5, ...style }}>
      {subject && <div style={{ flex: 'none', paddingRight: '0.5em', fontVariant: 'small-caps', ...(venue ? { marginRight: '0.5em', borderRight: '1px solid var(--myst-color-border)' } : null) }}>{subject}</div>}
      {venue && (
        <div style={{ flex: 'none', marginRight: '0.5em' }}>
          <span style={{ fontWeight: 600, fontVariant: 'small-caps' }}>{venue}</span>
          {volume != null && <span style={{ paddingLeft: '0.5em', marginLeft: '0.5em', borderLeft: '1px solid var(--myst-color-border)' }}>Volume {volume}{issue != null && <>, Issue {issue}</>}</span>}
        </div>
      )}
      <div style={{ flexGrow: 1 }} />
      {right}
      {number != null && <div style={{ flex: 'none', marginLeft: '1em', fontVariantNumeric: 'tabular-nums', color: 'var(--myst-color-text-tertiary)' }}>{number}{total != null && <> / {total}</>}</div>}
    </div>
  );
}
