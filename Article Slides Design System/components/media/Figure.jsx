import React from 'react';

export function Figure({ src, alt = '', label = 'Figure', number, caption, ratio = '16 / 9', fit = 'contain', align = 'left', children, style }) {
  const media = children || (src
    ? <img src={src} alt={alt} style={{ display: 'block', width: '100%', aspectRatio: ratio, objectFit: fit, margin: 0 }} />
    : <div style={{ aspectRatio: ratio, width: '100%', background: 'var(--myst-color-bg-secondary)', border: '1px dashed var(--myst-color-border-strong)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--myst-color-text-tertiary)', fontSize: 'var(--slide-fs-label)' }}>{alt || 'Image'}</div>);
  return (
    <figure style={{ margin: 0, ...style }}>
      {media}
      {(caption || number != null) && (
        <figcaption style={{ marginTop: '0.857em', fontSize: '0.875em', lineHeight: 1.714, color: 'var(--myst-color-text-tertiary)', textAlign: align }}>
          {number != null && <span style={{ fontWeight: 700, marginRight: '0.25em', color: 'var(--myst-color-text-secondary)' }}>{label} {number}:</span>}
          {caption}
        </figcaption>
      )}
    </figure>
  );
}
