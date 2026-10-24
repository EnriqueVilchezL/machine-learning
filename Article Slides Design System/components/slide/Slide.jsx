import React from 'react';

const COLS = { body: 'var(--slide-body-w)', 'body-margin': 'var(--slide-body-w) var(--slide-gutter) var(--slide-margin-w)', page: 'minmax(0,1fr)', center: 'var(--slide-body-w)', screen: 'minmax(0,1fr)' };

export function Slide({ layout = 'body', header, margin, footer, children, align = 'start', dark = false, background, label, style, className }) {
  const isScreen = layout === 'screen';
  const isCenter = layout === 'center';
  return (
    <section data-screen-label={label} className={['ats-prose', dark ? 'dark' : '', className].filter(Boolean).join(' ')}
      style={{ position: 'relative', width: 'var(--slide-w)', height: 'var(--slide-h)', overflow: 'hidden', boxSizing: 'border-box', background: background || 'var(--myst-color-bg)', display: 'flex', flexDirection: 'column', padding: isScreen ? 0 : 'var(--slide-pad-top) var(--slide-pad-x) var(--slide-pad-bottom)', ...style }}>
      {header && <div style={{ flex: 'none', marginBottom: 56, ...(isScreen ? { position: 'absolute', top: 'var(--slide-pad-top)', left: 'var(--slide-pad-x)', right: 'var(--slide-pad-x)', zIndex: 2, marginBottom: 0 } : null) }}>{header}</div>}
      <div style={{ flex: 1, minHeight: 0, display: 'grid', gridTemplateColumns: COLS[layout] || COLS.body, justifyContent: isCenter ? 'center' : 'start', alignContent: isCenter ? 'center' : align === 'center' ? 'center' : align === 'end' ? 'end' : 'start' }}>
        <div style={{ minWidth: 0 }}>{children}</div>
        {layout === 'body-margin' && <div />}
        {layout === 'body-margin' && <div style={{ minWidth: 0, fontSize: 'var(--slide-fs-small)', lineHeight: 'var(--slide-lh-small)' }}>{margin}</div>}
      </div>
      {footer && <div style={{ flex: 'none', marginTop: 40 }}>{footer}</div>}
    </section>
  );
}
