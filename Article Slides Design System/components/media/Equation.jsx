import React from 'react';

export function Equation({ tex, number, children, style }) {
  const [html, setHtml] = React.useState(null);
  React.useEffect(() => {
    if (!tex) return;
    let tries = 0, t;
    const go = () => {
      if (window.katex) { try { setHtml(window.katex.renderToString(tex, { displayMode: true, throwOnError: false })); } catch (e) {} return; }
      if (tries++ < 40) t = setTimeout(go, 100);
    };
    go();
    return () => clearTimeout(t);
  }, [tex]);
  return (
    <div style={{ display: 'flex', alignItems: 'center', margin: '1em 0', ...style }}>
      <div style={{ flexGrow: 1, minWidth: 0, textAlign: 'center', overflowX: 'auto', color: 'var(--myst-color-text)', fontSize: '1.1em' }}>
        {html ? <div className="katex-display" style={{ margin: 0 }} dangerouslySetInnerHTML={{ __html: html }} /> : (children || <span style={{ fontFamily: 'var(--font-serif)', fontStyle: 'italic' }}>{tex}</span>)}
      </div>
      {number != null && <div style={{ flex: 'none', paddingLeft: '1em', color: 'var(--myst-color-text)', userSelect: 'none' }}>({number})</div>}
    </div>
  );
}
