import React from 'react';
import { Icon } from '../core/Icon.jsx';

const KINDS = {
  note: ['blue', 'outline/information-circle'], important: ['blue', 'solid/bolt'], seealso: ['blue', 'outline/arrow-right-circle'],
  tip: ['green', 'outline/pencil-square'], hint: ['green', 'outline/light-bulb'],
  attention: ['yellow', 'outline/megaphone'], warning: ['yellow', 'solid/exclamation-triangle'], caution: ['yellow', 'outline/exclamation-circle'],
  danger: ['red', 'solid/exclamation-circle'], error: ['red', 'solid/x-circle'],
};
const C = { blue: 'info', green: 'tip', yellow: 'warning', red: 'danger' };
const TITLES = { note: 'Note', important: 'Important', seealso: 'See Also', tip: 'Tip', hint: 'Hint', attention: 'Attention', warning: 'Warning', caution: 'Caution', danger: 'Danger', error: 'Error' };

export function Admonition({ kind = 'note', title, color, simple = false, hideIcon = false, children, style }) {
  const [defColor, icon] = KINDS[kind] || KINDS.note;
  const c = C[color || defColor];
  const t = title === undefined ? TITLES[kind] : title;
  return (
    <aside style={{ margin: '1.25em 0', boxShadow: 'var(--shadow-md)', overflow: 'hidden', borderLeft: (simple ? 2 : 4) / 16 + 'em solid var(--myst-color-' + c + ')', borderRadius: simple ? 0 : 'var(--radius)', background: 'var(--myst-color-bg)', ...style }}>
      {t && (
        <div style={{ margin: 0, fontWeight: 500, padding: '0.25rem 0', display: 'flex', minWidth: 0, fontSize: simple ? '1em' : '1.125em', lineHeight: 1.75, background: simple ? 'var(--myst-color-surface)' : 'var(--myst-color-' + c + '-bg)', color: 'var(--myst-color-' + c + '-text)' }}>
          {!hideIcon && <span style={{ display: 'inline-flex', paddingLeft: '0.5rem', marginRight: '0.5rem', alignSelf: 'center', flex: 'none', color: 'var(--myst-color-' + c + '-text)' }}><Icon name={icon} size={simple ? '1.5em' : '1.6em'} /></span>}
          <div style={{ color: 'var(--myst-color-text)', flexGrow: 1, alignSelf: 'center', overflow: 'hidden', marginLeft: hideIcon ? '1rem' : 0 }}>{t}</div>
        </div>
      )}
      <div style={{ padding: simple ? '0.5em 1em' : '0.5em 1em' }}>{children}</div>
    </aside>
  );
}
