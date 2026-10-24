import * as React from 'react';

/** MyST admonition callout: colored 4px left border, tinted title bar with Heroicon, shadow-md. */
export interface AdmonitionProps {
  kind?: 'note' | 'important' | 'seealso' | 'tip' | 'hint' | 'attention' | 'warning' | 'caution' | 'danger' | 'error';
  /** Title text; defaults to the kind name ("Note", "Warning"…). Pass null for no title bar. */
  title?: React.ReactNode | null;
  /** Override the kind's color family */
  color?: 'blue' | 'green' | 'yellow' | 'red';
  /** `simple` class: 2px border, gray title bar, square corners */
  simple?: boolean;
  hideIcon?: boolean;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}
export function Admonition(props: AdmonitionProps): JSX.Element;
