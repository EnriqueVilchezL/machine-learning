import * as React from 'react';

/** Prose blockquote (italic, medium, 0.25rem gray left rule). pull = MyST pull-quote at 1.4em. Attribution right-aligned with em-dash. */
export interface BlockquoteProps {
  children?: React.ReactNode;
  attribution?: React.ReactNode;
  pull?: boolean;
  style?: React.CSSProperties;
}
export function Blockquote(props: BlockquoteProps): JSX.Element;
