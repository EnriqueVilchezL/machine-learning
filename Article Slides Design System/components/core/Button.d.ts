import * as React from 'react';

/** MyST `.button` (blue-500 fill, white bold text, 0.25rem radius) or the slate "chip" from the article back-bar. */
export interface ButtonProps {
  variant?: 'primary' | 'chip';
  href?: string;
  /** Icon name, e.g. "outline/document-arrow-down" */
  icon?: string;
  onClick?: () => void;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}
export function Button(props: ButtonProps): JSX.Element;
