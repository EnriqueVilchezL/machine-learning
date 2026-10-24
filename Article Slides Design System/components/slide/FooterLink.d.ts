import * as React from 'react';

/** Prev/next navigation card from the MyST footer links block. */
export interface FooterLinkProps {
  title: React.ReactNode;
  /** Small tertiary label above the title, e.g. the section or "Next" */
  group?: string;
  href?: string;
  direction?: 'next' | 'prev';
  style?: React.CSSProperties;
}
export function FooterLink(props: FooterLinkProps): JSX.Element;
