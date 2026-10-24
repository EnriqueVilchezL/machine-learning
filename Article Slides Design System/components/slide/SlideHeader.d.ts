import * as React from 'react';

/** Running header from the MyST frontmatter block: subject (small caps) | venue · volume, badges and slide number on the right. */
export interface SlideHeaderProps {
  /** Small-caps subject, e.g. "Research Article" or the section name */
  subject?: string;
  /** Journal / conference / venue, bold small caps */
  venue?: string;
  volume?: string | number;
  issue?: string | number;
  number?: number;
  total?: number;
  /** Extra right-aligned content (badges, icons) */
  right?: React.ReactNode;
  style?: React.CSSProperties;
}
export function SlideHeader(props: SlideHeaderProps): JSX.Element;
