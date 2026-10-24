import * as React from 'react';

/** MyST abstract part: small semibold heading, gray-50 box, optional "Keywords:" line. */
export interface AbstractProps {
  /** Heading text, default "Abstract" (also "Summary", "Plain language summary") */
  title?: string;
  keywords?: string[];
  children?: React.ReactNode;
  style?: React.CSSProperties;
}
export function Abstract(props: AbstractProps): JSX.Element;
