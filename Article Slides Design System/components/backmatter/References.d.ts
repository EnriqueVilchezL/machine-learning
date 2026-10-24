import * as React from 'react';

/** MyST bibliography: semibold "References" header, small tertiary numbered list. */
export interface ReferencesProps {
  title?: string;
  /** Formatted citations (strings or nodes) */
  items?: React.ReactNode[];
  /** First number when the list spans slides */
  start?: number;
  style?: React.CSSProperties;
}
export function References(props: ReferencesProps): JSX.Element;
