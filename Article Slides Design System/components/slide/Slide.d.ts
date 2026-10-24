import * as React from 'react';

/**
 * A 1920×1080 slide frame laid out on the MyST article grid (body column, optional right margin column).
 * Applies the article prose styles (`.ats-prose`) so plain h1/h2/p/ul/table children are styled.
 */
export interface SlideProps {
  /** body = one reading column; body-margin = column + right margin notes; page = full content width; center = centered column, vertically centered; screen = full-bleed, no padding */
  layout?: 'body' | 'body-margin' | 'page' | 'center' | 'screen';
  /** Running header, usually <SlideHeader/> */
  header?: React.ReactNode;
  /** Right-margin content for layout="body-margin" (asides, figure notes) */
  margin?: React.ReactNode;
  footer?: React.ReactNode;
  /** Vertical alignment of the content block */
  align?: 'start' | 'center' | 'end';
  /** Apply the dark MyST scheme */
  dark?: boolean;
  background?: string;
  /** data-screen-label, e.g. "03 Methods" */
  label?: string;
  style?: React.CSSProperties;
  className?: string;
  children?: React.ReactNode;
}
export function Slide(props: SlideProps): JSX.Element;
