import * as React from 'react';

/** MyST {aside}/{margin}/{sidebar}/{topic}. margin = small text for the right margin column; topic = boxed, gray title bar. */
export interface AsideProps {
  kind?: 'margin' | 'sidebar' | 'topic';
  title?: React.ReactNode;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}
export function Aside(props: AsideProps): JSX.Element;
