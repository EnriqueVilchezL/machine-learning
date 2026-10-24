import * as React from 'react';

/** Display math, centered, with right-aligned "(n)" number. Renders with KaTeX when window.katex is loaded (load KaTeX CSS+JS from CDN in the page). */
export interface EquationProps {
  /** LaTeX source */
  tex?: string;
  number?: string | number;
  /** Fallback content when KaTeX is unavailable */
  children?: React.ReactNode;
  style?: React.CSSProperties;
}
export function Equation(props: EquationProps): JSX.Element;
