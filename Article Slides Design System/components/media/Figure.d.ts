import * as React from 'react';

/** MyST figure: media + tertiary caption with bold "Figure N:" enumerator. Renders a dashed placeholder when no src/children. */
export interface FigureProps {
  src?: string;
  alt?: string;
  /** Enumerator label: "Figure", "Table", "Program" */
  label?: string;
  number?: string | number;
  caption?: React.ReactNode;
  /** CSS aspect-ratio for img/placeholder, default "16 / 9" */
  ratio?: string;
  fit?: 'contain' | 'cover';
  align?: 'left' | 'center';
  /** Custom media (chart, table, <image-slot>) instead of src */
  children?: React.ReactNode;
  style?: React.CSSProperties;
}
export function Figure(props: FigureProps): JSX.Element;
