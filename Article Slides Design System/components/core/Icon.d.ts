import * as React from 'react';

/** Inline SVG icon from the copied Heroicons (outline/solid) and scienceicons sets. Inherits currentColor. */
export interface IconProps {
  /** "outline/<name>" | "solid/<name>" | "science/<name>" — see ICON_NAMES */
  name: string;
  /** CSS size, default "1em" */
  size?: string | number;
  title?: string;
  className?: string;
  style?: React.CSSProperties;
}
export function Icon(props: IconProps): JSX.Element | null;
export const ICON_NAMES: string[];
