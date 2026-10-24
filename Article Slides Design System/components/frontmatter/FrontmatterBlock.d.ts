import * as React from 'react';

export interface Author { name: string; affiliations?: string[]; orcid?: string; email?: string; corresponding?: boolean; }
export interface Affiliation { id: string; name: string; }

/**
 * The MyST article title block for a title slide: subject | venue header, h1 title, subtitle, authors × affiliations grid, date · DOI.
 */
export interface FrontmatterBlockProps {
  subject?: string;
  venue?: string;
  volume?: string | number;
  issue?: string | number;
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  authors?: Author[];
  affiliations?: Affiliation[];
  /** ISO date YYYY-MM-DD, rendered "October 22, 2026" */
  date?: string;
  /** Bare DOI or doi.org URL */
  doi?: string;
  openAccess?: boolean;
  github?: string;
  style?: React.CSSProperties;
}
export function FrontmatterBlock(props: FrontmatterBlockProps): JSX.Element;
