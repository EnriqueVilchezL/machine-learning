import * as React from 'react';

/** MyST code block: mono 0.875em on a faint stone tint with shadow; executable cells get a 1px border + 4px blue-400 left rule. No syntax colors. */
export interface CodeBlockProps {
  code?: string;
  /** Shows a gray filename bar with a document icon */
  filename?: string;
  /** Notebook-cell style (bordered, blue left rule) */
  executable?: boolean;
  showLineNumbers?: boolean;
  startingLineNumber?: number;
  /** Line numbers to highlight */
  emphasizeLines?: number[];
  style?: React.CSSProperties;
}
export function CodeBlock(props: CodeBlockProps): JSX.Element;
