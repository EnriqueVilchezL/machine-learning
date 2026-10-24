import * as React from 'react';

/** MyST {prf:…} block: "Theorem 1 (Title)" header on a tinted bar, 4px colored left border, shadow. */
export interface ProofProps {
  /** theorem/lemma/conjecture → purple; definition/proposition → blue; example → green; criterion/corollary/property → orange; axiom/assumption/observation → amber; remark → red; proof/algorithm → gray */
  kind?: 'proof' | 'axiom' | 'lemma' | 'definition' | 'criterion' | 'remark' | 'conjecture' | 'corollary' | 'algorithm' | 'example' | 'property' | 'observation' | 'proposition' | 'assumption' | 'theorem';
  number?: string | number;
  title?: React.ReactNode;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}
export function Proof(props: ProofProps): JSX.Element;
