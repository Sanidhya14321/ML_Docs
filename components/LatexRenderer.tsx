
import React, { useEffect, useRef } from 'react';

interface LatexRendererProps {
  formula: string;
  displayMode?: boolean;
}

export const LatexRenderer: React.FC<LatexRendererProps> = ({ formula, displayMode = false }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current && (window as any).katex) {
      try {
        (window as any).katex.render(formula, containerRef.current, {
          throwOnError: false,
          displayMode: displayMode,
          output: 'html',
        });
      } catch (e) {
        console.error("KaTeX Render Error", e);
        containerRef.current.innerText = formula;
      }
    }
  }, [formula, displayMode]);

  return <div ref={containerRef} className={`${displayMode ? 'my-4 text-center py-2' : 'inline-block px-1'}`} />;
};
