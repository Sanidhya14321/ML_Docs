
import React, { useEffect, useRef, useState } from 'react';

interface LatexRendererProps {
  formula: string;
  displayMode?: boolean;
}

export const LatexRenderer: React.FC<LatexRendererProps> = ({ formula, displayMode = false }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isKatexLoaded, setIsKatexLoaded] = useState(false);

  useEffect(() => {
    const checkKatex = () => {
      if ((window as any).katex) {
        setIsKatexLoaded(true);
      }
    };

    checkKatex();
    const interval = setInterval(checkKatex, 100);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (containerRef.current && isKatexLoaded && (window as any).katex) {
      // Guard against quirks mode which causes KaTeX to throw
      if (document.compatMode === 'BackCompat') {
         console.warn("KaTeX skipped due to Quirks Mode.");
         containerRef.current.innerText = formula;
         return;
      }

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
  }, [formula, displayMode, isKatexLoaded]);

  return <div ref={containerRef} className={`${displayMode ? 'my-4 text-center py-2' : 'inline-block px-1'}`} />;
};
