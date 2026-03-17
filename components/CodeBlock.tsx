
import React, { useState, useEffect, useRef } from 'react';
import { Copy, Check, Terminal } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Prism from 'prismjs';

// We need to import the CSS for the theme, but it's currently loaded via CDN in index.html.
// If we wanted to be fully self-contained, we would import it here:
// import 'prismjs/themes/prism-one-dark.css'; 
// However, to respect the "reduce initial bundle size" request for *components*, 
// we will stick to dynamic imports for languages.

interface CodeBlockProps {
  code: string;
  language?: string;
  filename?: string;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({ code, language = 'python', filename }) => {
  const [copied, setCopied] = useState(false);
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const highlight = async () => {
      if (codeRef.current) {
        try {
            // Core languages like 'javascript', 'css', 'clike', 'markup' are included in the main bundle usually,
            // but for others we need to import them.
            // We use a try-catch because the language might not exist or might be 'text'.
            if (language && language !== 'text' && language !== 'javascript' && language !== 'css' && language !== 'html') {
                await import(`prismjs/components/prism-${language}`);
            }
            Prism.highlightElement(codeRef.current);
        } catch (e) {
            console.warn(`Failed to load Prism language: ${language}`, e);
            // Fallback to plain text or just highlight what we can
            Prism.highlightElement(codeRef.current);
        }
      }
    };
    highlight();
  }, [code, language]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <div className="my-8 rounded-none overflow-hidden border border-border-strong bg-app group relative font-mono text-sm transition-all duration-300">
      {/* Terminal Header */}
      <div className="flex items-center justify-between px-5 py-3 bg-surface border-b border-border-strong select-none">
        <div className="flex items-center gap-6">
          <div className="flex space-x-1.5 opacity-40 group-hover:opacity-100 transition-opacity">
            <div className="w-2 h-2 rounded-none bg-text-muted"></div>
            <div className="w-2 h-2 rounded-none bg-text-muted"></div>
            <div className="w-2 h-2 rounded-none bg-text-muted"></div>
          </div>
          <div className="flex items-center gap-3">
            {filename && (
                <div className="flex items-center gap-2 opacity-60">
                    <Terminal size={12} className="text-brand" />
                    <span className="text-[10px] text-text-primary font-black uppercase tracking-widest">{filename}</span>
                </div>
            )}
          </div>
        </div>
        <div className="flex items-center gap-4">
             <div className="px-2 py-0.5 rounded-none bg-app border border-border-strong text-[9px] font-black text-text-muted uppercase tracking-[0.2em]">
                {language}
             </div>
             <button 
                onClick={handleCopy}
                className="p-1.5 rounded-none hover:bg-brand/10 transition-all text-text-muted hover:text-brand focus:outline-none focus:ring-1 focus:ring-brand"
                title="Copy Code"
                aria-label="Copy code to clipboard"
                >
                <AnimatePresence mode="wait">
                    {copied ? (
                    <motion.div key="check" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }}>
                        <Check size={14} className="text-emerald-500" />
                    </motion.div>
                    ) : (
                    <motion.div key="copy" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }}>
                        <Copy size={14} />
                    </motion.div>
                    )}
                </AnimatePresence>
            </button>
        </div>
      </div>

      {/* Code Area */}
      <div className="relative overflow-x-auto custom-scrollbar bg-app/50">
        <pre className={`language-${language} !bg-transparent !m-0 !p-8 !font-mono !text-[13px] !leading-relaxed`}>
          <code ref={codeRef} className={`language-${language}`}>
            {code.trim()}
          </code>
        </pre>
      </div>
    </div>
  );
};
