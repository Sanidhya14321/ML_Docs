
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
    <div className="my-8 rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700/50 shadow-2xl bg-[#1e1e1e] group relative font-mono text-sm transition-colors duration-300">
      {/* Terminal Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-[#252526] border-b border-white/5 select-none">
        <div className="flex items-center gap-4">
          <div className="flex space-x-2 opacity-80 hover:opacity-100 transition-opacity">
            <div className="w-3 h-3 rounded-full bg-[#ff5f56]"></div>
            <div className="w-3 h-3 rounded-full bg-[#ffbd2e]"></div>
            <div className="w-3 h-3 rounded-full bg-[#27c93f]"></div>
          </div>
          <div className="flex items-center gap-2 ml-2">
            {filename && (
                <div className="flex items-center gap-2 opacity-60">
                    <Terminal size={12} className="text-slate-400" />
                    <span className="text-xs text-slate-300 font-medium tracking-wide">{filename}</span>
                </div>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3">
             <div className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                {language}
             </div>
             <button 
                onClick={handleCopy}
                className="p-1.5 rounded-md hover:bg-white/10 transition-all text-slate-400 hover:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                title="Copy Code"
                aria-label="Copy code to clipboard"
                >
                <AnimatePresence mode="wait">
                    {copied ? (
                    <motion.div key="check" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }}>
                        <Check size={14} className="text-emerald-400" />
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
      <div className="relative overflow-x-auto custom-scrollbar bg-[#1e1e1e]">
        <pre className={`language-${language} !bg-transparent !m-0 !p-6 !font-mono !text-[13.5px] !leading-loose`}>
          <code ref={codeRef} className={`language-${language}`}>
            {code.trim()}
          </code>
        </pre>
      </div>
    </div>
  );
};
