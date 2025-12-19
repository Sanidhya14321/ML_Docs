
import React, { useState, useEffect, useRef } from 'react';
import { Copy, Check, Terminal } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Declare Prism global
declare const Prism: any;

interface CodeBlockProps {
  code: string;
  language?: string;
  filename?: string;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({ code, language = 'python', filename }) => {
  const [copied, setCopied] = useState(false);
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (typeof Prism !== 'undefined' && codeRef.current) {
      Prism.highlightElement(codeRef.current);
    }
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
    <div className="my-8 rounded-xl overflow-hidden border border-slate-800 shadow-2xl bg-[#1e222a] group relative">
      {/* Terminal Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-[#282c34] border-b border-black/20">
        <div className="flex items-center gap-4">
          <div className="flex space-x-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-[#ff5f56]"></div>
            <div className="w-2.5 h-2.5 rounded-full bg-[#ffbd2e]"></div>
            <div className="w-2.5 h-2.5 rounded-full bg-[#27c93f]"></div>
          </div>
          <div className="flex items-center gap-2">
            {filename && (
                <>
                    <Terminal size={10} className="text-slate-500" />
                    <span className="text-[10px] text-slate-400 font-mono tracking-wide">{filename}</span>
                </>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3">
             <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">{language}</span>
             <button 
                onClick={handleCopy}
                className="p-1.5 rounded-md hover:bg-slate-700/50 transition-all text-slate-400 hover:text-white"
                title="Copy Code"
                >
                <AnimatePresence mode="wait">
                    {copied ? (
                    <motion.div key="check" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }}>
                        <Check size={12} className="text-emerald-400" />
                    </motion.div>
                    ) : (
                    <motion.div key="copy" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }}>
                        <Copy size={12} />
                    </motion.div>
                    )}
                </AnimatePresence>
            </button>
        </div>
      </div>

      {/* Code Area */}
      <div className="relative overflow-x-auto custom-scrollbar">
        <pre className={`language-${language} !bg-transparent !m-0 !p-6 text-sm`}>
          <code ref={codeRef} className={`language-${language}`}>
            {code.trim()}
          </code>
        </pre>
      </div>
    </div>
  );
};
