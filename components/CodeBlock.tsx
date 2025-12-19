
import React, { useState } from 'react';
import { Copy, Check, Terminal } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface CodeBlockProps {
  code: string;
  language?: string;
  filename?: string;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({ code, language = 'python', filename }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const lines = code.trim().split('\n');

  return (
    <div className="my-8 rounded-2xl overflow-hidden border border-slate-800 shadow-2xl bg-[#0d1117] group relative">
      {/* Header Bar */}
      <div className="flex items-center justify-between px-5 py-3 bg-slate-900/80 border-b border-slate-800 backdrop-blur-md">
        <div className="flex items-center gap-4">
          <div className="flex space-x-1.5">
            <div className="w-3 h-3 rounded-full bg-rose-500/20 border border-rose-500/40"></div>
            <div className="w-3 h-3 rounded-full bg-amber-500/20 border border-amber-500/40"></div>
            <div className="w-3 h-3 rounded-full bg-emerald-500/20 border border-emerald-500/40"></div>
          </div>
          <div className="flex items-center gap-2">
            <Terminal size={12} className="text-slate-500" />
            <span className="text-[10px] text-slate-400 font-mono uppercase tracking-widest">
              {filename || `${language}_module.py`}
            </span>
          </div>
        </div>
        <button 
          onClick={handleCopy}
          className="p-2 rounded-lg hover:bg-slate-800 transition-all text-slate-500 hover:text-white flex items-center gap-2 active:scale-95"
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
          <span className="text-[10px] font-bold uppercase tracking-tighter">{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>

      {/* Code Area */}
      <div className="p-0 overflow-x-auto custom-scrollbar flex font-mono text-sm leading-relaxed">
        {/* Line Numbers */}
        <div className="py-5 px-4 bg-slate-950/50 border-r border-slate-800 text-slate-600 text-right select-none min-w-[50px]">
          {lines.map((_, i) => (
            <div key={i}>{i + 1}</div>
          ))}
        </div>
        
        {/* Actual Code */}
        <pre className="py-5 px-6 text-slate-300 flex-1 whitespace-pre">
          <code>
            {lines.map((line, i) => (
              <div key={i} className="hover:bg-slate-800/30 transition-colors px-2 -mx-2 rounded">
                {line || ' '}
              </div>
            ))}
          </code>
        </pre>
      </div>

      {/* Syntax Shimmer */}
      <div className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-5 transition-opacity bg-gradient-to-r from-transparent via-indigo-500 to-transparent -translate-x-full group-hover:animate-[shimmer_3s_linear_infinite]"></div>
    </div>
  );
};
