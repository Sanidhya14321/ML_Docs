
import React, { useRef, useEffect, useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  language?: string;
  readOnly?: boolean;
}

export const CodeEditor: React.FC<CodeEditorProps> = ({ 
  value, 
  onChange, 
  language = 'python', 
  readOnly = false 
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const lineNumbersRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = useState(false);

  // Sync scrolling between textarea and line numbers
  const handleScroll = () => {
    if (textareaRef.current && lineNumbersRef.current) {
      lineNumbersRef.current.scrollTop = textareaRef.current.scrollTop;
    }
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const lineCount = value.split('\n').length;
  const lines = Array.from({ length: Math.max(lineCount, 15) }, (_, i) => i + 1);

  return (
    <div className="relative h-full w-full flex bg-[#1e1e1e] font-mono text-sm overflow-hidden group">
      {/* Line Numbers Gutter */}
      <div 
        ref={lineNumbersRef}
        className="w-12 pt-4 pb-4 pr-2 text-right text-slate-600 bg-[#1e1e1e] border-r border-[#333] select-none overflow-hidden"
      >
        {lines.map(line => (
          <div key={line} className="leading-6 h-6">{line}</div>
        ))}
      </div>

      {/* Editor Area */}
      <div className="relative flex-1 h-full">
         <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onScroll={handleScroll}
            readOnly={readOnly}
            spellCheck={false}
            className="absolute inset-0 w-full h-full bg-transparent text-slate-300 p-4 pl-4 leading-6 resize-none outline-none border-none focus:ring-0 selection:bg-indigo-500/30 font-mono"
            style={{ tabSize: 4 }}
         />
      </div>
      
      {/* Actions */}
      <div className="absolute top-4 right-4 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity z-10">
        <button
            onClick={handleCopy}
            className="p-2 rounded-lg bg-slate-800/90 text-slate-400 hover:text-white hover:bg-slate-700 border border-slate-700 shadow-xl backdrop-blur-sm transition-all"
            title="Copy Code"
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

      {/* Language Badge */}
      <div className="absolute bottom-2 right-4 pointer-events-none opacity-50 group-hover:opacity-100 transition-opacity">
        <span className="text-[10px] font-black uppercase tracking-widest text-slate-500 bg-slate-800/50 px-2 py-1 rounded">
            {language}
        </span>
      </div>
    </div>
  );
};
