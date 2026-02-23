import React from 'react';
import { motion } from 'framer-motion';

interface MathBlockProps {
  children: React.ReactNode;
  label?: string;
  type?: 'inline' | 'block';
}

export const MathBlock: React.FC<MathBlockProps> = ({ children, label, type = 'block' }) => {
  if (type === 'inline') {
    return (
      <span className="math-serif italic font-bold text-indigo-400 bg-indigo-500/5 px-1 rounded mx-0.5">
        {children}
      </span>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.98 }}
      whileInView={{ opacity: 1, scale: 1 }}
      className="my-10 p-8 bg-slate-100 dark:bg-slate-900/30 border-2 border-slate-200 dark:border-slate-800/50 rounded-3xl relative group overflow-hidden"
    >
      <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-30 transition-opacity">
        <svg width="40" height="40" viewBox="0 0 40 40" fill="currentColor" className="text-indigo-500">
           <path d="M10,10 L30,10 L30,30 L10,30 L10,10 Z" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="4 2" />
        </svg>
      </div>
      
      {label && (
        <div className="text-[9px] font-black text-slate-500 uppercase tracking-[0.3em] mb-6 flex items-center gap-2">
          <div className="w-4 h-px bg-slate-300 dark:bg-slate-800"></div>
          {label}
        </div>
      )}
      
      <div className="math-serif text-2xl md:text-3xl text-slate-800 dark:text-slate-100 leading-relaxed tracking-wider text-center py-4 font-light">
        {children}
      </div>
      
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <div className="w-1.5 h-1.5 rounded-full bg-slate-300 dark:bg-slate-800"></div>
        <div className="w-1.5 h-1.5 rounded-full bg-slate-300 dark:bg-slate-800"></div>
        <div className="w-1.5 h-1.5 rounded-full bg-slate-300 dark:bg-slate-800"></div>
      </div>
    </motion.div>
  );
};