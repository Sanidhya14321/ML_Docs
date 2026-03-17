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
      <span className="math-serif italic font-bold text-brand bg-brand/5 px-1.5 py-0.5 rounded-none mx-0.5 border border-brand/10">
        {children}
      </span>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.98 }}
      whileInView={{ opacity: 1, scale: 1 }}
      className="my-10 p-10 bg-surface border border-border-strong rounded-none relative group overflow-hidden transition-all duration-300"
    >
      <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-20 transition-opacity pointer-events-none">
        <svg width="60" height="60" viewBox="0 0 40 40" fill="currentColor" className="text-brand">
           <path d="M10,10 L30,10 L30,30 L10,30 L10,10 Z" fill="none" stroke="currentColor" strokeWidth="1" strokeDasharray="2 2" />
        </svg>
      </div>
      
      {label && (
        <div className="text-[9px] font-mono font-black text-text-muted uppercase tracking-[0.4em] mb-8 flex items-center gap-3">
          <div className="w-6 h-px bg-border-strong"></div>
          {label}
        </div>
      )}
      
      <div className="math-serif text-2xl md:text-4xl text-text-primary leading-relaxed tracking-wider text-center py-6 font-light">
        {children}
      </div>
      
      <div className="absolute bottom-4 left-6 flex gap-2 opacity-20">
        <div className="w-1 h-1 bg-brand"></div>
        <div className="w-1 h-1 bg-brand"></div>
        <div className="w-1 h-1 bg-brand"></div>
      </div>
    </motion.div>
  );
};
