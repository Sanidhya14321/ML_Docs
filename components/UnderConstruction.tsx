
import React from 'react';
import { motion } from 'framer-motion';
import { Construction, Hammer, BookDashed } from 'lucide-react';

interface UnderConstructionProps {
  title?: string;
}

export const UnderConstruction: React.FC<UnderConstructionProps> = ({ title }) => {
  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center text-center p-8 border border-dashed border-border-strong rounded-none bg-surface-active">
      <motion.div 
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: "spring", bounce: 0.5 }}
        className="w-24 h-24 bg-surface border border-border-strong rounded-none flex items-center justify-center mb-6 relative overflow-hidden"
      >
        <div className="absolute inset-0 bg-brand/10 animate-pulse"></div>
        <Construction size={48} className="text-brand relative z-10" />
      </motion.div>

      <h2 className="text-3xl font-heading font-black text-text-primary uppercase tracking-tight mb-4">
        {title || "Content In Development"}
      </h2>
      
      <p className="text-text-secondary font-mono max-w-md mx-auto leading-relaxed mb-8">
        This module is currently being authored by our Data Science team. 
        We are building interactive visualizations for this topic.
      </p>

      <div className="flex gap-4">
        <div className="flex items-center gap-2 px-4 py-2 rounded-none bg-surface border border-border-strong text-[10px] font-mono font-black uppercase tracking-widest text-text-muted">
            <Hammer size={12} />
            <span>Engineering</span>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 rounded-none bg-surface border border-border-strong text-[10px] font-mono font-black uppercase tracking-widest text-text-muted">
            <BookDashed size={12} />
            <span>Researching</span>
        </div>
      </div>
    </div>
  );
};
