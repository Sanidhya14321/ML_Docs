import React from 'react';
import { motion } from 'framer-motion';

export const LoadingOverlay: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-6">
      <div className="relative w-20 h-20">
        <motion.div 
          className="absolute inset-0 border-4 border-indigo-500/20 rounded-full"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        />
        <motion.div 
          className="absolute inset-0 border-4 border-t-indigo-500 rounded-full"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        />
      </div>
      <motion.p 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-slate-500 font-mono text-[10px] uppercase tracking-[0.3em] animate-pulse"
      >
        Initializing Neural Weights...
      </motion.p>
    </div>
  );
};