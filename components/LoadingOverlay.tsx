
import React from 'react';
import { motion } from 'framer-motion';
import { BrainCircuit, Sparkles } from 'lucide-react';

interface LoadingOverlayProps {
  message?: string;
  subMessage?: string;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({ 
  message = "Synthesizing Knowledge", 
  subMessage = "Loading Neural Weights..." 
}) => {
  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
      className="min-h-[60vh] w-full flex flex-col items-center justify-center relative bg-transparent"
    >
      <div className="relative w-24 h-24 mb-8">
        <motion.div 
          className="absolute inset-0 border-4 border-slate-200/20 dark:border-slate-800/50 rounded-full"
        />
        <motion.div 
          className="absolute inset-0 border-4 border-t-indigo-500 rounded-full"
          animate={{ rotate: 360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
        />
        <motion.div 
          className="absolute inset-3 border-4 border-b-emerald-500 rounded-full"
          animate={{ rotate: -360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        />
        <motion.div 
          className="absolute inset-0 flex items-center justify-center"
          animate={{ scale: [0.9, 1.1, 0.9], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
           <BrainCircuit size={28} className="text-slate-400 dark:text-slate-600" />
        </motion.div>
      </div>

      <div className="space-y-3 text-center z-10">
        <motion.h3 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="text-lg font-serif font-bold text-slate-700 dark:text-slate-200 flex items-center justify-center gap-2"
        >
          <Sparkles size={16} className="text-indigo-500" />
          {message}
        </motion.h3>
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="flex flex-col items-center gap-2"
        >
          <p className="text-[10px] font-mono text-slate-500 uppercase tracking-[0.2em]">
            {subMessage}
          </p>
          <div className="w-32 h-1 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden mt-1">
            <motion.div 
              className="h-full bg-indigo-500"
              animate={{ x: [-130, 130] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
            />
          </div>
        </motion.div>
      </div>
      
      {/* Background Glow Effect */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none" />
    </motion.div>
  );
};
