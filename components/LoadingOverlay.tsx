
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
          className="absolute inset-0 border-4 border-border-strong rounded-full"
        />
        <motion.div 
          className="absolute inset-0 border-4 border-t-brand rounded-full"
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
           <BrainCircuit size={28} className="text-text-muted" />
        </motion.div>
      </div>

      <div className="space-y-3 text-center z-10">
        <motion.h3 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="text-lg font-heading font-black text-text-primary flex items-center justify-center gap-2 uppercase tracking-tight"
        >
          <Sparkles size={16} className="text-brand" />
          {message}
        </motion.h3>
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="flex flex-col items-center gap-2"
        >
          <p className="text-[10px] font-mono font-black text-text-muted uppercase tracking-[0.2em]">
            {subMessage}
          </p>
          <div className="w-32 h-1 bg-border-strong rounded-none overflow-hidden mt-1">
            <motion.div 
              className="h-full bg-brand"
              animate={{ x: [-130, 130] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
            />
          </div>
        </motion.div>
      </div>
      
      {/* Background Glow Effect */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-brand/5 rounded-full blur-3xl pointer-events-none" />
    </motion.div>
  );
};
