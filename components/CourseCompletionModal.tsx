
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Trophy, RotateCcw, FileText, CheckCircle, Sparkles } from 'lucide-react';

interface CourseCompletionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onViewCertificate: () => void;
  onStartOver: () => void;
}

export const CourseCompletionModal: React.FC<CourseCompletionModalProps> = ({ 
  isOpen, 
  onClose,
  onViewCertificate, 
  onStartOver 
}) => {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-app/80 backdrop-blur-sm z-[150]"
            onClick={onClose}
          />
          <motion.div 
            initial={{ opacity: 0, scale: 0.8, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.8, y: 20 }}
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md z-[160] px-4"
          >
            <div className="bg-surface border border-border-strong rounded-none shadow-2xl overflow-hidden relative p-8 text-center">
              
              {/* Decorative Glow */}
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-64 h-64 bg-brand/20 rounded-full blur-[80px] pointer-events-none"></div>

              <motion.div 
                initial={{ scale: 0, rotate: -20 }}
                animate={{ scale: 1, rotate: 0 }}
                transition={{ type: "spring", stiffness: 200, delay: 0.1 }}
                className="w-24 h-24 bg-brand rounded-none flex items-center justify-center mx-auto mb-6 shadow-xl shadow-brand/30 relative"
              >
                 <Trophy size={40} className="text-app" />
                 <motion.div 
                    animate={{ rotate: 360 }} 
                    transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                    className="absolute inset-0 border-2 border-app/20 border-dashed rounded-none"
                 />
                 <div className="absolute -top-1 -right-1 bg-emerald-500 p-1.5 rounded-none border-2 border-surface">
                    <Sparkles size={12} className="text-app" />
                 </div>
              </motion.div>

              <h2 className="text-2xl font-heading font-black text-text-primary mb-2 uppercase tracking-tight">CURRICULUM_COMPLETE!</h2>
              <p className="text-text-secondary text-sm leading-relaxed mb-8 font-light">
                You have successfully mastered all topics in the AI Engineering Certification. Your neural pathways are now upgraded.
              </p>

              <div className="space-y-3">
                <button 
                  onClick={onViewCertificate}
                  className="w-full py-3.5 rounded-none bg-text-primary hover:bg-brand text-app font-mono font-black text-[11px] uppercase tracking-[0.2em] shadow-lg shadow-brand/10 transition-all flex items-center justify-center gap-3 group"
                >
                  <FileText size={18} className="text-app group-hover:scale-110 transition-transform" />
                  VIEW_CERTIFICATE
                </button>
                
                <button 
                  onClick={onStartOver}
                  className="w-full py-3.5 rounded-none bg-surface border border-border-strong hover:bg-surface-hover text-text-muted hover:text-text-primary transition-all flex items-center justify-center gap-3 font-mono font-black text-[11px] uppercase tracking-[0.2em]"
                >
                  <RotateCcw size={16} />
                  START_OVER
                </button>
              </div>

              <div className="mt-6 flex items-center justify-center gap-2 text-[10px] text-emerald-500 font-mono font-black uppercase tracking-widest">
                 <CheckCircle size={12} /> ALL_MODULES_VERIFIED
              </div>

            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
