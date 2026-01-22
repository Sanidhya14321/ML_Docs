
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
            className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-[150]"
            onClick={onClose}
          />
          <motion.div 
            initial={{ opacity: 0, scale: 0.8, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.8, y: 20 }}
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md z-[160] px-4"
          >
            <div className="bg-[#020617] border border-indigo-500/30 rounded-3xl shadow-2xl overflow-hidden relative p-8 text-center">
              
              {/* Decorative Glow */}
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-64 h-64 bg-indigo-500/20 rounded-full blur-[80px] pointer-events-none"></div>

              <motion.div 
                initial={{ scale: 0, rotate: -20 }}
                animate={{ scale: 1, rotate: 0 }}
                transition={{ type: "spring", stiffness: 200, delay: 0.1 }}
                className="w-24 h-24 bg-gradient-to-br from-indigo-500 to-violet-600 rounded-full flex items-center justify-center mx-auto mb-6 shadow-xl shadow-indigo-500/30 relative"
              >
                 <Trophy size={40} className="text-white" />
                 <motion.div 
                    animate={{ rotate: 360 }} 
                    transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                    className="absolute inset-0 border-2 border-white/20 border-dashed rounded-full"
                 />
                 <div className="absolute -top-1 -right-1 bg-yellow-400 p-1.5 rounded-full border-2 border-[#020617]">
                    <Sparkles size={12} className="text-yellow-900" />
                 </div>
              </motion.div>

              <h2 className="text-2xl font-serif font-bold text-white mb-2">Curriculum Complete!</h2>
              <p className="text-slate-400 text-sm leading-relaxed mb-8">
                You have successfully mastered all topics in the AI Engineering Certification. Your neural pathways are now upgraded.
              </p>

              <div className="space-y-3">
                <button 
                  onClick={onViewCertificate}
                  className="w-full py-3.5 rounded-xl bg-white hover:bg-slate-100 text-slate-900 font-bold shadow-lg shadow-indigo-500/10 transition-all flex items-center justify-center gap-2 group"
                >
                  <FileText size={18} className="text-indigo-600 group-hover:scale-110 transition-transform" />
                  View Certificate
                </button>
                
                <button 
                  onClick={onStartOver}
                  className="w-full py-3.5 rounded-xl bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-400 hover:text-white transition-all flex items-center justify-center gap-2"
                >
                  <RotateCcw size={16} />
                  Start Over
                </button>
              </div>

              <div className="mt-6 flex items-center justify-center gap-2 text-[10px] text-emerald-500 font-mono uppercase tracking-widest">
                 <CheckCircle size={12} /> All Modules Verified
              </div>

            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
