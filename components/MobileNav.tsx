
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, BrainCircuit } from 'lucide-react';
import { Sidebar } from './Sidebar';

interface MobileNavProps {
  isOpen: boolean;
  onClose: () => void;
  currentPath: string;
  onNavigate: (path: string) => void;
}

export const MobileNav: React.FC<MobileNavProps> = ({ isOpen, onClose, currentPath, onNavigate }) => {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div 
            initial={{ opacity: 0 }} 
            animate={{ opacity: 1 }} 
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 md:hidden"
          />
          <motion.div
            initial={{ x: '-100%' }}
            animate={{ x: 0 }}
            exit={{ x: '-100%' }}
            transition={{ type: "spring", bounce: 0, duration: 0.4 }}
            className="fixed top-0 left-0 bottom-0 w-[85vw] max-w-xs bg-slate-950 border-r border-slate-800 z-50 md:hidden flex flex-col shadow-2xl"
          >
             <div className="p-4 border-b border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                   <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-600/20">
                      <BrainCircuit size={18} className="text-white" />
                   </div>
                   <span className="font-serif font-bold text-white tracking-tight">AI Codex</span>
                </div>
                <button onClick={onClose} className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-900 transition-colors">
                   <X size={20} />
                </button>
             </div>
             <div className="flex-1 overflow-y-auto custom-scrollbar pt-4">
                <Sidebar currentPath={currentPath} onNavigate={(path) => { onNavigate(path); onClose(); }} />
             </div>
             <div className="p-4 border-t border-slate-800 text-[10px] text-slate-600 text-center">
                © 2024 AI Mastery Hub
             </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
