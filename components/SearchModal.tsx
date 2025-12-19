import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, X, Hash, ArrowRight, Zap, BookOpen, BrainCircuit } from 'lucide-react';
import { ViewSection } from '../types';

interface SearchModalProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigate: (section: ViewSection) => void;
}

const SEARCH_REGISTRY = [
  { id: ViewSection.FOUNDATIONS, label: 'Linear Algebra', category: 'Math', icon: <Hash size={14}/> },
  { id: ViewSection.OPTIMIZATION, label: 'Gradient Descent', category: 'Engine', icon: <Zap size={14}/> },
  { id: ViewSection.REGRESSION, label: 'Linear Regression', category: 'Supervised', icon: <BookOpen size={14}/> },
  { id: ViewSection.CLASSIFICATION, label: 'Logistic Regression', category: 'Supervised', icon: <BookOpen size={14}/> },
  { id: ViewSection.DEEP_LEARNING, label: 'Neural Networks', category: 'Deep Learning', icon: <BrainCircuit size={14}/> },
  { id: ViewSection.DEEP_LEARNING, label: 'Backpropagation', category: 'Deep Learning', icon: <Zap size={14}/> },
  { id: ViewSection.REINFORCEMENT, label: 'Q-Learning', category: 'Advanced', icon: <BrainCircuit size={14}/> },
];

export const SearchModal: React.FC<SearchModalProps> = ({ isOpen, onClose, onNavigate }) => {
  const [query, setQuery] = useState('');
  
  const filtered = SEARCH_REGISTRY.filter(item => 
    item.label.toLowerCase().includes(query.toLowerCase()) ||
    item.category.toLowerCase().includes(query.toLowerCase())
  );

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        isOpen ? onClose() : undefined; // Logic handled in parent, but good to have
      }
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-slate-950/80 backdrop-blur-md z-[100]"
          />
          <motion.div 
            initial={{ opacity: 0, scale: 0.95, y: -20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -20 }}
            className="fixed top-[15%] left-1/2 -translate-x-1/2 w-full max-w-xl z-[101] p-4"
          >
            <div className="bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden">
              <div className="p-4 border-b border-slate-800 flex items-center gap-3">
                <Search size={20} className="text-slate-500" />
                <input 
                  autoFocus
                  type="text" 
                  placeholder="Search topics, algorithms, math..."
                  className="bg-transparent border-none outline-none w-full text-white placeholder:text-slate-600 font-medium"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-slate-805 border border-slate-800 text-[9px] font-black text-slate-500 uppercase">
                  ESC
                </div>
              </div>
              
              <div className="max-h-[350px] overflow-y-auto p-2">
                {filtered.length > 0 ? (
                  <div className="space-y-1">
                    {filtered.map((item, idx) => (
                      <button
                        key={idx}
                        onClick={() => {
                          onNavigate(item.id);
                          onClose();
                        }}
                        className="w-full flex items-center justify-between p-3 rounded-xl hover:bg-slate-800 group transition-all"
                      >
                        <div className="flex items-center gap-4">
                          <div className="w-8 h-8 rounded-lg bg-slate-950 flex items-center justify-center text-slate-400 group-hover:text-indigo-400 transition-colors">
                            {item.icon}
                          </div>
                          <div className="text-left">
                            <div className="text-sm font-bold text-white">{item.label}</div>
                            <div className="text-[10px] text-slate-500 uppercase font-black tracking-widest">{item.category}</div>
                          </div>
                        </div>
                        <ArrowRight size={14} className="text-slate-700 group-hover:text-indigo-500 transition-all group-hover:translate-x-1" />
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="p-12 text-center">
                    <div className="text-slate-500 text-sm italic">No topics found matching "{query}"</div>
                  </div>
                )}
              </div>
              
              <div className="p-3 bg-slate-950/50 border-t border-slate-800 flex justify-between items-center">
                <div className="flex gap-4 text-[9px] font-black text-slate-600 uppercase tracking-widest">
                   <span className="flex items-center gap-1"><ArrowRight size={10}/> Select</span>
                   <span className="flex items-center gap-1"><Search size={10}/> Navigate</span>
                </div>
                <div className="text-[9px] text-slate-600 font-mono">Found {filtered.length} results</div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};