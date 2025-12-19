
import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Hash, ArrowRight, Zap, BookOpen, BrainCircuit, Command, CornerDownLeft } from 'lucide-react';
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
  { id: ViewSection.CLASSIFICATION, label: 'Support Vector Machines', category: 'Supervised', icon: <BookOpen size={14}/> },
  { id: ViewSection.ENSEMBLE, label: 'Random Forests', category: 'Ensemble', icon: <BrainCircuit size={14}/> },
  { id: ViewSection.DEEP_LEARNING, label: 'Neural Networks', category: 'Deep Learning', icon: <BrainCircuit size={14}/> },
  { id: ViewSection.DEEP_LEARNING, label: 'Backpropagation', category: 'Deep Learning', icon: <Zap size={14}/> },
  { id: 'deep-learning/attention-mechanism', label: 'Attention Mechanism', category: 'Deep Learning', icon: <Zap size={14}/> },
  { id: ViewSection.REINFORCEMENT, label: 'Q-Learning', category: 'RL', icon: <BrainCircuit size={14}/> },
];

export const SearchModal: React.FC<SearchModalProps> = ({ isOpen, onClose, onNavigate }) => {
  const [query, setQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);
  
  const filtered = SEARCH_REGISTRY.filter(item => 
    item.label.toLowerCase().includes(query.toLowerCase()) ||
    item.category.toLowerCase().includes(query.toLowerCase())
  );

  useEffect(() => {
    setActiveIndex(0);
  }, [query]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        if (!isOpen) {
             // Logic to open is handled by parent, but strict checking here prevents ghost triggers
        }
      }
      
      if (!isOpen) return;

      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setActiveIndex(prev => (prev + 1) % filtered.length);
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setActiveIndex(prev => (prev - 1 + filtered.length) % filtered.length);
      }
      if (e.key === 'Enter') {
        e.preventDefault();
        if (filtered[activeIndex]) {
          onNavigate(filtered[activeIndex].id as ViewSection);
          onClose();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, filtered, activeIndex, onNavigate]);

  useEffect(() => {
    // Scroll active item into view
    if (listRef.current && filtered.length > 0) {
      const activeElement = listRef.current.children[activeIndex] as HTMLElement;
      if (activeElement) {
        activeElement.scrollIntoView({ block: 'nearest' });
      }
    }
  }, [activeIndex, filtered]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-slate-950/60 backdrop-blur-sm z-[100]"
          />
          <motion.div 
            initial={{ opacity: 0, scale: 0.95, y: -20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -20 }}
            transition={{ type: "spring", duration: 0.3, bounce: 0 }}
            className="fixed top-[15%] left-1/2 -translate-x-1/2 w-full max-w-xl z-[101] px-4"
          >
            <div className="bg-[#0f1117] border border-slate-700/50 rounded-xl shadow-2xl overflow-hidden ring-1 ring-white/10 flex flex-col max-h-[60vh]">
              {/* Search Header */}
              <div className="p-4 border-b border-slate-800 flex items-center gap-3 bg-slate-900/50">
                <Search size={18} className="text-slate-500" />
                <input 
                  autoFocus
                  type="text" 
                  placeholder="Search documentation..."
                  className="bg-transparent border-none outline-none w-full text-white placeholder:text-slate-500 text-sm font-medium"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <div className="hidden sm:flex items-center gap-1.5 px-2 py-1 rounded-md bg-slate-800 border border-slate-700 text-[10px] font-bold text-slate-400">
                  <span className="text-xs">ESC</span>
                </div>
              </div>
              
              {/* Results List */}
              <div ref={listRef} className="overflow-y-auto p-2 custom-scrollbar">
                {filtered.length > 0 ? (
                  <div className="space-y-1">
                    {filtered.map((item, idx) => (
                      <button
                        key={`${item.id}-${idx}`}
                        onClick={() => {
                          onNavigate(item.id as ViewSection);
                          onClose();
                        }}
                        onMouseEnter={() => setActiveIndex(idx)}
                        className={`
                          w-full flex items-center justify-between p-3 rounded-lg transition-all duration-75 group relative
                          ${activeIndex === idx ? 'bg-indigo-600/10' : 'hover:bg-slate-800/50'}
                        `}
                      >
                         {/* Active Indicator Bar */}
                         {activeIndex === idx && (
                           <motion.div 
                              layoutId="command-active"
                              className="absolute left-0 top-0 bottom-0 w-1 bg-indigo-500 rounded-l-lg"
                           />
                         )}

                        <div className="flex items-center gap-4 pl-2">
                          <div className={`
                            w-6 h-6 rounded flex items-center justify-center transition-colors
                            ${activeIndex === idx ? 'text-indigo-400 bg-indigo-500/20' : 'text-slate-500 bg-slate-800'}
                          `}>
                            {item.icon}
                          </div>
                          <div className="text-left">
                            <div className={`text-sm font-medium transition-colors ${activeIndex === idx ? 'text-white' : 'text-slate-300'}`}>
                              {item.label}
                            </div>
                            <div className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">{item.category}</div>
                          </div>
                        </div>
                        
                        {activeIndex === idx && (
                          <motion.div 
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="flex items-center gap-2 text-slate-400 pr-2"
                          >
                            <span className="text-[10px] uppercase font-bold tracking-widest">Jump to</span>
                            <CornerDownLeft size={14} />
                          </motion.div>
                        )}
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="py-12 flex flex-col items-center justify-center text-slate-500">
                    <Search size={32} className="mb-4 opacity-20" />
                    <p className="text-sm font-medium">No results found for "{query}"</p>
                    <p className="text-xs mt-1 opacity-50">Try searching for "Regression" or "Neural"</p>
                  </div>
                )}
              </div>
              
              {/* Footer */}
              <div className="px-4 py-2 bg-slate-900/80 border-t border-slate-800 flex justify-between items-center text-[10px] text-slate-500 font-medium select-none">
                <div className="flex gap-4">
                   <span className="flex items-center gap-1"><span className="bg-slate-800 px-1 rounded">↑↓</span> to navigate</span>
                   <span className="flex items-center gap-1"><span className="bg-slate-800 px-1 rounded">↵</span> to select</span>
                </div>
                <div className="flex items-center gap-1 opacity-50">
                   <Command size={10} /> + K
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
