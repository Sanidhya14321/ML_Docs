
import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Hash, CornerDownLeft, Command, FileText, Terminal, Award } from 'lucide-react';
import { getAllTopics } from '../lib/contentHelpers';

interface SearchModalProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigate: (section: string) => void;
}

export const SearchModal: React.FC<SearchModalProps> = ({ isOpen, onClose, onNavigate }) => {
  const [query, setQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);
  
  // Dynamic Search Registry derived from Real Data
  const allTopics = getAllTopics();
  
  const filtered = allTopics.filter(topic => 
    topic.title.toLowerCase().includes(query.toLowerCase()) ||
    topic.description?.toLowerCase().includes(query.toLowerCase())
  );

  const getIcon = (type: string) => {
      switch(type) {
          case 'lab': return <Terminal size={14} />;
          case 'quiz': return <Award size={14} />;
          default: return <FileText size={14} />;
      }
  };

  useEffect(() => {
    setActiveIndex(0);
  }, [query]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
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
          onNavigate(filtered[activeIndex].id);
          onClose();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, filtered, activeIndex, onNavigate]);

  useEffect(() => {
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
              <div className="p-4 border-b border-slate-800 flex items-center gap-3 bg-slate-900/50">
                <Search size={18} className="text-slate-500" />
                <input 
                  autoFocus
                  type="text" 
                  placeholder="Search curriculum..."
                  className="bg-transparent border-none outline-none w-full text-white placeholder:text-slate-500 text-sm font-medium"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <div className="hidden sm:flex items-center gap-1.5 px-2 py-1 rounded-md bg-slate-800 border border-slate-700 text-[10px] font-bold text-slate-400">
                  <span className="text-xs">ESC</span>
                </div>
              </div>
              
              <div ref={listRef} className="overflow-y-auto p-2 custom-scrollbar">
                {filtered.length > 0 ? (
                  <div className="space-y-1">
                    {filtered.map((item, idx) => (
                      <button
                        key={item.id}
                        onClick={() => {
                          onNavigate(item.id);
                          onClose();
                        }}
                        onMouseEnter={() => setActiveIndex(idx)}
                        className={`
                          w-full flex items-center justify-between p-3 rounded-lg transition-all duration-75 group relative
                          ${activeIndex === idx ? 'bg-indigo-600/10' : 'hover:bg-slate-800/50'}
                        `}
                      >
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
                            {getIcon(item.type)}
                          </div>
                          <div className="text-left">
                            <div className={`text-sm font-medium transition-colors ${activeIndex === idx ? 'text-white' : 'text-slate-300'}`}>
                              {item.title}
                            </div>
                            <div className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">{item.type}</div>
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
                  </div>
                )}
              </div>
              
              <div className="px-4 py-2 bg-slate-900/80 border-t border-slate-800 flex justify-between items-center text-[10px] text-slate-500 font-medium select-none">
                <div className="flex gap-4">
                   <span className="flex items-center gap-1"><span className="bg-slate-800 px-1 rounded">↑↓</span> to navigate</span>
                   <span className="flex items-center gap-1"><span className="bg-slate-800 px-1 rounded">↵</span> to select</span>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
