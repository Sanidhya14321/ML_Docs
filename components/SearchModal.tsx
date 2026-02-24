
import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, CornerDownLeft, FileText, Terminal, Award } from 'lucide-react';
import { getAllTopics } from '../lib/contentHelpers';
import { Topic } from '../types';

interface SearchModalProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigate: (section: string) => void;
}

export const SearchModal: React.FC<SearchModalProps> = ({ isOpen, onClose, onNavigate }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Topic[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100);
      setQuery('');
      setResults(getAllTopics());
    }
  }, [isOpen]);

  useEffect(() => {
    if (!query.trim()) {
      setResults(getAllTopics());
      return;
    }
    const all = getAllTopics();
    const filtered = all.filter(t => 
      t.title.toLowerCase().includes(query.toLowerCase()) || 
      (t.description && t.description.toLowerCase().includes(query.toLowerCase()))
    );
    setResults(filtered);
    setSelectedIndex(0);
  }, [query]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => Math.min(prev + 1, results.length - 1));
        if (listRef.current) {
            // Basic scroll into view logic
            const element = listRef.current.children[0]?.children[Math.min(selectedIndex + 1, results.length - 1)] as HTMLElement;
            if(element) element.scrollIntoView({ block: 'nearest' });
        }
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => Math.max(prev - 1, 0));
        if (listRef.current) {
            const element = listRef.current.children[0]?.children[Math.max(selectedIndex - 1, 0)] as HTMLElement;
            if(element) element.scrollIntoView({ block: 'nearest' });
        }
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (results[selectedIndex]) {
          handleSelect(results[selectedIndex].id);
        }
      } else if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, results, selectedIndex, onClose]);

  const handleSelect = (id: string) => {
    onNavigate(id);
    onClose();
  };

  const getIcon = (type: string) => {
    switch (type) {
      case 'lab': return <Terminal size={14} />;
      case 'quiz': return <Award size={14} />;
      default: return <FileText size={14} />;
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-[100]"
          />
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="fixed top-[15%] left-1/2 -translate-x-1/2 w-full max-w-xl z-[101] px-4"
          >
            <div className="bg-white dark:bg-[#0f1117] border border-slate-200 dark:border-slate-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[60vh] transition-colors duration-300">
              <div className="flex items-center gap-3 p-4 border-b border-slate-200 dark:border-slate-800 transition-colors duration-300">
                <Search size={20} className="text-slate-400" />
                <input 
                  ref={inputRef}
                  type="text" 
                  placeholder="Search topics, labs, or concepts..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="flex-1 bg-transparent text-slate-900 dark:text-white outline-none placeholder:text-slate-500 text-lg transition-colors duration-300"
                />
                <div className="hidden sm:flex items-center gap-2">
                   <span className="text-[10px] bg-slate-100 dark:bg-slate-800 px-2 py-1 rounded text-slate-500 border border-slate-200 dark:border-slate-700 transition-colors duration-300">ESC</span>
                </div>
              </div>
              
              <div ref={listRef} className="flex-1 overflow-y-auto p-2 custom-scrollbar">
                 {results.length === 0 ? (
                   <div className="p-8 text-center text-slate-500">
                      <p>No results found for "{query}"</p>
                   </div>
                 ) : (
                   <div className="space-y-1">
                      {results.map((item, idx) => (
                        <button
                          key={item.id}
                          onClick={() => handleSelect(item.id)}
                          onMouseEnter={() => setSelectedIndex(idx)}
                          className={`w-full text-left px-4 py-3 rounded-xl flex items-center justify-between group transition-colors ${idx === selectedIndex ? 'bg-indigo-600 text-white' : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-900'}`}
                        >
                           <div className="flex items-center gap-3">
                              <div className={`p-2 rounded-lg ${idx === selectedIndex ? 'bg-white/20 text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-500'}`}>
                                 {getIcon(item.type)}
                              </div>
                              <div>
                                 <div className={`font-medium ${idx === selectedIndex ? 'text-white' : 'text-slate-900 dark:text-slate-200'}`}>{item.title}</div>
                                 <div className={`text-xs ${idx === selectedIndex ? 'text-indigo-200' : 'text-slate-500'} truncate max-w-[300px]`}>{item.description}</div>
                              </div>
                           </div>
                           {idx === selectedIndex && (
                             <CornerDownLeft size={16} className="text-white/50" />
                           )}
                        </button>
                      ))}
                   </div>
                 )}
              </div>
              
              <div className="p-3 bg-slate-50 dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800 text-center">
                 <p className="text-[10px] text-slate-400 font-mono">
                    <span className="font-bold">{results.length}</span> results found
                 </p>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
