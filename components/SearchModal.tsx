
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
            className="fixed inset-0 bg-app/90 backdrop-blur-sm z-[100]"
          />
          <motion.div 
            initial={{ opacity: 0, scale: 0.98, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.98, y: 10 }}
            className="fixed top-[15%] left-1/2 -translate-x-1/2 w-full max-w-2xl z-[101] px-4"
            role="dialog"
            aria-modal="true"
            aria-label="Search Topics"
          >
            <div className="bg-surface border border-border-strong rounded-none shadow-2xl overflow-hidden flex flex-col max-h-[70vh] transition-all duration-300">
              <div className="flex items-center gap-4 p-6 border-b border-border-strong bg-app/50">
                <Search size={20} className="text-brand" />
                <input 
                  ref={inputRef}
                  type="text" 
                  placeholder="SEARCH_NEURAL_NODES..."
                  aria-label="Search query"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="flex-1 bg-transparent text-text-primary outline-none placeholder:text-text-muted font-mono font-black text-lg uppercase tracking-widest"
                />
                <div className="hidden sm:flex items-center gap-2">
                   <span className="text-[9px] font-mono font-black bg-surface px-2 py-1 border border-border-strong text-text-muted uppercase tracking-widest">ESC</span>
                </div>
              </div>
              
              <div ref={listRef} className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                 {results.length === 0 ? (
                   <div className="p-12 text-center text-text-muted">
                      <p className="font-mono font-black text-[10px] uppercase tracking-[0.4em]">NO_RECORDS_FOUND_FOR: "{query}"</p>
                   </div>
                 ) : (
                   <div className="space-y-1">
                      {results.map((item, idx) => (
                        <button
                          key={item.id}
                          onClick={() => handleSelect(item.id)}
                          onMouseEnter={() => setSelectedIndex(idx)}
                          className={`w-full text-left px-6 py-4 rounded-none flex items-center justify-between group transition-all ${idx === selectedIndex ? 'bg-brand text-app' : 'text-text-secondary hover:bg-app'}`}
                        >
                           <div className="flex items-center gap-5">
                              <div className={`p-2 rounded-none ${idx === selectedIndex ? 'bg-app/20 text-app' : 'bg-surface border border-border-strong text-text-muted'}`}>
                                 {getIcon(item.type)}
                              </div>
                              <div>
                                 <div className={`text-[11px] font-mono font-black uppercase tracking-widest ${idx === selectedIndex ? 'text-app' : 'text-text-primary'}`}>{item.title}</div>
                                 <div className={`text-[9px] font-mono uppercase tracking-tighter mt-1 ${idx === selectedIndex ? 'text-app/70' : 'text-text-muted'} truncate max-w-[400px]`}>{item.description}</div>
                              </div>
                           </div>
                           {idx === selectedIndex && (
                             <CornerDownLeft size={14} className="text-app/50" />
                           )}
                        </button>
                      ))}
                   </div>
                 )}
              </div>
              
              <div className="p-4 bg-app border-t border-border-strong text-center">
                 <p className="text-[9px] text-text-muted font-mono font-black uppercase tracking-[0.4em]">
                    <span className="text-brand">{results.length}</span> NODES_INDEXED_IN_MEMORY
                 </p>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
