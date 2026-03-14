import React, { useState, useEffect } from 'react';
import { Search as SearchIcon, FileText, Terminal, Award, Command } from 'lucide-react';
import { Modal } from '../ui/Modal';
import { Input } from '../ui/Input';
import { Badge } from '../ui/Badge';
import { CURRICULUM } from '../../../data/curriculum';
import { Topic, Module, Chapter } from '../../../types';
import { cn } from '../../lib/utils';

interface SearchModalProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigate: (id: string) => void;
}

export const SearchModal: React.FC<SearchModalProps> = ({ isOpen, onClose, onNavigate }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Topic[]>([]);

  useEffect(() => {
    if (query.trim().length < 2) {
      setResults([]);
      return;
    }

    const searchResults = CURRICULUM.modules
      .flatMap((m: Module) => m.chapters)
      .flatMap((c: Chapter) => c.topics)
      .filter((t: Topic) => 
        t.title.toLowerCase().includes(query.toLowerCase()) ||
        t.description?.toLowerCase().includes(query.toLowerCase())
      )
      .slice(0, 8);

    setResults(searchResults);
  }, [query]);

  const handleSelect = (id: string) => {
    onNavigate(id);
    onClose();
    setQuery('');
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      className="p-0"
      size="lg"
    >
      <div className="p-4 border-b border-border-subtle">
        <Input
          autoFocus
          placeholder="Search topics, labs, quizzes..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          leftIcon={<SearchIcon size={18} />}
          className="h-12 text-lg border-none focus:ring-0"
        />
      </div>

      <div className="max-h-[60vh] overflow-y-auto p-2 custom-scrollbar">
        {results.length > 0 ? (
          <div className="space-y-1">
            {results.map((topic) => (
              <button
                key={topic.id}
                onClick={() => handleSelect(topic.id)}
                className="w-full flex items-center gap-4 p-3 rounded-xl hover:bg-surface-hover transition-colors text-left group"
              >
                <div className="w-10 h-10 rounded-lg bg-surface-hover border border-border-subtle flex items-center justify-center text-text-muted group-hover:text-brand transition-colors">
                  {topic.type === 'lab' ? <Terminal size={20} /> : topic.type === 'quiz' ? <Award size={20} /> : <FileText size={20} />}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-text-primary truncate">{topic.title}</span>
                    <Badge variant="secondary" size="sm" className="capitalize">{topic.type}</Badge>
                  </div>
                  <p className="text-xs text-text-muted truncate mt-0.5">{topic.description}</p>
                </div>
              </button>
            ))}
          </div>
        ) : query.length >= 2 ? (
          <div className="p-12 text-center">
            <p className="text-text-secondary">No results found for "{query}"</p>
          </div>
        ) : (
          <div className="p-8">
             <h4 className="text-xs font-bold text-text-muted uppercase tracking-wider mb-4">Quick Actions</h4>
             <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <button onClick={() => handleSelect('intro')} className="flex items-center gap-3 p-3 rounded-xl bg-surface-hover border border-border-subtle hover:border-brand/30 transition-all text-left">
                    <FileText size={16} className="text-brand" />
                    <span className="text-sm font-medium">Introduction</span>
                </button>
                <button onClick={() => handleSelect('lab-setup')} className="flex items-center gap-3 p-3 rounded-xl bg-surface-hover border border-border-subtle hover:border-brand/30 transition-all text-left">
                    <Terminal size={16} className="text-brand" />
                    <span className="text-sm font-medium">Lab Setup</span>
                </button>
             </div>
          </div>
        )}
      </div>

      <div className="p-4 bg-surface-hover border-t border-border-subtle flex items-center justify-between text-[10px] text-text-muted">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1"><Command size={10} /> + K to search</span>
          <span className="flex items-center gap-1"><kbd className="px-1 rounded bg-zinc-200 dark:bg-zinc-800 border border-border-subtle">ESC</kbd> to close</span>
        </div>
        <div>
          <span>Search powered by AI Codex</span>
        </div>
      </div>
    </Modal>
  );
};
