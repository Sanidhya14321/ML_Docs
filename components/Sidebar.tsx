
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, ChevronDown, Circle, FileText, Terminal, FolderOpen } from 'lucide-react';
import { CURRICULUM } from '../data/curriculum';
import { Module, Chapter, Topic } from '../types';

interface SidebarProps {
  currentPath: string;
  onNavigate: (path: string) => void;
}

const TopicItem: React.FC<{ topic: Topic; currentPath: string; onNavigate: (path: string) => void }> = ({ topic, currentPath, onNavigate }) => {
  const isActive = currentPath === topic.id;
  
  return (
    <button
      onClick={() => onNavigate(topic.id)}
      className={`
        w-full flex items-center gap-3 px-3 py-2 rounded-lg text-[11px] font-medium transition-all duration-200 group relative z-10 ml-2
        ${isActive ? 'text-indigo-300' : 'text-slate-400 hover:text-slate-200'}
      `}
    >
      {/* Active Indicator */}
      {isActive && (
        <motion.div 
          layoutId="sidebar-active"
          className="absolute inset-0 bg-indigo-500/10 border border-indigo-500/20 rounded-lg -z-10 shadow-[0_0_15px_rgba(99,102,241,0.1)]"
        />
      )}
      
      <span className={`${isActive ? "text-indigo-400" : "text-slate-600 group-hover:text-slate-400"}`}>
        {topic.type === 'lab' ? <Terminal size={12} /> : <FileText size={12} />}
      </span>
      <span className="truncate">{topic.title}</span>
    </button>
  );
};

const ChapterItem: React.FC<{ chapter: Chapter; currentPath: string; onNavigate: (path: string) => void }> = ({ chapter, currentPath, onNavigate }) => {
  const hasActiveChild = chapter.topics.some(t => t.id === currentPath);
  const [isOpen, setIsOpen] = useState(hasActiveChild);

  useEffect(() => {
    if (hasActiveChild) setIsOpen(true);
  }, [currentPath]);

  return (
    <div className="mb-2">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-2 w-full text-left px-2 py-1.5 text-xs font-bold uppercase tracking-wider transition-colors ${hasActiveChild ? 'text-indigo-400' : 'text-slate-500 hover:text-slate-300'}`}
      >
        <span className="opacity-70">{isOpen ? <ChevronDown size={10} /> : <ChevronRight size={10} />}</span>
        {chapter.title}
      </button>
      
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden pl-2 border-l border-slate-800 ml-3 space-y-1"
          >
            {chapter.topics.map(topic => (
              <TopicItem key={topic.id} topic={topic} currentPath={currentPath} onNavigate={onNavigate} />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const ModuleItem: React.FC<{ module: Module; currentPath: string; onNavigate: (path: string) => void }> = ({ module, currentPath, onNavigate }) => {
  return (
    <div className="mb-8">
      <div className="flex items-center gap-3 px-2 mb-3 text-slate-200">
        <span className="text-indigo-500">{module.icon}</span>
        <h3 className="text-sm font-serif font-bold tracking-tight">{module.title}</h3>
      </div>
      <div className="space-y-1">
        {module.chapters.map(chapter => (
          <ChapterItem key={chapter.id} chapter={chapter} currentPath={currentPath} onNavigate={onNavigate} />
        ))}
      </div>
    </div>
  );
};

export const Sidebar: React.FC<SidebarProps> = ({ currentPath, onNavigate }) => {
  return (
    <nav className="pb-24 px-4">
      {CURRICULUM.map(module => (
        <ModuleItem key={module.id} module={module} currentPath={currentPath} onNavigate={onNavigate} />
      ))}
    </nav>
  );
};
