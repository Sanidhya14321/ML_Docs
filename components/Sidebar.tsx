
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, ChevronDown, FileText, Terminal, Award, CheckCircle, Layers, Minimize2, Maximize2, LocateFixed } from 'lucide-react';
import { CURRICULUM } from '../data/curriculum';
import { Module, Chapter, Topic } from '../types';
import { useCourseProgress } from '../hooks/useCourseProgress';

interface SidebarProps {
  currentPath: string;
  onNavigate: (path: string) => void;
}

const TopicItem: React.FC<{ topic: Topic; currentPath: string; onNavigate: (path: string) => void }> = ({ topic, currentPath, onNavigate }) => {
  const { isCompleted } = useCourseProgress();
  const isActive = currentPath === topic.id;
  const completed = isCompleted(topic.id);
  
  const getIcon = () => {
    if (completed) return <CheckCircle size={12} className="text-emerald-600 dark:text-emerald-500" />;
    
    // Check if topic.icon is a React component (Lucide icon)
    if (topic.icon) {
        const Icon = topic.icon;
        return <Icon size={12} />;
    }

    switch (topic.type) {
      case 'lab': return <Terminal size={12} />;
      case 'quiz': return <Award size={12} />;
      default: return <FileText size={12} />;
    }
  };

  return (
    <button
      onClick={() => onNavigate(topic.id)}
      data-active={isActive}
      className={`
        w-full flex items-center gap-3 px-3 py-2 rounded-lg text-[11px] font-medium transition-all duration-300 group relative z-10 ml-2 text-left
        ${isActive ? 'text-indigo-700 dark:text-indigo-300' : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'}
      `}
    >
      {/* Active Indicator */}
      {isActive && (
        <motion.div 
          layoutId="sidebar-active"
          className="absolute inset-0 bg-indigo-50 dark:bg-indigo-500/10 border border-indigo-200 dark:border-indigo-500/20 rounded-lg -z-10 transition-colors duration-300"
        />
      )}
      
      <span className={`shrink-0 transition-colors duration-300 ${isActive ? "text-indigo-600 dark:text-indigo-400" : "text-slate-400 dark:text-slate-600 group-hover:text-slate-600 dark:group-hover:text-slate-400"}`}>
        {getIcon()}
      </span>
      <span className="truncate leading-tight">{topic.title}</span>
    </button>
  );
};

const ChapterItem: React.FC<{ 
  chapter: Chapter; 
  currentPath: string; 
  onNavigate: (path: string) => void;
  expandAction: 'expand' | 'collapse' | null;
  expandTimestamp: number;
}> = ({ chapter, currentPath, onNavigate, expandAction, expandTimestamp }) => {
  const hasActiveChild = chapter.topics.some(t => t.id === currentPath);
  const [isOpen, setIsOpen] = useState(hasActiveChild);

  // Sync open state with active path changes (navigating via Next/Prev buttons)
  useEffect(() => {
    if (hasActiveChild) setIsOpen(true);
  }, [currentPath, hasActiveChild]);

  // Handle global expand/collapse actions
  useEffect(() => {
    if (expandAction === 'expand') {
      setIsOpen(true);
    } else if (expandAction === 'collapse') {
      // Don't collapse if it contains the active item
      if (!hasActiveChild) setIsOpen(false);
    }
  }, [expandAction, expandTimestamp, hasActiveChild]);

  return (
    <div className="mb-2">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-2 w-full text-left px-2 py-1.5 text-xs font-bold uppercase tracking-wider transition-colors duration-300 ${hasActiveChild ? 'text-indigo-600 dark:text-indigo-400' : 'text-slate-600 dark:text-slate-500 hover:text-slate-900 dark:hover:text-slate-300'}`}
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
            className="overflow-hidden pl-2 border-l border-slate-200 dark:border-slate-800 ml-3 space-y-1 transition-colors duration-300"
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

const ModuleItem: React.FC<{ 
  module: Module; 
  currentPath: string; 
  onNavigate: (path: string) => void;
  expandAction: 'expand' | 'collapse' | null;
  expandTimestamp: number;
}> = ({ module, currentPath, onNavigate, expandAction, expandTimestamp }) => {
  const Icon = module.icon;
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="mb-8"
    >
      <div className="flex items-center gap-3 px-2 mb-3 text-slate-800 dark:text-slate-200 sticky top-0 bg-white/95 dark:bg-slate-950/95 backdrop-blur-md py-2 z-20 border-b border-transparent transition-colors duration-300">
        <span className="text-indigo-600 dark:text-indigo-500 transition-colors duration-300">{Icon ? <Icon size={16} /> : null}</span>
        <h3 className="text-sm font-serif font-bold tracking-tight">{module.title}</h3>
      </div>
      <div className="space-y-1">
        {module.chapters.map(chapter => (
          <ChapterItem 
            key={chapter.id} 
            chapter={chapter} 
            currentPath={currentPath} 
            onNavigate={onNavigate}
            expandAction={expandAction}
            expandTimestamp={expandTimestamp}
          />
        ))}
      </div>
    </motion.div>
  );
};

export const Sidebar: React.FC<SidebarProps> = ({ currentPath, onNavigate }) => {
  const [expandAction, setExpandAction] = useState<'expand' | 'collapse' | null>(null);
  const [expandTimestamp, setExpandTimestamp] = useState(0);

  const handleExpandAll = () => {
    setExpandAction('expand');
    setExpandTimestamp(Date.now());
  };

  const handleCollapseAll = () => {
    setExpandAction('collapse');
    setExpandTimestamp(Date.now());
  };

  const handleLocateActive = () => {
    const activeEl = document.querySelector('[data-active="true"]');
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Sidebar Controls */}
      <div className="px-4 py-2 flex items-center justify-between border-b border-slate-100 dark:border-slate-800/50 mb-4 transition-colors duration-300">
        <div className="flex items-center gap-1">
          <button 
            onClick={handleExpandAll}
            className="p-1.5 text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-400 hover:bg-slate-100 dark:hover:bg-slate-800 rounded transition-colors"
            title="Expand All"
          >
            <Maximize2 size={14} />
          </button>
          <button 
            onClick={handleCollapseAll}
            className="p-1.5 text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-400 hover:bg-slate-100 dark:hover:bg-slate-800 rounded transition-colors"
            title="Collapse All"
          >
            <Minimize2 size={14} />
          </button>
        </div>
        <button 
          onClick={handleLocateActive}
          className="p-1.5 text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-400 hover:bg-slate-100 dark:hover:bg-slate-800 rounded transition-colors flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider"
          title="Locate Active Topic"
        >
          <LocateFixed size={14} />
          <span>Locate</span>
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto custom-scrollbar px-4 pb-32">
        {CURRICULUM.modules.map(module => (
          <ModuleItem 
            key={module.id} 
            module={module} 
            currentPath={currentPath} 
            onNavigate={onNavigate}
            expandAction={expandAction}
            expandTimestamp={expandTimestamp}
          />
        ))}
      </nav>
    </div>
  );
};

