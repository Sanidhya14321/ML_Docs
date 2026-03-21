import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ChevronRight, 
  ChevronDown, 
  FileText, 
  Terminal, 
  Award, 
  CheckCircle, 
  Minimize2, 
  Maximize2, 
  LocateFixed,
  BrainCircuit,
  Hash,
  ListTodo
} from 'lucide-react';
import { CURRICULUM } from '../../../data/curriculum';
import { Module, Chapter, Topic } from '../../../types';
import { useCourseProgress } from '../../../hooks/useCourseProgress';
import { useUIStore } from '../../stores/useUIStore';
import { cn } from '../../lib/utils';
import { Button } from '../ui/Button';

interface SidebarProps {
  currentPath: string;
  onNavigate: (path: string) => void;
}

const TopicItem: React.FC<{ topic: Topic; currentPath: string; onNavigate: (path: string) => void }> = ({ topic, currentPath, onNavigate }) => {
  const { isCompleted } = useCourseProgress();
  const isActive = currentPath === topic.id;
  const completed = isCompleted(topic.id);
  
  const getIcon = () => {
    if (completed) return <CheckCircle size={10} className="text-emerald-500" />;
    if (topic.icon) {
        const Icon = topic.icon;
        return <Icon size={10} />;
    }
    switch (topic.type) {
      case 'lab': return <Terminal size={10} />;
      case 'quiz': return <Award size={10} />;
      default: return <FileText size={10} />;
    }
  };

  return (
    <button
      onClick={() => onNavigate(topic.id)}
      data-active={isActive}
      className={cn(
        'w-full flex items-center gap-3 px-5 py-2.5 text-[10px] font-mono font-bold uppercase tracking-tight transition-all duration-200 group relative border-b border-border-strong/10',
        isActive 
          ? 'text-app bg-text-primary' 
          : 'text-text-secondary hover:text-text-primary hover:bg-surface-active'
      )}
    >
      <span className={cn(
        'shrink-0 transition-colors',
        isActive ? 'text-app' : 'text-text-muted group-hover:text-text-secondary'
      )}>
        {getIcon()}
      </span>
      <span className="truncate leading-tight flex-1">{topic.title}</span>
      {isActive && <div className="w-1 h-3 bg-brand absolute left-0" />}
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

  useEffect(() => {
    if (hasActiveChild) setIsOpen(true);
  }, [currentPath, hasActiveChild]);

  useEffect(() => {
    if (expandAction === 'expand') setIsOpen(true);
    else if (expandAction === 'collapse' && !hasActiveChild) setIsOpen(false);
  }, [expandAction, expandTimestamp, hasActiveChild]);

  return (
    <div className="border-b border-border-strong/20">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          'flex items-center gap-3 w-full text-left px-5 py-4 text-[10px] font-mono font-black uppercase tracking-[0.2em] transition-colors bg-surface/30',
          hasActiveChild ? 'text-brand' : 'text-text-secondary hover:text-text-primary'
        )}
      >
        <span className="opacity-40">{isOpen ? <ChevronDown size={10} /> : <ChevronRight size={10} />}</span>
        <span className="font-heading">{chapter.title}</span>
      </button>
      
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden bg-app/20"
          >
            {chapter.topics.map((topic: Topic) => (
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
  const isActive = currentPath === module.id;
  
  return (
    <div className="mb-0 border-b border-border-strong">
      <button 
        onClick={() => onNavigate(module.id)}
        className={cn(
          "w-full flex items-center gap-4 px-5 py-5 sticky top-0 bg-surface/95 backdrop-blur-md z-20 border-b border-border-strong transition-colors text-left group",
          isActive ? "bg-surface-active" : "hover:bg-surface-hover"
        )}
      >
        <span className={cn(
          "transition-colors",
          isActive ? "text-brand" : "text-brand opacity-60 group-hover:opacity-100"
        )}>
          {Icon ? <Icon size={14} /> : null}
        </span>
        <h3 className={cn(
          "text-[11px] font-mono font-black uppercase tracking-[0.3em] transition-colors",
          isActive ? "text-text-primary" : "text-text-muted group-hover:text-text-primary"
        )}>
          {module.title}
        </h3>
      </button>
      <div className="divide-y divide-border-strong/10">
        {module.chapters.map((chapter: Chapter) => (
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
    </div>
  );
};

export const SidebarContent: React.FC<SidebarProps> = ({ currentPath, onNavigate }) => {
  const [expandAction, setExpandAction] = useState<'expand' | 'collapse' | null>(null);
  const [expandTimestamp, setExpandTimestamp] = useState(0);
  const { isSidebarOpen } = useUIStore();

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
    <aside className={cn(
      'fixed inset-y-0 left-0 z-40 w-72 bg-surface border-r border-border-strong transition-transform duration-normal md:relative md:translate-x-0 flex flex-col',
      !isSidebarOpen && '-translate-x-full'
    )}>
      {/* Brand Header */}
      <div className="p-8 border-b border-border-strong bg-app/30">
          <div className="flex items-center gap-4 mb-3">
            <div className="w-8 h-8 rounded-none bg-text-primary flex items-center justify-center">
              <BrainCircuit size={18} className="text-app" />
            </div>
            <h1 className="font-heading font-black text-lg text-text-primary tracking-tighter uppercase">AI_CODEX</h1>
          </div>
          <p className="text-[9px] text-text-muted font-mono font-black uppercase tracking-[0.4em] opacity-50">NEURAL_ARCH_V3.2</p>
      </div>

      {/* Controls */}
      <div className="px-5 py-3 flex items-center justify-between border-b border-border-strong bg-surface/50">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="xs" onClick={handleExpandAll} className="h-7 w-7 p-0 rounded-none border border-border-strong hover:border-brand">
            <Maximize2 size={12} />
          </Button>
          <Button variant="ghost" size="xs" onClick={handleCollapseAll} className="h-7 w-7 p-0 rounded-none border border-border-strong hover:border-brand">
            <Minimize2 size={12} />
          </Button>
        </div>
        <Button variant="ghost" size="xs" onClick={handleLocateActive} className="h-7 gap-2 text-[9px] font-mono font-black uppercase tracking-widest px-3 rounded-none border border-border-strong hover:border-brand">
          <LocateFixed size={12} />
          <span>SYNC</span>
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto custom-scrollbar divide-y divide-border-strong/20">
        <div className="mb-0 border-b border-border-strong">
          <button 
            onClick={() => onNavigate('tasks')}
            className={cn(
              "w-full flex items-center gap-4 px-5 py-5 sticky top-0 bg-surface/95 backdrop-blur-md z-20 border-b border-border-strong transition-colors text-left group",
              currentPath === 'tasks' ? "bg-surface-active" : "hover:bg-surface-hover"
            )}
          >
            <span className={cn(
              "transition-colors",
              currentPath === 'tasks' ? "text-brand" : "text-brand opacity-60 group-hover:opacity-100"
            )}>
              <ListTodo size={14} />
            </span>
            <h3 className={cn(
              "text-[11px] font-mono font-black uppercase tracking-[0.3em] transition-colors",
              currentPath === 'tasks' ? "text-text-primary" : "text-text-muted group-hover:text-text-primary"
            )}>
              Study Plan
            </h3>
          </button>
        </div>

        {CURRICULUM.modules.map((module: Module) => (
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

      {/* Footer */}
      <div className="p-6 border-t border-border-strong text-[9px] font-mono font-black text-text-muted flex justify-between bg-app/20 uppercase tracking-widest">
          <div className="flex items-center gap-2">
             <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
             <span>SYSTEM_READY</span>
          </div>
          <span>© 2026</span>
      </div>
    </aside>
  );
};
