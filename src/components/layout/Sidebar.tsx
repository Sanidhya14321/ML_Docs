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
  BrainCircuit
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
    if (completed) return <CheckCircle size={12} className="text-emerald-500" />;
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
      className={cn(
        'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-[11px] font-medium transition-all duration-fast group relative z-10 ml-2 text-left',
        isActive 
          ? 'text-brand bg-brand/5' 
          : 'text-text-secondary hover:text-text-primary hover:bg-surface-hover'
      )}
    >
      {isActive && (
        <motion.div 
          layoutId="sidebar-active"
          className="absolute inset-0 border border-brand/20 rounded-lg -z-10"
        />
      )}
      <span className={cn(
        'shrink-0 transition-colors',
        isActive ? 'text-brand' : 'text-text-muted group-hover:text-text-secondary'
      )}>
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

  useEffect(() => {
    if (hasActiveChild) setIsOpen(true);
  }, [currentPath, hasActiveChild]);

  useEffect(() => {
    if (expandAction === 'expand') setIsOpen(true);
    else if (expandAction === 'collapse' && !hasActiveChild) setIsOpen(false);
  }, [expandAction, expandTimestamp, hasActiveChild]);

  return (
    <div className="mb-2">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          'flex items-center gap-2 w-full text-left px-2 py-1.5 text-xs font-bold uppercase tracking-wider transition-colors',
          hasActiveChild ? 'text-brand' : 'text-text-secondary hover:text-text-primary'
        )}
      >
        <span className="opacity-70">{isOpen ? <ChevronDown size={10} /> : <ChevronRight size={10} />}</span>
        {chapter.title}
      </button>
      
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden pl-2 border-l border-border-subtle ml-3 space-y-1"
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
  return (
    <div className="mb-8">
      <div className="flex items-center gap-3 px-2 mb-3 text-text-primary sticky top-0 bg-surface/95 backdrop-blur-md py-2 z-20 border-b border-transparent">
        <span className="text-brand">{Icon ? <Icon size={16} /> : null}</span>
        <h3 className="text-sm font-serif font-bold tracking-tight">{module.title}</h3>
      </div>
      <div className="space-y-1">
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

export const Sidebar: React.FC = () => {
  const { isSidebarOpen, toggleSidebar } = useUIStore();
  // We need currentPath and onNavigate from App.tsx context or props
  // For now, I'll assume they are passed via a custom hook or we'll wrap this in a way that App.tsx provides them.
  // Actually, I'll use a simplified version that AppShell will manage.
  
  // To keep it functional with the existing App.tsx, I'll need to pass these down.
  // But wait, the AppShell is a wrapper. I'll need to use a store for navigation too if I want it truly decoupled.
  // For now, I'll just make it a component that AppShell renders and we'll pass props.
  
  // Actually, I'll check how App.tsx uses it.
  // App.tsx: <Sidebar currentPath={currentPath} onNavigate={navigateTo} />
  
  // I'll update AppShell to accept these props.
  
  return null; // I'll rewrite this in a moment after updating AppShell
};

// I'll rename the actual sidebar component to SidebarContent and use it in AppShell
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
      'fixed inset-y-0 left-0 z-40 w-72 bg-surface border-r border-border-subtle transition-transform duration-normal md:relative md:translate-x-0',
      !isSidebarOpen && '-translate-x-full'
    )}>
      <div className="h-full flex flex-col">
        {/* Brand Header */}
        <div className="p-6 border-b border-border-subtle">
           <div className="flex items-center gap-3 mb-6">
              <div className="w-8 h-8 rounded-lg bg-brand flex items-center justify-center shadow-lg shadow-brand/20">
                <BrainCircuit size={18} className="text-white" />
              </div>
              <div>
                <h1 className="font-serif font-black text-lg text-text-primary tracking-tighter">AI Codex</h1>
                <p className="text-[9px] text-text-muted font-mono uppercase tracking-[0.3em]">v3.2.0</p>
              </div>
            </div>
        </div>

        {/* Controls */}
        <div className="px-4 py-2 flex items-center justify-between border-b border-border-subtle mb-4">
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="xs" onClick={handleExpandAll} title="Expand All">
              <Maximize2 size={14} />
            </Button>
            <Button variant="ghost" size="xs" onClick={handleCollapseAll} title="Collapse All">
              <Minimize2 size={14} />
            </Button>
          </div>
          <Button variant="ghost" size="xs" onClick={handleLocateActive} className="gap-1.5 text-[10px] font-bold uppercase tracking-wider">
            <LocateFixed size={14} />
            <span>Locate</span>
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto custom-scrollbar px-4 pb-32">
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
        <div className="p-4 border-t border-border-subtle text-[10px] text-text-muted flex justify-between">
            <span>© 2024 AI Codex</span>
        </div>
      </div>
    </aside>
  );
};
