import React from 'react';
import { motion } from 'framer-motion';
import { Module, Chapter, Topic } from '../../types';
import { useCourseProgress } from '../../hooks/useCourseProgress';
import { CheckCircle, Circle, PlayCircle, FileText, Terminal, Award } from 'lucide-react';
import { cn } from '../lib/utils';

interface ModuleOverviewViewProps {
  module: Module;
  onNavigate: (path: string) => void;
}

export const ModuleOverviewView: React.FC<ModuleOverviewViewProps> = ({ module, onNavigate }) => {
  const { isCompleted } = useCourseProgress();

  const getTopicIcon = (topic: Topic) => {
    if (topic.icon) {
      const Icon = topic.icon;
      return <Icon size={16} />;
    }
    switch (topic.type) {
      case 'lab': return <Terminal size={16} />;
      case 'quiz': return <Award size={16} />;
      default: return <FileText size={16} />;
    }
  };

  const calculateModuleProgress = () => {
    let totalTopics = 0;
    let completedTopics = 0;

    module.chapters.forEach(chapter => {
      chapter.topics.forEach(topic => {
        totalTopics++;
        if (isCompleted(topic.id)) {
          completedTopics++;
        }
      });
    });

    return totalTopics === 0 ? 0 : Math.round((completedTopics / totalTopics) * 100);
  };

  const progress = calculateModuleProgress();
  const ModuleIcon = module.icon;

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-5xl mx-auto px-6 py-12 md:py-20"
    >
      <div className="mb-12">
        <div className="flex items-center gap-4 mb-6">
          {ModuleIcon && (
            <div className="w-16 h-16 rounded-none border border-brand/30 bg-brand/5 flex items-center justify-center text-brand">
              <ModuleIcon size={32} />
            </div>
          )}
          <div>
            <h1 className="text-4xl md:text-5xl font-heading font-black text-text-primary uppercase tracking-tight">
              {module.title}
            </h1>
            <p className="text-text-secondary mt-2 text-lg font-light">
              {module.description || 'Module Overview'}
            </p>
          </div>
        </div>

        <div className="bg-surface border border-border-strong rounded-none p-6 flex items-center gap-6 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-16 h-16 bg-brand/5 rotate-45 translate-x-8 -translate-y-8" />
          <div className="flex-1 relative z-10">
            <div className="flex justify-between items-end mb-2">
              <span className="text-[10px] font-mono font-black text-text-muted uppercase tracking-widest">MODULE_SYNC_PROGRESS</span>
              <span className="text-2xl font-heading font-black text-brand">{progress}%</span>
            </div>
            <div className="h-1 bg-border-subtle rounded-none overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 1, ease: "easeOut" }}
                className="h-full bg-brand relative"
              >
                <div className="absolute top-0 right-0 w-1 h-full bg-white animate-pulse" />
              </motion.div>
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-8">
        {module.chapters.map((chapter, chapterIdx) => {
          const chapterTopics = chapter.topics;
          const completedInChapter = chapterTopics.filter(t => isCompleted(t.id)).length;
          const chapterProgress = chapterTopics.length === 0 ? 0 : Math.round((completedInChapter / chapterTopics.length) * 100);

          return (
            <motion.div 
              key={chapter.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: chapterIdx * 0.1 }}
              className="bg-surface border border-border-strong rounded-none overflow-hidden group"
            >
              <div className="p-6 border-b border-border-strong bg-app/30 flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h2 className="text-xl font-heading font-black text-text-primary uppercase tracking-tight">
                    {chapter.title}
                  </h2>
                  <p className="text-[10px] text-text-muted font-mono mt-1 uppercase tracking-widest">
                    {completedInChapter} / {chapterTopics.length} NODES_SYNCED
                  </p>
                </div>
                <div className="w-full md:w-48 h-1 bg-border-subtle rounded-none overflow-hidden shrink-0">
                  <div 
                    className="h-full bg-brand/60"
                    style={{ width: `${chapterProgress}%` }}
                  />
                </div>
              </div>

              <div className="divide-y divide-border-strong/50">
                {chapter.topics.map((topic) => {
                  const completed = isCompleted(topic.id);
                  return (
                    <button
                      key={topic.id}
                      onClick={() => onNavigate(topic.id)}
                      className="w-full text-left p-4 hover:bg-surface-active transition-colors flex items-center gap-4 group/topic"
                    >
                      <div className={cn(
                        "shrink-0 transition-colors",
                        completed ? "text-emerald-500" : "text-text-muted group-hover/topic:text-brand"
                      )}>
                        {completed ? <CheckCircle size={16} /> : <Circle size={16} />}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3">
                          <span className={cn(
                            "shrink-0 transition-colors",
                            completed ? "text-emerald-500/50" : "text-text-muted group-hover/topic:text-brand/50"
                          )}>
                            {getTopicIcon(topic)}
                          </span>
                          <h3 className={cn(
                            "font-mono font-bold text-sm uppercase tracking-tight truncate transition-colors",
                            completed ? "text-text-secondary line-through decoration-emerald-500/30" : "text-text-primary group-hover/topic:text-brand"
                          )}>
                            {topic.title}
                          </h3>
                          {topic.type === 'lab' && (
                            <span className="text-[8px] font-mono font-black uppercase border border-brand/30 text-brand px-1.5 py-0.5 ml-2">
                              LAB
                            </span>
                          )}
                        </div>
                        {topic.description && (
                          <p className="text-xs text-text-muted mt-1.5 truncate ml-7 font-sans">
                            {topic.description}
                          </p>
                        )}
                      </div>

                      <div className="shrink-0 opacity-0 group-hover/topic:opacity-100 transition-opacity text-brand">
                        <PlayCircle size={16} />
                      </div>
                    </button>
                  );
                })}
              </div>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
};
