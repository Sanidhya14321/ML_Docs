
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { CURRICULUM } from '../data/curriculum';
import { useCourseProgress } from '../hooks/useCourseProgress';
import { Play, CheckCircle, Circle, Trophy } from 'lucide-react';
import { DashboardSkeleton } from './Skeletons';

interface DashboardProps {
  onNavigate: (path: string) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onNavigate }) => {
  const { getModuleProgress, isCompleted, lastActiveTopic, getOverallProgress } = useCourseProgress();
  const overallProgress = getOverallProgress();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 800);
    return () => clearTimeout(timer);
  }, []);

  const handleResume = () => {
    if (lastActiveTopic) {
      onNavigate(lastActiveTopic);
    } else {
      const firstTopic = CURRICULUM.modules[0]?.chapters[0]?.topics[0]?.id;
      if (firstTopic) onNavigate(firstTopic);
    }
  };

  if (isLoading) {
    return <DashboardSkeleton />;
  }

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="pb-20 space-y-12"
    >
      {/* Hero Section */}
      <header className="relative overflow-hidden rounded-none bg-app border border-border-strong p-8 md:p-12 shadow-sm transition-all duration-300">
        <div className="relative z-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-12">
          <div className="flex-1">
             <div className="flex items-center gap-3 text-brand mb-6">
                <div className="w-2 h-2 bg-brand rounded-full animate-pulse" />
                <span className="text-[10px] font-mono font-black uppercase tracking-[0.3em]">CORE_SYSTEM_STATUS: ACTIVE</span>
             </div>
             <motion.h1 
               initial={{ y: 20, opacity: 0 }}
               animate={{ y: 0, opacity: 1 }}
               transition={{ delay: 0.2 }}
               className="text-4xl md:text-6xl font-display font-black text-text-primary mb-6 uppercase tracking-tight leading-none"
             >
                {CURRICULUM.title}
             </motion.h1>
             <motion.p 
               initial={{ y: 20, opacity: 0 }}
               animate={{ y: 0, opacity: 1 }}
               transition={{ delay: 0.3 }}
               className="text-text-secondary max-w-xl text-lg font-light leading-relaxed"
             >
                Neural synchronization at <strong className="text-text-primary font-bold">{overallProgress}%</strong>. 
                {overallProgress === 100 ? " System fully optimized. Ready for deployment." : " Expanding cognitive architecture through iterative learning cycles."}
             </motion.p>
          </div>

          <motion.div 
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="bg-surface p-8 border border-border-strong w-full md:w-auto min-w-[320px] relative"
          >
             <div className="absolute -top-px -left-px w-4 h-4 border-t border-l border-brand" />
             <div className="absolute -bottom-px -right-px w-4 h-4 border-b border-r border-brand" />
             
             <div className="flex justify-between items-end mb-4">
                <span className="text-[9px] font-mono font-black text-text-muted uppercase tracking-widest">COMPLETION_INDEX</span>
                <span className="text-3xl font-display font-black text-text-primary">{overallProgress}%</span>
             </div>
             <div className="w-full h-1 bg-border-subtle mb-8 relative overflow-hidden">
                <motion.div 
                   initial={{ width: 0 }}
                   animate={{ width: `${overallProgress}%` }}
                   transition={{ duration: 1.5, ease: [0.16, 1, 0.3, 1], delay: 0.5 }}
                   className="h-full bg-brand relative"
                >
                   <div className="absolute top-0 right-0 w-1 h-full bg-white animate-pulse" />
                </motion.div>
             </div>
             <button 
                onClick={handleResume}
                className="w-full py-4 bg-text-primary text-app hover:bg-brand transition-all duration-300 font-mono font-black text-[11px] uppercase tracking-[0.2em] flex items-center justify-center gap-3 group"
             >
                <Play size={14} fill="currentColor" className="group-hover:translate-x-1 transition-transform" />
                {lastActiveTopic ? "RESUME_PROTOCOL" : "INITIALIZE_SEQUENCE"}
             </button>
          </motion.div>
        </div>

        {/* Grid Pattern Overlay */}
        <div className="absolute inset-0 opacity-[0.03] pointer-events-none" 
             style={{ backgroundImage: 'radial-gradient(circle, currentColor 1px, transparent 1px)', backgroundSize: '24px 24px' }} />
      </header>

      {/* Dynamic Modules Grid */}
      <motion.div 
        initial="hidden"
        animate="show"
        variants={{
          hidden: { opacity: 0 },
          show: {
            opacity: 1,
            transition: {
              staggerChildren: 0.1
            }
          }
        }}
        className="grid grid-cols-1 xl:grid-cols-2 gap-px bg-border-strong border border-border-strong"
      >
        {CURRICULUM.modules.map((module, idx) => {
          const progress = getModuleProgress(module.id);
          const isStarted = progress > 0;
          const isComplete = progress === 100;
          const Icon = module.icon;

          return (
            <motion.div 
              key={module.id}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: idx * 0.05 }}
              className="bg-surface group"
            >
              <div className="p-8 border-b border-border-subtle bg-app/20 flex flex-col sm:flex-row sm:justify-between sm:items-center gap-6">
                 <div className="flex items-center gap-5">
                    <div className={`w-12 h-12 rounded-none border flex items-center justify-center transition-all duration-300 ${isComplete ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-500' : 'bg-brand/5 border-brand/20 text-brand'}`}>
                       {isComplete ? <CheckCircle size={20} /> : (Icon ? <Icon size={20} /> : null)}
                    </div>
                    <div>
                       <h3 className="text-xl font-display font-black text-text-primary uppercase tracking-tight group-hover:text-brand transition-colors">{module.title}</h3>
                       <div className="flex items-center gap-3 text-[9px] font-mono font-black text-text-muted uppercase tracking-widest mt-1">
                          <span>{module.chapters.reduce((acc, c) => acc + c.topics.length, 0)} NODES</span>
                          <span className="w-1 h-1 bg-border-strong rounded-full" />
                          <span className={isComplete ? "text-emerald-500" : isStarted ? "text-brand" : ""}>{progress}% SYNCED</span>
                       </div>
                    </div>
                 </div>
                 {isComplete && <div className="text-emerald-500/10 hidden sm:block"><Trophy size={48} /></div>}
              </div>

              <div className="p-8 space-y-10">
                 {module.chapters.map(chapter => (
                   <div key={chapter.id}>
                      <h4 className="text-[9px] font-mono font-black text-text-muted uppercase tracking-[0.3em] mb-4 flex items-center gap-3">
                        <div className="w-4 h-px bg-border-strong" />
                        {chapter.title}
                      </h4>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                         {chapter.topics.map(topic => {
                           const completed = isCompleted(topic.id);
                           return (
                             <button 
                                key={topic.id}
                                onClick={() => onNavigate(topic.id)}
                                className={`group/topic text-left px-4 py-3 border border-transparent hover:border-border-strong hover:bg-app transition-all duration-200 flex items-center gap-3 ${completed ? 'opacity-60' : ''}`}
                             >
                                <div className={`shrink-0 transition-colors duration-300 ${completed ? 'text-emerald-500' : 'text-text-muted group-hover/topic:text-brand'}`}>
                                   {completed ? <CheckCircle size={12} /> : <Circle size={12} />}
                                </div>
                                <span className={`flex-1 text-[11px] font-mono font-bold uppercase tracking-tight truncate ${completed ? 'line-through decoration-emerald-500/30' : 'text-text-secondary group-hover/topic:text-text-primary'}`}>
                                   {topic.title}
                                </span>
                                {topic.type === 'lab' && (
                                   <span className="text-[8px] font-mono font-black uppercase border border-brand/30 text-brand px-1.5 py-0.5">
                                      LAB
                                   </span>
                                )}
                             </button>
                           );
                         })}
                      </div>
                   </div>
                 ))}
              </div>
            </motion.div>
          );
        })}
      </motion.div>
    </motion.div>
  );
};
