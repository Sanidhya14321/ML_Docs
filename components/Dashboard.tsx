
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
    // Simulate data loading for skeleton demonstration
    const timer = setTimeout(() => setIsLoading(false), 800);
    return () => clearTimeout(timer);
  }, []);

  const handleResume = () => {
    if (lastActiveTopic) {
      onNavigate(lastActiveTopic);
    } else {
      // Default to first topic of first module
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
      <header className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-indigo-900 via-slate-900 to-slate-950 border border-slate-800 p-8 md:p-12 shadow-2xl transition-colors duration-300">
        <div className="relative z-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-8">
          <div>
             <div className="flex items-center gap-2 text-indigo-400 mb-2">
                <Trophy size={16} />
                <span className="text-xs font-black uppercase tracking-widest">Course Progress</span>
             </div>
             <motion.h1 
               initial={{ y: 20, opacity: 0 }}
               animate={{ y: 0, opacity: 1 }}
               transition={{ delay: 0.2 }}
               className="text-4xl md:text-5xl font-serif font-bold text-white mb-4"
             >
                {CURRICULUM.title}
             </motion.h1>
             <motion.p 
               initial={{ y: 20, opacity: 0 }}
               animate={{ y: 0, opacity: 1 }}
               transition={{ delay: 0.3 }}
               className="text-slate-400 max-w-xl text-lg font-light leading-relaxed"
             >
                You have completed <strong className="text-white">{overallProgress}%</strong> of the certification. 
                {overallProgress === 100 ? " Ready for deployment." : " Continue building your neural architecture."}
             </motion.p>
          </div>

          <motion.div 
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="bg-slate-950/50 backdrop-blur-md p-6 rounded-2xl border border-slate-800/50 w-full md:w-auto min-w-[280px] transition-colors duration-300"
          >
             <div className="flex justify-between items-end mb-2">
                <span className="text-[10px] font-mono text-slate-500 uppercase">Current Velocity</span>
                <span className="text-2xl font-bold text-white">{overallProgress}%</span>
             </div>
             <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden mb-6 transition-colors duration-300">
                <motion.div 
                   initial={{ width: 0 }}
                   animate={{ width: `${overallProgress}%` }}
                   transition={{ duration: 1, ease: "easeOut", delay: 0.5 }}
                   className="h-full bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)]"
                />
             </div>
             <button 
                onClick={handleResume}
                className="w-full py-3 bg-white text-slate-950 hover:bg-indigo-50 font-bold rounded-lg transition-colors duration-300 flex items-center justify-center gap-2 shadow-lg"
             >
                <Play size={16} fill="currentColor" />
                {lastActiveTopic ? "Resume Learning" : "Start Course"}
             </button>
          </motion.div>
        </div>

        {/* Decorator */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none -translate-y-1/2 translate-x-1/3"></div>
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
        className="grid grid-cols-1 xl:grid-cols-2 gap-8"
      >
        {CURRICULUM.modules.map((module, idx) => {
          const progress = getModuleProgress(module.id);
          const isStarted = progress > 0;
          const isComplete = progress === 100;
          const Icon = module.icon;

          return (
            <motion.div 
              key={module.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="bg-white dark:bg-slate-900/40 border border-slate-200 dark:border-slate-800 rounded-2xl overflow-hidden hover:border-indigo-500/30 transition-all duration-300 group shadow-lg dark:shadow-none"
            >
              <div className="p-6 border-b border-slate-100 dark:border-slate-800/50 bg-slate-50/50 dark:bg-slate-900/50 flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4 transition-colors duration-300">
                 <div className="flex items-center gap-4">
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-lg shrink-0 transition-colors duration-300 ${isComplete ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400' : 'bg-indigo-500/10 text-indigo-600 dark:text-indigo-400'}`}>
                       {isComplete ? <CheckCircle size={20} /> : (Icon ? <Icon size={20} /> : null)}
                    </div>
                    <div>
                       <h3 className="font-bold text-slate-900 dark:text-slate-200 text-lg transition-colors duration-300">{module.title}</h3>
                       <div className="flex items-center gap-2 text-[10px] font-mono text-slate-500 uppercase mt-1">
                          <span>{module.chapters.reduce((acc, c) => acc + c.topics.length, 0)} Topics</span>
                          <span>•</span>
                          <span className={isComplete ? "text-emerald-600 dark:text-emerald-500" : isStarted ? "text-indigo-600 dark:text-indigo-400" : ""}>{progress}% Complete</span>
                       </div>
                    </div>
                 </div>
                 {isComplete && <div className="text-emerald-500/20 hidden sm:block"><Trophy size={48} /></div>}
              </div>

              <div className="p-6 space-y-6">
                 {module.chapters.map(chapter => (
                   <div key={chapter.id}>
                      <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 pl-2 border-l-2 border-slate-200 dark:border-slate-800 transition-colors duration-300">
                        {chapter.title}
                      </h4>
                      <div className="space-y-1">
                         {chapter.topics.map(topic => {
                           const completed = isCompleted(topic.id);
                           return (
                             <button 
                                key={topic.id}
                                onClick={() => onNavigate(topic.id)}
                                className={`w-full text-left p-2 rounded-lg flex items-center gap-3 transition-colors duration-300 ${completed ? 'text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800/50' : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-white'}`}
                             >
                                <div className={`shrink-0 transition-colors duration-300 ${completed ? 'text-emerald-500' : 'text-slate-400 dark:text-slate-700'}`}>
                                   {completed ? <CheckCircle size={14} /> : <Circle size={14} />}
                                </div>
                                <span className={`flex-1 text-sm font-medium truncate ${completed ? 'line-through decoration-slate-300 dark:decoration-slate-700' : ''}`}>
                                   {topic.title}
                                </span>
                                {topic.type === 'lab' && (
                                   <span className="text-[9px] font-black uppercase bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 px-1.5 py-0.5 rounded border border-indigo-500/20">
                                      Lab
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
