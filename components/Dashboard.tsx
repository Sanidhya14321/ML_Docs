
import React from 'react';
import { motion } from 'framer-motion';
import { CURRICULUM } from '../data/curriculum';
import { useCourseProgress } from '../hooks/useCourseProgress';
import { Play, CheckCircle, Circle, Trophy } from 'lucide-react';

interface DashboardProps {
  onNavigate: (path: string) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onNavigate }) => {
  const { getModuleProgress, isCompleted, lastActiveTopic, getOverallProgress } = useCourseProgress();
  const overallProgress = getOverallProgress();

  const handleResume = () => {
    if (lastActiveTopic) {
      onNavigate(lastActiveTopic);
    } else {
      // Default to first topic of first module
      const firstTopic = CURRICULUM[0]?.chapters[0]?.topics[0]?.id;
      if (firstTopic) onNavigate(firstTopic);
    }
  };

  return (
    <div className="pb-20 animate-fade-in space-y-12">
      {/* Hero Section */}
      <header className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-indigo-900 via-slate-900 to-slate-950 border border-slate-800 p-8 md:p-12 shadow-2xl">
        <div className="relative z-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-8">
          <div>
             <div className="flex items-center gap-2 text-indigo-400 mb-2">
                <Trophy size={16} />
                <span className="text-xs font-black uppercase tracking-widest">Course Progress</span>
             </div>
             <h1 className="text-4xl md:text-5xl font-serif font-bold text-white mb-4">
                Full Stack AI Engineer
             </h1>
             <p className="text-slate-400 max-w-xl text-lg font-light leading-relaxed">
                You have completed <strong className="text-white">{overallProgress}%</strong> of the certification. 
                {overallProgress === 100 ? " Ready for deployment." : " Continue building your neural architecture."}
             </p>
          </div>

          <div className="bg-slate-950/50 backdrop-blur-md p-6 rounded-2xl border border-slate-800/50 w-full md:w-auto min-w-[280px]">
             <div className="flex justify-between items-end mb-2">
                <span className="text-[10px] font-mono text-slate-500 uppercase">Current Velocity</span>
                <span className="text-2xl font-bold text-white">{overallProgress}%</span>
             </div>
             <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden mb-6">
                <motion.div 
                   initial={{ width: 0 }}
                   animate={{ width: `${overallProgress}%` }}
                   transition={{ duration: 1, ease: "easeOut" }}
                   className="h-full bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)]"
                />
             </div>
             <button 
                onClick={handleResume}
                className="w-full py-3 bg-white text-slate-950 hover:bg-indigo-50 font-bold rounded-lg transition-colors flex items-center justify-center gap-2 shadow-lg"
             >
                <Play size={16} fill="currentColor" />
                {lastActiveTopic ? "Resume Learning" : "Start Course"}
             </button>
          </div>
        </div>

        {/* Decorator */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none -translate-y-1/2 translate-x-1/3"></div>
      </header>

      {/* Dynamic Modules Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {CURRICULUM.map((module, idx) => {
          const progress = getModuleProgress(module.id);
          const isStarted = progress > 0;
          const isComplete = progress === 100;

          return (
            <motion.div 
              key={module.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="bg-slate-900/40 border border-slate-800 rounded-2xl overflow-hidden hover:border-indigo-500/30 transition-all group"
            >
              <div className="p-6 border-b border-slate-800/50 bg-slate-900/50 flex justify-between items-center">
                 <div className="flex items-center gap-4">
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-lg ${isComplete ? 'bg-emerald-500/10 text-emerald-400' : 'bg-indigo-500/10 text-indigo-400'}`}>
                       {isComplete ? <CheckCircle size={20} /> : module.icon}
                    </div>
                    <div>
                       <h3 className="font-bold text-slate-200 text-lg">{module.title}</h3>
                       <div className="flex items-center gap-2 text-[10px] font-mono text-slate-500 uppercase mt-1">
                          <span>{module.chapters.reduce((acc, c) => acc + c.topics.length, 0)} Topics</span>
                          <span>•</span>
                          <span className={isComplete ? "text-emerald-500" : isStarted ? "text-indigo-400" : ""}>{progress}% Complete</span>
                       </div>
                    </div>
                 </div>
                 {isComplete && <div className="text-emerald-500/20"><Trophy size={48} /></div>}
              </div>

              <div className="p-6 space-y-6">
                 {module.chapters.map(chapter => (
                   <div key={chapter.id}>
                      <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 pl-2 border-l-2 border-slate-800">
                        {chapter.title}
                      </h4>
                      <div className="space-y-1">
                         {chapter.topics.map(topic => {
                           const completed = isCompleted(topic.id);
                           return (
                             <button 
                                key={topic.id}
                                onClick={() => onNavigate(topic.id)}
                                className={`w-full text-left p-2 rounded-lg flex items-center gap-3 transition-colors ${completed ? 'text-slate-500 hover:bg-slate-800/50' : 'text-slate-300 hover:bg-slate-800 hover:text-white'}`}
                             >
                                <div className={`shrink-0 transition-colors ${completed ? 'text-emerald-500' : 'text-slate-700'}`}>
                                   {completed ? <CheckCircle size={14} /> : <Circle size={14} />}
                                </div>
                                <span className={`flex-1 text-sm font-medium truncate ${completed ? 'line-through decoration-slate-700' : ''}`}>
                                   {topic.title}
                                </span>
                                {topic.type === 'lab' && (
                                   <span className="text-[9px] font-black uppercase bg-indigo-500/10 text-indigo-400 px-1.5 py-0.5 rounded border border-indigo-500/20">
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
      </div>
    </div>
  );
};
