
import React from 'react';
import { ChevronLeft, ChevronRight, CheckCircle, Circle, Award, PartyPopper } from 'lucide-react';
import { getNextTopic, getPrevTopic, getTopicById } from '../lib/contentHelpers';
import { useCourseProgress } from '../hooks/useCourseProgress';
import { triggerConfetti } from './Confetti';
import { ViewSection } from '../types';

interface DocPaginationProps {
  currentPath: string;
}

export const DocPagination: React.FC<DocPaginationProps> = ({ currentPath }) => {
  const nextId = getNextTopic(currentPath);
  const prevId = getPrevTopic(currentPath);
  const { markAsCompleted, isCompleted } = useCourseProgress();
  const completed = isCompleted(currentPath);

  const nextTopic = nextId ? getTopicById(nextId) : null;
  const prevTopic = prevId ? getTopicById(prevId) : null;

  const handleMarkComplete = () => {
    markAsCompleted(currentPath);
  };

  const handleFinishCourse = () => {
    markAsCompleted(currentPath);
    triggerConfetti();
    setTimeout(() => {
        window.location.hash = `#/${ViewSection.DASHBOARD}`;
    }, 2500);
  };

  return (
    <div className="mt-20 pt-10 border-t border-slate-800">
      
      {/* Completion Toggle */}
      <div className="flex justify-center mb-12">
        <button 
          onClick={handleMarkComplete}
          disabled={completed}
          className={`
            flex items-center gap-3 px-6 py-3 rounded-full border transition-all duration-300
            ${completed 
               ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400 cursor-default' 
               : 'bg-slate-900 border-slate-700 text-slate-300 hover:border-indigo-500 hover:text-white hover:bg-indigo-500/10'
            }
          `}
        >
           {completed ? <CheckCircle size={20} className="fill-emerald-500/20" /> : <Circle size={20} />}
           <span className="font-bold text-sm tracking-wide">
             {completed ? 'Topic Completed' : 'Mark as Complete'}
           </span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Previous Link - Only show if prevTopic exists */}
        {prevTopic ? (
          <a 
            href={`#/${prevTopic.id}`}
            className="group flex flex-col items-start p-6 rounded-2xl border border-slate-800 bg-slate-900/30 hover:bg-slate-900 hover:border-indigo-500/30 transition-all duration-300 relative overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/0 via-indigo-500/0 to-indigo-500/0 group-hover:via-indigo-500/5 group-hover:to-indigo-500/10 transition-all duration-500" />
            <span className="flex items-center gap-2 text-xs font-mono font-bold text-slate-500 uppercase tracking-widest mb-2 group-hover:text-indigo-400">
              <ChevronLeft size={12} /> Previous
            </span>
            <span className="text-lg font-serif font-bold text-slate-200 group-hover:text-white">
              {prevTopic.title}
            </span>
          </a>
        ) : (
          <div /> /* Spacer if no prev topic */
        )}

        {/* Next Link OR Finish Button */}
        {nextTopic ? (
          <a 
            href={`#/${nextTopic.id}`}
            className="group flex flex-col items-end text-right p-6 rounded-2xl border border-slate-800 bg-slate-900/30 hover:bg-slate-900 hover:border-indigo-500/30 transition-all duration-300 relative overflow-hidden"
          >
             <div className="absolute inset-0 bg-gradient-to-l from-indigo-500/0 via-indigo-500/0 to-indigo-500/0 group-hover:via-indigo-500/5 group-hover:to-indigo-500/10 transition-all duration-500" />
             <span className="flex items-center gap-2 text-xs font-mono font-bold text-slate-500 uppercase tracking-widest mb-2 group-hover:text-indigo-400">
              Next <ChevronRight size={12} />
            </span>
            <span className="text-lg font-serif font-bold text-slate-200 group-hover:text-white">
              {nextTopic.title}
            </span>
          </a>
        ) : (
          <button 
            onClick={handleFinishCourse}
            className="group flex flex-col items-center justify-center text-center p-6 rounded-2xl border border-emerald-500/30 bg-emerald-900/10 hover:bg-emerald-900/20 hover:border-emerald-500/50 transition-all duration-300 relative overflow-hidden"
          >
             <div className="absolute inset-0 bg-gradient-to-t from-emerald-500/10 via-transparent to-transparent opacity-50" />
             <span className="flex items-center gap-2 text-xs font-mono font-bold text-emerald-400 uppercase tracking-widest mb-2">
               <Award size={14} /> Completion
            </span>
            <span className="text-xl font-serif font-bold text-white flex items-center gap-3">
              Finish Course <PartyPopper size={20} className="text-yellow-400" />
            </span>
          </button>
        )}
      </div>
    </div>
  );
};
