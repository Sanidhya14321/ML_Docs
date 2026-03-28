
import React, { useState } from 'react';
import { ChevronLeft, ChevronRight, CheckCircle, Circle, Award, PartyPopper } from 'lucide-react';
import { getNextTopic, getPrevTopic, getTopicById } from '../lib/contentHelpers';
import { useCourseProgress } from '../hooks/useCourseProgress';
import { triggerConfetti } from './Confetti';
import { ViewSection } from '../types';
import { CourseCompletionModal } from './CourseCompletionModal';

interface DocPaginationProps {
  currentPath: string;
}

export const DocPagination: React.FC<DocPaginationProps> = ({ currentPath }) => {
  const nextId = getNextTopic(currentPath);
  const prevId = getPrevTopic(currentPath);
  const { markAsCompleted, isCompleted, resetProgress } = useCourseProgress();
  const completed = isCompleted(currentPath);
  const [showCompletionModal, setShowCompletionModal] = useState(false);

  const nextTopic = nextId ? getTopicById(nextId) : null;
  const prevTopic = prevId ? getTopicById(prevId) : null;

  const handleMarkComplete = () => {
    markAsCompleted(currentPath);
  };

  const handleFinishCourse = () => {
    markAsCompleted(currentPath);
    triggerConfetti();
    // Instead of auto-redirect, show the modal
    setShowCompletionModal(true);
  };

  const handleStartOver = () => {
    if (window.confirm("Are you sure you want to reset all progress?")) {
        resetProgress();
        setShowCompletionModal(false);
        window.location.hash = `#/${ViewSection.DASHBOARD}`;
    }
  };

  const handleViewCertificate = () => {
    setShowCompletionModal(false);
    window.location.hash = `#/${ViewSection.CERTIFICATE}`;
  };

  return (
    <>
      <CourseCompletionModal 
        isOpen={showCompletionModal}
        onClose={() => setShowCompletionModal(false)}
        onStartOver={handleStartOver}
        onViewCertificate={handleViewCertificate}
      />

      <div className="mt-20 pt-10 border-t border-border-strong">
        
        {/* Completion Toggle */}
        <div className="flex justify-center mb-12">
          <button 
            onClick={handleMarkComplete}
            disabled={completed}
            className={`
              flex items-center gap-3 px-6 py-3 rounded-none border transition-all duration-300
              ${completed 
                 ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400 cursor-default' 
                 : 'bg-surface border-border-strong text-text-secondary hover:border-brand hover:text-text-primary hover:bg-brand/10'
              }
            `}
          >
             {completed ? <CheckCircle size={20} className="fill-emerald-500/20" /> : <Circle size={20} />}
             <span className="font-mono font-black text-[10px] uppercase tracking-widest">
               {completed ? 'TOPIC_COMPLETED' : 'MARK_AS_COMPLETE'}
             </span>
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Previous Link - Only show if prevTopic exists */}
          {prevTopic ? (
            <a 
              href={`#/${prevTopic.id}`}
              aria-label={`Previous Topic: ${prevTopic.title}`}
              className="group flex flex-col items-start p-6 rounded-none border border-border-strong bg-surface hover:bg-surface-hover hover:border-brand/50 transition-all duration-300 relative overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-brand/0 via-brand/0 to-brand/0 group-hover:via-brand/5 group-hover:to-brand/10 transition-all duration-500" />
              <span className="flex items-center gap-2 text-[9px] font-mono font-black text-text-muted uppercase tracking-[0.3em] mb-3 group-hover:text-brand">
                <ChevronLeft size={12} /> PREVIOUS_NODE
                <span className="hidden md:inline-block ml-2 px-1.5 py-0.5 rounded-none border border-border-strong text-[9px] text-text-muted group-hover:border-brand/30 group-hover:text-brand/70 transition-colors">K</span>
              </span>
              <span className="text-lg font-heading font-black text-text-secondary group-hover:text-text-primary uppercase tracking-tight">
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
              aria-label={`Next Topic: ${nextTopic.title}`}
              className="group flex flex-col items-end text-right p-6 rounded-none border border-border-strong bg-surface hover:bg-surface-hover hover:border-brand/50 transition-all duration-300 relative overflow-hidden"
            >
               <div className="absolute inset-0 bg-gradient-to-l from-brand/0 via-brand/0 to-brand/0 group-hover:via-brand/5 group-hover:to-brand/10 transition-all duration-500" />
               <span className="flex items-center gap-2 text-[9px] font-mono font-black text-text-muted uppercase tracking-[0.3em] mb-3 group-hover:text-brand">
                <span className="hidden md:inline-block mr-2 px-1.5 py-0.5 rounded-none border border-border-strong text-[9px] text-text-muted group-hover:border-brand/30 group-hover:text-brand/70 transition-colors">J</span>
                NEXT_NODE <ChevronRight size={12} />
              </span>
              <span className="text-lg font-heading font-black text-text-secondary group-hover:text-text-primary uppercase tracking-tight">
                {nextTopic.title}
              </span>
            </a>
          ) : (
            <button 
              onClick={handleFinishCourse}
              className="group flex flex-col items-center justify-center text-center p-6 rounded-none border border-emerald-500/30 bg-emerald-500/5 hover:bg-emerald-500/10 hover:border-emerald-500/50 transition-all duration-300 relative overflow-hidden"
            >
               <div className="absolute inset-0 bg-gradient-to-t from-emerald-500/10 via-transparent to-transparent opacity-50" />
               <span className="flex items-center gap-2 text-[9px] font-mono font-black text-emerald-500 uppercase tracking-[0.3em] mb-3">
                 <Award size={14} /> SYSTEM_MASTERY
              </span>
              <span className="text-xl font-heading font-black text-text-primary uppercase tracking-tight flex items-center gap-3">
                FINISH_COURSE <PartyPopper size={20} className="text-emerald-500" />
              </span>
            </button>
          )}
        </div>
      </div>
    </>
  );
};
