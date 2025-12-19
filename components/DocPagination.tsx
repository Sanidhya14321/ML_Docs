
import React from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { getNextTopic, getPrevTopic, getTopicById } from '../lib/contentHelpers';

interface DocPaginationProps {
  currentPath: string;
}

export const DocPagination: React.FC<DocPaginationProps> = ({ currentPath }) => {
  const nextId = getNextTopic(currentPath);
  const prevId = getPrevTopic(currentPath);

  const nextTopic = nextId ? getTopicById(nextId) : null;
  const prevTopic = prevId ? getTopicById(prevId) : null;

  if (!prevTopic && !nextTopic) return null;

  return (
    <div className="mt-20 pt-10 border-t border-slate-800 grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Previous Link */}
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
        <div /> /* Spacer */
      )}

      {/* Next Link */}
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
        <div />
      )}
    </div>
  );
};
