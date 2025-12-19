
import React, { useMemo } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { NavigationItem } from '../types';
import { NAV_ITEMS } from '../lib/navigation-data';

interface DocPaginationProps {
  currentPath: string;
}

export const DocPagination: React.FC<DocPaginationProps> = ({ currentPath }) => {
  // Flatten the recursive nav tree into a linear list of navigable leaf nodes
  const flatNav = useMemo(() => {
    const flatten = (items: NavigationItem[]): NavigationItem[] => {
      let result: NavigationItem[] = [];
      items.forEach(item => {
        // Only include items that don't have children (leaf nodes) as navigable pages
        if (!item.items || item.items.length === 0) {
          result.push(item);
        }
        if (item.items) {
          result = [...result, ...flatten(item.items)];
        }
      });
      return result;
    };
    return flatten(NAV_ITEMS);
  }, []);

  const currentIndex = flatNav.findIndex(item => item.id === currentPath);
  const prevItem = currentIndex > 0 ? flatNav[currentIndex - 1] : null;
  const nextItem = currentIndex < flatNav.length - 1 ? flatNav[currentIndex + 1] : null;

  if (!prevItem && !nextItem) return null;

  return (
    <div className="mt-20 pt-10 border-t border-slate-800 grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Previous Link */}
      {prevItem ? (
        <a 
          href={`#/${prevItem.id}`}
          className="group flex flex-col items-start p-6 rounded-2xl border border-slate-800 bg-slate-900/30 hover:bg-slate-900 hover:border-indigo-500/30 transition-all duration-300 relative overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/0 via-indigo-500/0 to-indigo-500/0 group-hover:via-indigo-500/5 group-hover:to-indigo-500/10 transition-all duration-500" />
          <span className="flex items-center gap-2 text-xs font-mono font-bold text-slate-500 uppercase tracking-widest mb-2 group-hover:text-indigo-400">
            <ChevronLeft size={12} /> Previous
          </span>
          <span className="text-lg font-serif font-bold text-slate-200 group-hover:text-white">
            {prevItem.label}
          </span>
        </a>
      ) : (
        <div /> /* Spacer */
      )}

      {/* Next Link */}
      {nextItem ? (
        <a 
          href={`#/${nextItem.id}`}
          className="group flex flex-col items-end text-right p-6 rounded-2xl border border-slate-800 bg-slate-900/30 hover:bg-slate-900 hover:border-indigo-500/30 transition-all duration-300 relative overflow-hidden"
        >
           <div className="absolute inset-0 bg-gradient-to-l from-indigo-500/0 via-indigo-500/0 to-indigo-500/0 group-hover:via-indigo-500/5 group-hover:to-indigo-500/10 transition-all duration-500" />
           <span className="flex items-center gap-2 text-xs font-mono font-bold text-slate-500 uppercase tracking-widest mb-2 group-hover:text-indigo-400">
            Next <ChevronRight size={12} />
          </span>
          <span className="text-lg font-serif font-bold text-slate-200 group-hover:text-white">
            {nextItem.label}
          </span>
        </a>
      ) : (
        <div />
      )}
    </div>
  );
};
