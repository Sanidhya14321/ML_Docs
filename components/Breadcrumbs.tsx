
import React from 'react';
import { Home, ChevronRight } from 'lucide-react';
import { ViewSection, NavigationItem } from '../types';

interface BreadcrumbsProps {
  currentPath: string;
  navItems: NavigationItem[];
  onNavigate: (path: string) => void;
}

export const Breadcrumbs: React.FC<BreadcrumbsProps> = ({ currentPath, navItems, onNavigate }) => {
  // Helper to find path to current item in the navigation tree
  const findPath = (items: NavigationItem[], targetId: string, path: NavigationItem[] = []): NavigationItem[] | null => {
    for (const item of items) {
      if (item.id === targetId) return [...path, item];
      if (item.items) {
        const found = findPath(item.items, targetId, [...path, item]);
        if (found) return found;
      }
    }
    return null;
  };

  const breadcrumbPath = findPath(navItems, currentPath);

  // Hide breadcrumbs on Dashboard or if path is invalid
  if (!breadcrumbPath || currentPath === ViewSection.DASHBOARD) return null;

  return (
    <nav aria-label="Breadcrumb" className="flex items-center gap-1 text-[11px] font-medium text-slate-500 dark:text-slate-400 overflow-x-auto whitespace-nowrap scrollbar-hide px-1">
      <button 
        onClick={() => onNavigate(ViewSection.DASHBOARD)}
        className="hover:text-indigo-500 dark:hover:text-indigo-400 transition-colors flex items-center gap-1 p-1.5 rounded hover:bg-slate-100 dark:hover:bg-slate-800"
        title="Back to Dashboard"
      >
        <Home size={14} />
      </button>
      
      {breadcrumbPath.map((item, idx) => (
        <React.Fragment key={item.id}>
            <ChevronRight size={12} className="text-slate-400 dark:text-slate-600 shrink-0" />
            <button
                onClick={() => onNavigate(item.id)}
                disabled={idx === breadcrumbPath.length - 1}
                className={`
                    px-2 py-1 rounded transition-all truncate max-w-[120px] sm:max-w-[200px] flex items-center
                    ${idx === breadcrumbPath.length - 1 
                        ? 'text-indigo-600 dark:text-indigo-400 font-bold bg-indigo-50 dark:bg-indigo-500/10 cursor-default ring-1 ring-indigo-500/20' 
                        : 'hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800'}
                `}
            >
                {item.label}
            </button>
        </React.Fragment>
      ))}
    </nav>
  );
};
