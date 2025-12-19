
import React from 'react';
import { Home } from 'lucide-react';
import { ViewSection, NavigationItem } from '../types';

interface BreadcrumbsProps {
  currentPath: string;
  navItems: NavigationItem[];
  onNavigate: (path: string) => void;
}

export const Breadcrumbs: React.FC<BreadcrumbsProps> = ({ currentPath, navItems, onNavigate }) => {
  // Helper to find path to current item
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

  if (!breadcrumbPath) return null;

  return (
    <nav className="flex items-center gap-2 mb-8 text-[11px] font-mono font-medium text-slate-500 uppercase tracking-wide overflow-x-auto whitespace-nowrap scrollbar-hide">
      <button 
        onClick={() => onNavigate(ViewSection.FOUNDATIONS)}
        className="hover:text-indigo-400 transition-colors flex items-center gap-1.5"
      >
        <Home size={12} />
        <span className="hidden sm:inline">Home</span>
      </button>
      
      {breadcrumbPath.map((item, idx) => (
        <React.Fragment key={item.id}>
            <span className="text-slate-700 select-none">/</span>
            <button
                onClick={() => onNavigate(item.id)}
                disabled={idx === breadcrumbPath.length - 1}
                className={`
                    transition-colors truncate max-w-[150px]
                    ${idx === breadcrumbPath.length - 1 ? 'text-indigo-400 font-bold cursor-default' : 'hover:text-slate-300'}
                `}
            >
                {item.label}
            </button>
        </React.Fragment>
      ))}
    </nav>
  );
};
