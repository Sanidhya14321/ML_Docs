
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
    <nav aria-label="Breadcrumb" className="flex items-center gap-1 text-[11px] font-mono font-medium text-text-muted overflow-x-auto whitespace-nowrap scrollbar-hide px-1">
      <button 
        onClick={() => onNavigate(ViewSection.DASHBOARD)}
        className="hover:text-brand transition-colors flex items-center gap-1 p-1.5 rounded-none hover:bg-surface-hover"
        title="Back to Dashboard"
      >
        <Home size={14} />
      </button>
      
      {breadcrumbPath.map((item, idx) => (
        <React.Fragment key={item.id}>
            <ChevronRight size={12} className="text-text-secondary shrink-0" />
            <button
                onClick={() => onNavigate(item.id)}
                disabled={idx === breadcrumbPath.length - 1}
                className={`
                    px-2 py-1 rounded-none transition-all truncate max-w-[120px] sm:max-w-[200px] flex items-center uppercase tracking-wider
                    ${idx === breadcrumbPath.length - 1 
                        ? 'text-brand font-black bg-brand/10 cursor-default ring-1 ring-brand/20' 
                        : 'hover:text-text-primary hover:bg-surface-hover'}
                `}
            >
                {item.label}
            </button>
        </React.Fragment>
      ))}
    </nav>
  );
};
