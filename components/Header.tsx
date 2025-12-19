
import React from 'react';
import { Menu, Search, Command } from 'lucide-react';
import { Breadcrumbs } from './Breadcrumbs';
import { NavigationItem } from '../types';

interface HeaderProps {
  onMenuClick: () => void;
  onSearchClick: () => void;
  currentPath: string;
  navItems: NavigationItem[];
  onNavigate: (path: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ onMenuClick, onSearchClick, currentPath, navItems, onNavigate }) => {
  return (
    <header className="sticky top-0 z-30 flex items-center justify-between px-4 py-3 bg-[#020617]/80 backdrop-blur-md border-b border-slate-800/50">
       <div className="flex items-center gap-4 flex-1 min-w-0">
          <button 
             onClick={onMenuClick}
             className="md:hidden p-2 -ml-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-colors"
             aria-label="Open Menu"
          >
             <Menu size={20} />
          </button>
          
          <div className="flex-1 min-w-0 overflow-hidden">
             <Breadcrumbs currentPath={currentPath} navItems={navItems} onNavigate={onNavigate} />
          </div>
       </div>

       <div className="flex items-center gap-2 pl-4 shrink-0">
          <button 
             onClick={onSearchClick}
             className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:border-indigo-500/50 transition-all group"
          >
             <Search size={14} className="group-hover:text-indigo-400" />
             <span className="text-xs font-medium hidden sm:inline">Quick Find</span>
             <div className="hidden sm:flex items-center gap-0.5 ml-1 px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px]">
                <Command size={8} /> K
             </div>
          </button>
       </div>
    </header>
  );
};
