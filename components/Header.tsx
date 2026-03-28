
import React from 'react';
import { Menu, Search, Command, Settings, Moon, Sun } from 'lucide-react';
import { Breadcrumbs } from './Breadcrumbs';
import { NavigationItem } from '../types';
import { useTheme } from '../hooks/useTheme';

interface HeaderProps {
  onMenuClick: () => void;
  onSearchClick: () => void;
  onSettingsClick: () => void;
  currentPath: string;
  navItems: NavigationItem[];
  onNavigate: (path: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ onMenuClick, onSearchClick, onSettingsClick, currentPath, navItems, onNavigate }) => {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="sticky top-0 z-30 flex items-center justify-between px-4 py-3 bg-app/80 backdrop-blur-md border-b border-border-strong transition-colors duration-300">
       <div className="flex items-center gap-4 flex-1 min-w-0">
          <button 
             onClick={onMenuClick}
             className="md:hidden p-2 -ml-2 text-text-muted hover:text-brand rounded-none hover:bg-surface-hover transition-colors duration-300"
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
             className="flex items-center gap-2 px-3 py-1.5 rounded-none bg-surface-active border border-border-strong text-text-muted hover:text-brand hover:border-brand transition-all duration-300 group"
             aria-label="Quick Find"
          >
             <Search size={14} className="group-hover:text-brand transition-colors duration-300" />
             <span className="text-xs font-mono font-bold hidden sm:inline uppercase">Quick Find</span>
             <div className="hidden sm:flex items-center gap-0.5 ml-1 px-1.5 py-0.5 rounded-none bg-surface-hover border border-border-subtle text-[10px] transition-colors duration-300">
                <Command size={8} /> K
             </div>
          </button>

          <button
             onClick={toggleTheme}
             className="p-2 text-text-muted hover:text-brand rounded-none hover:bg-surface-hover transition-colors duration-300"
             aria-label={theme === 'dark' ? "Switch to Light Mode" : "Switch to Dark Mode"}
             title={theme === 'dark' ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
             {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
          </button>

          <button
             onClick={onSettingsClick}
             className="p-2 text-text-muted hover:text-brand rounded-none hover:bg-surface-hover transition-colors duration-300"
             aria-label="Settings"
          >
             <Settings size={20} />
          </button>
       </div>
    </header>
  );
};
