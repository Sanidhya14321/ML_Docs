import React from 'react';
import { Menu, Search, Settings, Moon, Sun, Command } from 'lucide-react';
import { useUIStore } from '../../stores/useUIStore';
import { Button } from '../ui/Button';
import { cn } from '../../lib/utils';

export const Navbar: React.FC = () => {
  const { 
    toggleSidebar, 
    toggleSearch, 
    toggleSettings, 
    toggleTheme, 
    theme 
  } = useUIStore();

  return (
    <header className="sticky top-0 z-30 flex items-center justify-between px-4 h-16 bg-surface/80 backdrop-blur-md border-b border-border-subtle transition-colors duration-normal">
      <div className="flex items-center gap-4 flex-1 min-w-0">
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleSidebar}
          className="md:hidden"
          aria-label="Toggle menu"
        >
          <Menu size={20} />
        </Button>
        
        {/* Breadcrumbs or Title could go here */}
        <div className="hidden sm:block">
          <span className="text-sm font-medium text-text-secondary">AI Codex</span>
        </div>
      </div>

      <div className="flex items-center gap-2 pl-4 shrink-0">
        {/* Quick Find Button */}
        <button
          onClick={toggleSearch}
          className="flex items-center gap-2 px-3 h-9 rounded-lg bg-surface-hover border border-border-subtle text-text-secondary hover:text-text-primary hover:border-brand/50 transition-all duration-fast group"
        >
          <Search size={14} className="group-hover:text-brand transition-colors" />
          <span className="text-xs font-bold hidden sm:inline">Quick Find</span>
          <div className="hidden sm:flex items-center gap-0.5 ml-1 px-1.5 py-0.5 rounded bg-zinc-200 dark:bg-zinc-800 border border-border-subtle text-[10px] font-mono">
            <Command size={8} /> K
          </div>
        </button>

        {/* Theme Toggle */}
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleTheme}
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
        </Button>

        {/* Settings */}
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleSettings}
          aria-label="Settings"
        >
          <Settings size={20} />
        </Button>
      </div>
    </header>
  );
};
