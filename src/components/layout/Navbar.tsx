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
    <header className="sticky top-0 z-30 flex items-center justify-between px-6 h-14 bg-surface/80 backdrop-blur-md border-b border-border-strong transition-colors duration-normal">
      <div className="flex items-center gap-4 flex-1 min-w-0">
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleSidebar}
          className="md:hidden rounded-none"
          aria-label="Toggle menu"
        >
          <Menu size={18} />
        </Button>
        
        <div className="hidden sm:flex items-center gap-3">
          <div className="w-1 h-4 bg-brand" />
          <span className="text-[10px] font-mono font-black text-text-primary uppercase tracking-[0.2em]">
            SYSTEM_INTERFACE_V2.5
          </span>
        </div>
      </div>

      <div className="flex items-center gap-4 pl-4 shrink-0">
        {/* Quick Find Button */}
        <button
          onClick={toggleSearch}
          className="flex items-center gap-3 px-4 h-8 rounded-none bg-app border border-border-strong text-text-secondary hover:text-text-primary hover:border-brand transition-all duration-fast group"
        >
          <Search size={12} className="group-hover:text-brand transition-colors" />
          <span className="text-[10px] font-mono font-black uppercase tracking-widest hidden sm:inline">SEARCH_INDEX</span>
          <div className="hidden sm:flex items-center gap-1 ml-2 px-1.5 py-0.5 border border-border-strong bg-surface text-[9px] font-mono text-text-muted">
            <Command size={8} /> K
          </div>
        </button>

        <div className="h-4 w-px bg-border-strong hidden sm:block" />

        <div className="flex items-center gap-1">
          {/* Theme Toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleTheme}
            className="rounded-none hover:bg-brand/10 hover:text-brand"
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
          </Button>

          {/* Settings */}
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleSettings}
            className="rounded-none hover:bg-brand/10 hover:text-brand"
            aria-label="Settings"
          >
            <Settings size={18} />
          </Button>
        </div>
      </div>
    </header>
  );
};
