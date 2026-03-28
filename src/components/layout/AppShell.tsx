import React from 'react';
import { useUIStore } from '../../stores/useUIStore';
import { SidebarContent } from './Sidebar';
import { Navbar } from './Navbar';
import { cn } from '../../lib/utils';
import { AnimatePresence, motion } from 'framer-motion';

interface AppShellProps {
  children: React.ReactNode;
  currentPath: string;
  onNavigate: (path: string) => void;
}

export const AppShell: React.FC<AppShellProps> = ({ children, currentPath, onNavigate }) => {
  const { isSidebarOpen, setSidebarOpen } = useUIStore();

  return (
    <div className="flex h-screen bg-app text-text-primary overflow-hidden font-sans">
      {/* Mobile Sidebar Overlay */}
      <AnimatePresence>
        {isSidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSidebarOpen(false)}
            className="fixed inset-0 bg-black/50 z-30 md:hidden backdrop-blur-sm"
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <SidebarContent currentPath={currentPath} onNavigate={onNavigate} />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 relative z-10 bg-app">
        <Navbar />

        <main className="flex-1 overflow-y-auto scroll-smooth custom-scrollbar relative">
          {/* Background Grid Pattern */}
          <div className="absolute inset-0 opacity-[0.03] pointer-events-none z-0" 
               style={{ backgroundImage: 'radial-gradient(circle, currentColor 1px, transparent 1px)', backgroundSize: '24px 24px' }} />
          
          <div className="max-w-[1400px] mx-auto p-6 md:p-10 lg:p-16 min-h-full relative z-10">
            <AnimatePresence mode="wait">
              <motion.div
                key={currentPath}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.4, ease: [0.23, 1, 0.32, 1] }}
              >
                {children}
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </div>
  );
};
