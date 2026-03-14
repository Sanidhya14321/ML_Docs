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
  const { isSidebarOpen } = useUIStore();

  return (
    <div className="flex h-screen bg-app text-text-primary overflow-hidden transition-colors duration-normal">
      {/* Sidebar */}
      <SidebarContent currentPath={currentPath} onNavigate={onNavigate} />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 relative z-10">
        <Navbar />

        <main className="flex-1 overflow-y-auto scroll-smooth custom-scrollbar relative">
          <div className="max-w-[1200px] mx-auto p-4 md:p-8 lg:p-12 min-h-full">
            <AnimatePresence mode="wait">
              <motion.div
                key={window.location.hash}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3, ease: 'easeOut' }}
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
