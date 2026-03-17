
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Moon, Sun, Trash2, ShieldAlert } from 'lucide-react';
import { useTheme } from '../hooks/useTheme';
import { useCourseProgress } from '../hooks/useCourseProgress';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
  const { theme, toggleTheme } = useTheme();
  const { resetProgress } = useCourseProgress();

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-app/80 backdrop-blur-sm z-[100]"
          />
          <motion.div 
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md z-[101] px-4"
            role="dialog"
            aria-modal="true"
            aria-labelledby="settings-title"
          >
            <div className="bg-surface border border-border-strong rounded-none shadow-2xl overflow-hidden transition-all duration-300">
              <div className="p-6 border-b border-border-strong flex items-center justify-between bg-app/50">
                <div className="flex items-center gap-3">
                   <div className="w-2 h-2 bg-brand rounded-full" />
                   <h3 id="settings-title" className="text-[10px] font-mono font-black text-text-primary uppercase tracking-[0.3em]">SYSTEM_CONFIGURATION</h3>
                </div>
                <button onClick={onClose} className="p-2 text-text-muted hover:text-text-primary transition-colors" aria-label="Close Settings">
                  <X size={18} />
                </button>
              </div>

              <div className="p-8 space-y-12">
                {/* Appearance */}
                <section>
                  <h4 className="text-[9px] font-mono font-black text-text-muted uppercase tracking-[0.2em] mb-6">INTERFACE_THEME</h4>
                  <div className="flex items-center justify-between p-6 bg-app border border-border-strong">
                     <div className="flex items-center gap-4">
                        <div className="p-2 bg-brand/10 text-brand">
                           {theme === 'dark' ? <Moon size={18} /> : <Sun size={18} />}
                        </div>
                        <div className="text-[11px] font-mono font-bold text-text-primary uppercase tracking-widest">
                           {theme === 'dark' ? 'DARK_MODE_ACTIVE' : 'LIGHT_MODE_ACTIVE'}
                        </div>
                     </div>
                     <button 
                        onClick={toggleTheme}
                        className={`relative w-12 h-6 rounded-none transition-colors duration-300 ${theme === 'dark' ? 'bg-brand' : 'bg-border-strong'}`}
                        aria-label={theme === 'dark' ? "Switch to Light Mode" : "Switch to Dark Mode"}
                     >
                        <motion.div 
                          layout
                          className="absolute top-1 left-1 w-4 h-4 bg-app rounded-none shadow-md"
                          animate={{ x: theme === 'dark' ? 24 : 0 }}
                        />
                     </button>
                  </div>
                </section>

                {/* Danger Zone */}
                <section>
                  <h4 className="text-[9px] font-mono font-black text-rose-500 uppercase tracking-[0.2em] mb-6 flex items-center gap-2">
                     <ShieldAlert size={12} /> DESTRUCTIVE_OPERATIONS
                  </h4>
                  <div className="p-6 border border-rose-500/20 bg-rose-500/5">
                     <div className="mb-6">
                        <h5 className="text-[11px] font-mono font-black text-text-primary uppercase tracking-widest mb-2">RESET_PROGRESS_DATA</h5>
                        <p className="text-[10px] text-text-secondary uppercase tracking-tight leading-relaxed">
                           This will permanently delete all your progress, completed topics, and local settings. This action cannot be undone.
                        </p>
                     </div>
                     <button 
                        onClick={() => {
                           if(window.confirm('Are you sure you want to reset all progress?')) {
                              resetProgress();
                              onClose();
                           }
                        }}
                        className="w-full py-3 bg-rose-500/10 border border-rose-500/30 text-rose-500 hover:bg-rose-500 hover:text-app text-[10px] font-mono font-black uppercase tracking-[0.2em] transition-all flex items-center justify-center gap-3"
                     >
                        <Trash2 size={14} /> WIPE_ALL_DATA
                     </button>
                  </div>
                </section>
              </div>

              <div className="p-4 bg-app border-t border-border-strong text-center">
                 <p className="text-[8px] text-text-muted font-mono font-black uppercase tracking-[0.4em]">AI_CODEX_CORE_v3.2.0 // BUILD_2026.03</p>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
