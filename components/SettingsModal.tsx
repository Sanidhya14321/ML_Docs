
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
            className="fixed inset-0 bg-slate-950/60 backdrop-blur-sm z-[100]"
          />
          <motion.div 
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md z-[101] px-4"
          >
            <div className="bg-white dark:bg-[#0f1117] border border-slate-200 dark:border-slate-800 rounded-2xl shadow-2xl overflow-hidden">
              <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
                <h3 className="font-bold text-slate-900 dark:text-white">Settings</h3>
                <button onClick={onClose} className="p-2 text-slate-500 hover:text-slate-900 dark:hover:text-white rounded-lg transition-colors">
                  <X size={20} />
                </button>
              </div>

              <div className="p-6 space-y-8">
                {/* Appearance */}
                <section>
                  <h4 className="text-xs font-black text-slate-400 uppercase tracking-widest mb-4">Appearance</h4>
                  <div className="flex items-center justify-between p-4 rounded-xl bg-slate-100 dark:bg-slate-900">
                     <div className="flex items-center gap-3">
                        <div className="p-2 bg-indigo-500/10 text-indigo-500 rounded-lg">
                           {theme === 'dark' ? <Moon size={20} /> : <Sun size={20} />}
                        </div>
                        <div className="text-sm font-medium text-slate-900 dark:text-slate-200">
                           {theme === 'dark' ? 'Dark Mode' : 'Light Mode'}
                        </div>
                     </div>
                     <button 
                        onClick={toggleTheme}
                        className={`relative w-12 h-6 rounded-full transition-colors duration-300 ${theme === 'dark' ? 'bg-indigo-600' : 'bg-slate-300'}`}
                     >
                        <motion.div 
                          layout
                          className="absolute top-1 left-1 w-4 h-4 bg-white rounded-full shadow-md"
                          animate={{ x: theme === 'dark' ? 24 : 0 }}
                        />
                     </button>
                  </div>
                </section>

                {/* Danger Zone */}
                <section>
                  <h4 className="text-xs font-black text-rose-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                     <ShieldAlert size={12} /> Danger Zone
                  </h4>
                  <div className="p-4 rounded-xl border border-rose-200 dark:border-rose-900/30 bg-rose-50 dark:bg-rose-900/10">
                     <div className="mb-4">
                        <h5 className="text-sm font-bold text-slate-900 dark:text-white mb-1">Reset Course Progress</h5>
                        <p className="text-xs text-slate-500 dark:text-slate-400">
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
                        className="w-full py-2.5 bg-white dark:bg-rose-950 border border-rose-200 dark:border-rose-900 text-rose-600 dark:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-900/50 rounded-lg text-xs font-bold transition-colors flex items-center justify-center gap-2"
                     >
                        <Trash2 size={14} /> Reset Everything
                     </button>
                  </div>
                </section>
              </div>

              <div className="p-4 bg-slate-50 dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800 text-center">
                 <p className="text-[10px] text-slate-400 font-mono">AI Codex v3.2.0 Build 2024.10</p>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
