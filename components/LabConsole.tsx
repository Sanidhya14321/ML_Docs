
import React, { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Trash2, Terminal, AlertCircle, CheckCircle, Info } from 'lucide-react';
import { LogEntry } from '../hooks/useCodeRunner';

interface LabConsoleProps {
  logs: LogEntry[];
  onClear: () => void;
  isRunning?: boolean;
}

export const LabConsole: React.FC<LabConsoleProps> = ({ logs, onClear, isRunning }) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs, isRunning]);

  const getLogStyle = (type: string) => {
    switch (type) {
        case 'error': return 'text-rose-400';
        case 'system': return 'text-emerald-400 font-bold';
        default: return 'text-slate-300';
    }
  };

  const getLogIcon = (type: string) => {
    switch (type) {
        case 'error': return <AlertCircle size={12} className="mt-1 flex-shrink-0" />;
        case 'system': return <CheckCircle size={12} className="mt-1 flex-shrink-0" />;
        default: return null;
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#0f1117] font-mono text-xs">
      {/* Console Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-slate-800 bg-[#1e1e1e]">
        <div className="flex items-center gap-2 text-slate-400">
           <Terminal size={14} />
           <span className="font-bold uppercase tracking-wider text-[10px]">Output Terminal</span>
        </div>
        <button 
           onClick={onClear}
           className="p-1.5 text-slate-500 hover:text-white rounded-md hover:bg-slate-700 transition-colors"
           title="Clear Console"
        >
           <Trash2 size={14} />
        </button>
      </div>

      {/* Logs Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2 custom-scrollbar">
         <AnimatePresence initial={false}>
            {logs.length === 0 && !isRunning && (
                <motion.div 
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    className="text-slate-600 italic flex items-center gap-2 mt-2"
                >
                    <Info size={14} /> Ready to execute...
                </motion.div>
            )}

            {logs.map((log) => (
                <motion.div
                    key={log.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className={`flex items-start gap-3 ${getLogStyle(log.type)} leading-relaxed break-words`}
                >
                    <span className="opacity-50 select-none text-[10px] mt-1">
                        {new Date(log.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </span>
                    <div className="flex items-start gap-2">
                        {getLogIcon(log.type)}
                        <span>{log.content}</span>
                    </div>
                </motion.div>
            ))}

            {isRunning && (
                <motion.div 
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    className="flex items-center gap-2 text-indigo-400 mt-2"
                >
                    <span className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse" />
                    Running...
                </motion.div>
            )}
         </AnimatePresence>
         <div ref={bottomRef} />
      </div>
    </div>
  );
};
