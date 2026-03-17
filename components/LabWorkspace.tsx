
import React, { useState, useEffect } from 'react';
import { DocViewer } from './DocViewer';
import { ResizableLayout } from './ResizableLayout';
import { CodeEditor } from './CodeEditor';
import { LabConsole } from './LabConsole';
import { useCodeRunner } from '../hooks/useCodeRunner';
import { getTopicById } from '../lib/contentHelpers';
import { useCourseProgress } from '../hooks/useCourseProgress';
import { Play, RotateCcw, Terminal, FileCode, Loader2, Lightbulb, CheckCircle, Server } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { triggerConfetti } from './Confetti';
import { LoadingOverlay } from './LoadingOverlay';

interface LabWorkspaceProps {
  topicId: string;
  onBack: () => void;
}

const DEFAULT_CODE = `# No specific lab configuration found.\nprint("Hello World")`;

export const LabWorkspace: React.FC<LabWorkspaceProps> = ({ topicId, onBack }) => {
  const topic = getTopicById(topicId);
  const initialCode = topic?.labConfig?.initialCode || DEFAULT_CODE;

  const { code, setCode, resetCode, logs, isRunning, runCode, clearLogs } = useCodeRunner(topicId, initialCode);

  const [activeTab, setActiveTab] = useState<'editor' | 'console'>('editor');
  const [isMobile, setIsMobile] = useState(false);
  const [isBooting, setIsBooting] = useState(true);
  
  const { markAsCompleted, isCompleted } = useCourseProgress();
  const completed = isCompleted(topicId);

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  useEffect(() => {
    setIsBooting(true);
    const timer = setTimeout(() => {
        setIsBooting(false);
    }, 2000);
    return () => clearTimeout(timer);
  }, [topicId]);

  useEffect(() => {
    if (!isRunning && logs.length > 0) {
       const lastLog = logs[logs.length - 1];
       if (lastLog.type === 'system' && lastLog.content.includes('Execution completed')) {
           if (!isCompleted(topicId)) {
               markAsCompleted(topicId);
               triggerConfetti();
           }
       }
    }
  }, [isRunning, logs, topicId, markAsCompleted, isCompleted]);

  const handleRun = () => {
    setActiveTab('console');
    runCode();
  };

  const handleReset = () => {
    if (window.confirm("Reset code to default? This will lose your changes.")) {
      resetCode();
    }
  };

  if (isBooting) {
      return (
          <div className="h-full flex items-center justify-center bg-app text-text-secondary">
              <LoadingOverlay message="PROVISIONING_ENVIRONMENT" subMessage="ALLOCATING_GPU_CONTAINER_INSTANCE..." />
          </div>
      );
  }

  return (
    <div className="h-full flex flex-col bg-app relative z-50">
      {/* Lab Header */}
      <header className="h-14 border-b border-border-strong bg-surface flex items-center justify-between px-6 shrink-0">
        <div className="flex items-center gap-6">
           <button onClick={onBack} className="text-[10px] font-mono font-black text-text-muted hover:text-brand uppercase tracking-widest transition-colors">← EXIT_SESSION</button>
           <div className="h-4 w-px bg-border-strong hidden sm:block" />
           <div className="flex items-center gap-3">
              <Terminal size={14} className="text-brand" />
              <span className="text-[10px] font-mono font-black text-text-primary uppercase tracking-[0.2em] hidden sm:inline">LAB_WORKSPACE_V1.0</span>
           </div>
           <span className="text-[10px] text-text-muted font-mono hidden md:inline truncate max-w-[200px] uppercase tracking-tighter opacity-50">[{topic?.title || topicId}]</span>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="hidden lg:flex items-center gap-2 text-[9px] font-mono font-black text-emerald-500 bg-emerald-500/5 px-3 py-1 border border-emerald-500/20 uppercase tracking-widest">
             <Server size={10} /> SYSTEM_ONLINE
          </div>

          <AnimatePresence>
            {completed && (
                <motion.div 
                    initial={{ opacity: 0, x: 20 }} 
                    animate={{ opacity: 1, x: 0 }}
                    className="hidden sm:flex items-center gap-2 text-emerald-500 text-[9px] font-mono font-black uppercase tracking-widest px-3 py-1 bg-emerald-500/5 border border-emerald-500/20"
                >
                    <CheckCircle size={10} /> SYNC_COMPLETE
                </motion.div>
            )}
          </AnimatePresence>

          <div className="h-4 w-px bg-border-strong hidden sm:block" />

          <div className="flex items-center gap-2">
            {topic?.labConfig?.hints && topic.labConfig.hints.length > 0 && (
               <button className="flex items-center gap-2 px-3 py-1.5 text-[10px] font-mono font-black text-amber-500 bg-amber-500/5 border border-amber-500/20 hover:bg-amber-500/10 uppercase tracking-widest transition-colors">
                  <Lightbulb size={12} /> <span className="hidden sm:inline">HINTS</span>
               </button>
            )}
            <button 
              onClick={handleReset}
              disabled={isRunning}
              className="p-2 text-text-muted hover:text-text-primary transition-colors disabled:opacity-50"
              title="Reset Code"
            >
              <RotateCcw size={16} />
            </button>
            <button 
              onClick={handleRun}
              disabled={isRunning}
              className="flex items-center gap-3 px-6 py-1.5 bg-text-primary text-app hover:bg-brand disabled:opacity-50 transition-all font-mono font-black text-[10px] uppercase tracking-[0.2em]"
            >
              {isRunning ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} fill="currentColor" />}
              {isRunning ? 'EXECUTING...' : 'RUN_CODE'}
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <div className="flex-1 overflow-hidden">
        <ResizableLayout
          isMobile={isMobile}
          initialLeftWidth={35}
          left={
            <div className="bg-app min-h-full p-8 pb-24">
               <DocViewer topicId={topicId} title="Lab Instructions" isCompact={true} />
            </div>
          }
          right={
            <div className="flex flex-col h-full bg-[#1e1e1e]">
              {/* Editor Tabs */}
              <div className="flex bg-[#1a1a1a] border-b border-white/5 shrink-0">
                <button 
                   onClick={() => setActiveTab('editor')}
                   className={`px-6 py-3 text-[10px] font-mono font-black uppercase tracking-widest flex items-center gap-3 border-b-2 transition-all ${activeTab === 'editor' ? 'bg-white/5 text-brand border-brand' : 'text-white/40 border-transparent hover:text-white/60'}`}
                >
                  <FileCode size={14} /> SOURCE_CODE
                </button>
                <button 
                   onClick={() => setActiveTab('console')}
                   className={`px-6 py-3 text-[10px] font-mono font-black uppercase tracking-widest flex items-center gap-3 border-b-2 transition-all ${activeTab === 'console' ? 'bg-white/5 text-white border-brand' : 'text-white/40 border-transparent hover:text-white/60'}`}
                >
                  <Terminal size={14} /> SYSTEM_CONSOLE {logs.length > 0 && <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>}
                </button>
              </div>

              {/* Tab Content */}
              <div className="flex-1 overflow-hidden relative">
                 {activeTab === 'editor' ? (
                   <CodeEditor 
                     value={code} 
                     onChange={setCode} 
                     language="python"
                   />
                 ) : (
                   <LabConsole 
                     logs={logs} 
                     onClear={clearLogs}
                     isRunning={isRunning}
                   />
                 )}
              </div>
            </div>
          }
        />
      </div>
    </div>
  );
};
