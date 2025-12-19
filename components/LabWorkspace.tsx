
import React, { useState, useEffect } from 'react';
import { DocViewer } from './DocViewer';
import { ResizableLayout } from './ResizableLayout';
import { CodeEditor } from './CodeEditor';
import { LabConsole } from './LabConsole';
import { useCodeRunner } from '../hooks/useCodeRunner';
import { getTopicById } from '../lib/contentHelpers';
import { Play, RotateCcw, Terminal, FileCode, Loader2, Lightbulb } from 'lucide-react';

interface LabWorkspaceProps {
  topicId: string;
  onBack: () => void;
}

const DEFAULT_CODE = `# No specific lab configuration found.\nprint("Hello World")`;

export const LabWorkspace: React.FC<LabWorkspaceProps> = ({ topicId, onBack }) => {
  const topic = getTopicById(topicId);
  const initialCode = topic?.labConfig?.initialCode || DEFAULT_CODE;

  const [code, setCode] = useState(initialCode);
  const [activeTab, setActiveTab] = useState<'editor' | 'console'>('editor');
  
  // Use custom hook for execution logic
  const { logs, isRunning, runCode, clearLogs } = useCodeRunner();

  // Reset code when topic changes
  useEffect(() => {
    setCode(initialCode);
    clearLogs();
  }, [topicId, initialCode, clearLogs]);

  const handleRun = () => {
    setActiveTab('console');
    runCode(code);
  };

  const handleReset = () => {
    setCode(initialCode);
    clearLogs();
  };

  return (
    <div className="h-full flex flex-col bg-[#0f1117] relative z-50">
      {/* Lab Header */}
      <header className="h-14 border-b border-slate-800 bg-[#020617] flex items-center justify-between px-4 shrink-0">
        <div className="flex items-center gap-4">
           <button onClick={onBack} className="text-xs font-mono text-slate-500 hover:text-white transition-colors">← Exit Lab</button>
           <div className="h-4 w-px bg-slate-800" />
           <span className="text-sm font-bold text-slate-200 flex items-center gap-2">
              <Terminal size={14} className="text-indigo-400" />
              Interactive Workspace
           </span>
           <span className="text-xs text-slate-600 font-mono hidden sm:inline">{topic?.title || topicId}</span>
        </div>
        
        <div className="flex items-center gap-3">
          {topic?.labConfig?.hints && topic.labConfig.hints.length > 0 && (
             <button className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-md hover:bg-amber-500/20 transition-colors">
                <Lightbulb size={12} /> Hints
             </button>
          )}
          <button 
            onClick={handleReset}
            disabled={isRunning}
            className="p-2 text-slate-500 hover:text-white transition-colors disabled:opacity-50"
            title="Reset Code"
          >
            <RotateCcw size={16} />
          </button>
          <button 
            onClick={handleRun}
            disabled={isRunning}
            className="flex items-center gap-2 px-4 py-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-bold rounded-md transition-all shadow-lg shadow-indigo-900/20"
          >
            {isRunning ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
            {isRunning ? 'Running...' : 'Run Code'}
          </button>
        </div>
      </header>

      {/* Main Workspace */}
      <div className="flex-1 overflow-hidden">
        <ResizableLayout
          initialLeftWidth={35}
          left={
            <div className="bg-[#020617] min-h-full p-6 pb-20">
               <DocViewer topicId={topicId} title="Lab Instructions" isCompact={true} />
            </div>
          }
          right={
            <div className="flex flex-col h-full bg-[#1e1e1e]">
              {/* Editor Tabs */}
              <div className="flex bg-[#252526] border-b border-[#1e1e1e] shrink-0">
                <button 
                   onClick={() => setActiveTab('editor')}
                   className={`px-4 py-2.5 text-xs font-medium flex items-center gap-2 border-t-2 ${activeTab === 'editor' ? 'bg-[#1e1e1e] text-indigo-300 border-indigo-500' : 'text-slate-500 border-transparent hover:text-slate-300'}`}
                >
                  <FileCode size={14} /> main.py
                </button>
                <button 
                   onClick={() => setActiveTab('console')}
                   className={`px-4 py-2.5 text-xs font-medium flex items-center gap-2 border-t-2 ${activeTab === 'console' ? 'bg-[#1e1e1e] text-white border-indigo-500' : 'text-slate-500 border-transparent hover:text-slate-300'}`}
                >
                  <Terminal size={14} /> Console {logs.length > 0 && <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>}
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
