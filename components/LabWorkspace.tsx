
import React, { useState, useEffect } from 'react';
import { DocViewer } from './DocViewer';
import { ResizableLayout } from './ResizableLayout';
import { Play, RotateCcw, Terminal, FileCode, CheckCircle, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface LabWorkspaceProps {
  topicId: string;
  onBack: () => void;
}

const MOCK_BOILERPLATE: Record<string, string> = {
  'default': `# Initialize your parameters
learning_rate = 0.01
epochs = 100

def train_model(data):
    # TODO: Implement the training loop
    print("Training started...")
    pass

# Execute
train_model(dataset)
`,
  'math/linear-algebra': `import numpy as np

# 1. Define two vectors
v1 = np.array([2, 5, 1])
v2 = np.array([4, -2, 3])

# 2. Calculate Dot Product manually
# dot = v1[0]*v2[0] + ...
dot_product = 0 

print(f"Dot Product: {dot_product}")
`,
  'ml/supervised': `from sklearn.linear_model import LinearRegression
import numpy as np

# Training Data
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

# TODO: Fit the model
model = LinearRegression()

# Predict for X = 5
prediction = 0
print(f"Prediction for 5: {prediction}")
`
};

export const LabWorkspace: React.FC<LabWorkspaceProps> = ({ topicId, onBack }) => {
  const [code, setCode] = useState(MOCK_BOILERPLATE[topicId] || MOCK_BOILERPLATE['default']);
  const [output, setOutput] = useState<string[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [activeTab, setActiveTab] = useState<'editor' | 'console'>('editor');

  const runCode = () => {
    setIsRunning(true);
    setOutput([]);
    setActiveTab('console');
    
    // Simulate execution time
    setTimeout(() => {
      setIsRunning(false);
      setOutput([
        "> python script.py",
        "[INFO] Environment initialized",
        "...",
        `Result: Success (${Math.random().toFixed(4)})`,
        "Process finished with exit code 0"
      ]);
    }, 1500);
  };

  const resetCode = () => {
    setCode(MOCK_BOILERPLATE[topicId] || MOCK_BOILERPLATE['default']);
    setOutput([]);
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
           <span className="text-xs text-slate-600 font-mono hidden sm:inline">{topicId}</span>
        </div>
        
        <div className="flex items-center gap-3">
          <button 
            onClick={resetCode}
            className="p-2 text-slate-500 hover:text-white transition-colors"
            title="Reset Code"
          >
            <RotateCcw size={16} />
          </button>
          <button 
            onClick={runCode}
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
              <div className="flex bg-[#252526] border-b border-[#1e1e1e]">
                <button 
                   onClick={() => setActiveTab('editor')}
                   className={`px-4 py-2.5 text-xs font-medium flex items-center gap-2 border-t-2 ${activeTab === 'editor' ? 'bg-[#1e1e1e] text-indigo-300 border-indigo-500' : 'text-slate-500 border-transparent hover:text-slate-300'}`}
                >
                  <FileCode size={14} /> script.py
                </button>
                <button 
                   onClick={() => setActiveTab('console')}
                   className={`px-4 py-2.5 text-xs font-medium flex items-center gap-2 border-t-2 ${activeTab === 'console' ? 'bg-[#1e1e1e] text-white border-indigo-500' : 'text-slate-500 border-transparent hover:text-slate-300'}`}
                >
                  <Terminal size={14} /> Console {output.length > 0 && <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>}
                </button>
              </div>

              {/* Tab Content */}
              <div className="flex-1 overflow-hidden relative">
                 {activeTab === 'editor' ? (
                   <div className="h-full w-full relative">
                      {/* Mock Line Numbers */}
                      <div className="absolute left-0 top-0 bottom-0 w-10 bg-[#1e1e1e] border-r border-[#333] flex flex-col items-end pr-2 pt-4 text-[11px] font-mono text-slate-600 select-none">
                        {Array.from({length: 20}).map((_, i) => <div key={i}>{i+1}</div>)}
                      </div>
                      {/* Editor Area */}
                      <textarea
                        value={code}
                        onChange={(e) => setCode(e.target.value)}
                        className="w-full h-full pl-12 pr-4 pt-4 bg-[#1e1e1e] text-slate-300 font-mono text-sm resize-none outline-none focus:ring-0 leading-relaxed selection:bg-indigo-500/30"
                        spellCheck={false}
                      />
                   </div>
                 ) : (
                   <div className="h-full w-full bg-[#1e1e1e] p-4 font-mono text-xs text-slate-300 overflow-y-auto">
                      {output.length === 0 ? (
                        <div className="text-slate-600 italic mt-4 ml-2">No output yet. Run your code to see results.</div>
                      ) : (
                        output.map((line, i) => (
                          <motion.div 
                            key={i}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.1 }}
                            className="mb-1"
                          >
                             {line.startsWith('>') ? <span className="text-slate-500">{line}</span> : line}
                          </motion.div>
                        ))
                      )}
                      {output.length > 0 && !isRunning && (
                         <motion.div 
                           initial={{ opacity: 0 }} 
                           animate={{ opacity: 1 }} 
                           transition={{ delay: 0.5 }}
                           className="mt-4 flex items-center gap-2 text-emerald-500 font-bold"
                         >
                            <CheckCircle size={14} /> Execution Complete
                         </motion.div>
                      )}
                   </div>
                 )}
              </div>
            </div>
          }
        />
      </div>
    </div>
  );
};
