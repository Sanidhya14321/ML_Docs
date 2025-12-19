import React, { useState, useEffect, useRef, lazy, Suspense } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { MLModelType, ModelMetrics } from '../types';
import { MEDICAL_MODEL_DATA } from '../constants';
import { Loader2, Code, Activity, Database, TrendingUp, BarChart3, Play, Terminal, CheckCircle } from 'lucide-react';
import { CodeBlock } from '../components/CodeBlock';
import { motion, AnimatePresence } from 'framer-motion';

const CorrelationHeatmap = () => {
    const features = ['Age', 'BP', 'Chol', 'HR', 'Target'];
    const matrix = [
        [1.0, 0.3, 0.2, -0.4, 0.2],
        [0.3, 1.0, 0.1, -0.1, 0.4],
        [0.2, 0.1, 1.0, -0.1, 0.1],
        [-0.4, -0.1, -0.1, 1.0, -0.4],
        [0.2, 0.4, 0.1, -0.4, 1.0]
    ];

    const getColor = (val: number) => {
        if (val > 0.7) return 'bg-emerald-500';
        if (val > 0.4) return 'bg-emerald-600';
        if (val > 0.2) return 'bg-emerald-700';
        if (val < -0.4) return 'bg-rose-600';
        if (val < -0.2) return 'bg-rose-700';
        return 'bg-slate-800';
    };

    return (
        <div className="flex flex-col items-center">
            <div className="grid grid-cols-6 gap-2">
                <div className="w-16 h-12"></div>
                {features.map(f => <div key={f} className="w-16 h-12 flex items-center justify-center text-[10px] font-black text-slate-500 uppercase">{f}</div>)}
                
                {matrix.map((row, i) => (
                    <React.Fragment key={i}>
                        <div className="w-16 h-12 flex items-center justify-end pr-2 text-[10px] font-black text-slate-500 uppercase">{features[i]}</div>
                        {row.map((val, j) => (
                            <div key={j} className={`w-16 h-12 flex items-center justify-center rounded-lg border border-white/5 transition-all hover:scale-105 ${getColor(val)}`}>
                                <span className="text-[10px] font-mono font-bold text-white/90">{val.toFixed(1)}</span>
                            </div>
                        ))}
                    </React.Fragment>
                ))}
            </div>
        </div>
    );
};

export const ProjectLabView: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'eda' | 'code' | 'performance'>('performance');
  const [selectedModel, setSelectedModel] = useState<MLModelType>(MLModelType.LOGISTIC_REGRESSION);
  const [isTraining, setIsTraining] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [currentMetrics, setCurrentMetrics] = useState<ModelMetrics>(MEDICAL_MODEL_DATA[MLModelType.LOGISTIC_REGRESSION]);
  const [logs, setLogs] = useState<string[]>([]);
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  const trainModel = () => {
    setIsTraining(true);
    setShowSuccess(false);
    setLogs(['[SYSTEM] Initializing training pipeline...', '[INFO] Loading heart_vitals.csv...', '[INFO] Standardizing features...']);
    
    let step = 0;
    const interval = setInterval(() => {
      step++;
      if (step === 2) setLogs(prev => [...prev, '[DEBUG] Optimizing loss function (SGD)...']);
      if (step === 10) {
        setLogs(prev => [...prev, '[SUCCESS] Model convergence reached.', '[INFO] Calculating performance metrics...']);
        clearInterval(interval);
        setTimeout(() => {
          setCurrentMetrics(MEDICAL_MODEL_DATA[selectedModel]);
          setIsTraining(false);
          setShowSuccess(true);
          setTimeout(() => setShowSuccess(false), 3000);
        }, 500);
      }
    }, 400);
  };

  return (
    <div className="pb-24 space-y-16">
      <header className="relative">
        <h1 className="text-6xl font-serif font-bold text-white flex items-center gap-6">
          <span className="bg-indigo-600 px-5 py-2 rounded-2xl text-xl shadow-2xl shadow-indigo-600/30">Lab</span>
          Medical Case Study
        </h1>
        <p className="text-slate-400 mt-6 text-xl font-light leading-relaxed max-w-3xl">
          Simulating a clinical heart disease diagnostic tool. Select algorithms and analyze metrics to find the most reliable predictor.
        </p>
      </header>

      <div className="flex bg-slate-900/50 backdrop-blur-md p-1.5 rounded-2xl border border-slate-800/50 max-w-md">
         {[
           { id: 'eda', icon: <BarChart3 size={16} />, label: 'Data EDA' },
           { id: 'performance', icon: <TrendingUp size={16} />, label: 'Benchmarks' },
           { id: 'code', icon: <Code size={16} />, label: 'Notebook' }
         ].map(tab => (
           <button 
             key={tab.id}
             onClick={() => setActiveTab(tab.id as any)}
             className={`flex-1 py-3 text-[11px] font-black uppercase tracking-widest flex items-center justify-center gap-2 rounded-xl transition-all ${activeTab === tab.id ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30' : 'text-slate-500 hover:text-slate-300'}`}
           >
             {tab.icon} {tab.label}
           </button>
         ))}
      </div>

      <AnimatePresence mode="wait">
        {activeTab === 'performance' && (
          <motion.div 
            key="perf"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-12"
          >
            <div className="glass-card p-10 rounded-3xl flex flex-col lg:flex-row gap-12 items-center justify-between relative">
              <AnimatePresence>
                {showSuccess && (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 bg-indigo-600/10 backdrop-blur-sm z-20 flex items-center justify-center rounded-3xl border-2 border-indigo-500/50"
                  >
                    <div className="text-center">
                       <CheckCircle size={48} className="text-indigo-400 mx-auto mb-4" />
                       <h3 className="text-xl font-bold text-white">Analysis Complete</h3>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="w-full lg:w-1/3 space-y-6">
                <div>
                  <label className="block text-[10px] font-black text-indigo-400 uppercase tracking-[0.2em] mb-4">Select Algorithm</label>
                  <select 
                    value={selectedModel} 
                    onChange={(e) => setSelectedModel(e.target.value as MLModelType)} 
                    disabled={isTraining} 
                    className="w-full bg-slate-950/80 text-white border-2 border-slate-800 rounded-2xl p-4 font-black text-sm cursor-pointer hover:border-indigo-500/50 transition-all focus:outline-none"
                  >
                    {Object.values(MLModelType).map(m => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>
                <motion.button 
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={trainModel}
                  disabled={isTraining}
                  className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-black py-4 rounded-2xl shadow-xl shadow-indigo-600/20 transition-all flex items-center justify-center gap-3"
                >
                  {isTraining ? <Loader2 className="animate-spin" size={18} /> : <Play size={18} fill="white" />}
                  {isTraining ? 'Training...' : 'Fit & Evaluate'}
                </motion.button>
              </div>

              <div className="flex-1 w-full grid grid-cols-1 md:grid-cols-3 gap-6">
                {[
                  { label: 'Accuracy', key: 'accuracy', color: 'emerald' },
                  { label: 'Precision', key: 'precision', color: 'indigo' },
                  { label: 'Recall', key: 'recall', color: 'fuchsia' }
                ].map((metric) => (
                  <div key={metric.label} className="bg-slate-950/50 p-8 rounded-3xl border border-slate-800 text-center relative group overflow-hidden">
                    <div className="text-[9px] text-slate-500 uppercase font-black tracking-widest mb-2">{metric.label}</div>
                    <div className={`text-4xl font-mono font-black text-${metric.color}-400`}>
                      {isTraining ? (
                        <span className="inline-block animate-pulse">--</span>
                      ) : (
                        `${currentMetrics[metric.key as keyof ModelMetrics]}%`
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {isTraining && (
              <motion.div 
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                className="bg-slate-950 rounded-2xl border border-slate-800 p-6 overflow-hidden shadow-2xl"
              >
                <div className="flex items-center gap-2 mb-4 pb-4 border-b border-slate-900">
                   <Terminal size={16} className="text-indigo-400" />
                   <span className="text-[10px] font-black uppercase tracking-widest text-slate-500">Training_Log_Stream.txt</span>
                </div>
                <div 
                  ref={terminalRef}
                  className="h-48 overflow-y-auto font-mono text-[11px] leading-relaxed text-slate-400"
                >
                  {logs.map((log, i) => (
                    <div key={i} className="mb-1 flex gap-4">
                       <span className="text-slate-700">[{new Date().toLocaleTimeString()}]</span>
                       <span>{log}</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};