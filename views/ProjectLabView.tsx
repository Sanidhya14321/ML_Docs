
import React, { useState, useEffect, useRef } from 'react';
import { MLModelType, ModelMetrics } from '../types';
import { MEDICAL_MODEL_DATA } from '../constants';
import { Loader2, Code, TrendingUp, BarChart3, Play, Terminal, CheckCircle } from 'lucide-react';
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
        if (val === 1.0) return 'bg-slate-100 dark:bg-slate-800 border-slate-200 dark:border-slate-700';
        if (val > 0.7) return 'bg-emerald-400/80 dark:bg-emerald-500 border-emerald-300 dark:border-emerald-400';
        if (val > 0.3) return 'bg-emerald-200 dark:bg-emerald-600/80 border-emerald-300 dark:border-emerald-500/50';
        if (val > 0) return 'bg-emerald-100 dark:bg-emerald-900/40 border-emerald-200 dark:border-emerald-800';
        if (val < -0.3) return 'bg-rose-300 dark:bg-rose-600/80 border-rose-300 dark:border-rose-500/50';
        if (val < 0) return 'bg-rose-100 dark:bg-rose-900/40 border-rose-200 dark:border-rose-800';
        return 'bg-slate-100 dark:bg-slate-800 border-slate-200 dark:border-slate-700';
    };

    return (
        <div className="flex flex-col items-center select-none w-full overflow-x-auto pb-4">
            <div className="grid grid-cols-6 gap-2 min-w-[320px]">
                {/* Header Row */}
                <div className="w-12 h-10 md:w-16 md:h-12"></div>
                {features.map(f => (
                    <div key={f} className="w-12 h-10 md:w-16 md:h-12 flex items-center justify-center text-[10px] font-black text-slate-500 uppercase tracking-widest">{f}</div>
                ))}
                
                {/* Data Rows */}
                {matrix.map((row, i) => (
                    <React.Fragment key={i}>
                        <div className="w-12 h-10 md:w-16 md:h-12 flex items-center justify-end pr-3 text-[10px] font-black text-slate-500 uppercase tracking-widest">{features[i]}</div>
                        {row.map((val, j) => (
                            <div 
                                key={j} 
                                className={`w-12 h-10 md:w-16 md:h-12 flex items-center justify-center rounded-lg border transition-all hover:scale-110 hover:z-10 cursor-help group relative ${getColor(val)}`}
                            >
                                <span className={`text-[10px] font-mono font-bold ${val === 1 ? 'text-slate-400 dark:text-slate-600' : 'text-slate-800 dark:text-white/90'}`}>
                                    {val.toFixed(1)}
                                </span>
                                {val !== 1 && (
                                    <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-900 text-white text-[9px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-20 border border-slate-700">
                                        {features[i]} vs {features[j]}
                                    </div>
                                )}
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

  // Helper to safely append logs during async operations
  const addLog = (msg: string) => setLogs(prev => [...prev, msg]);

  const trainModel = async () => {
    setIsTraining(true);
    setShowSuccess(false);
    setLogs([]); // Clear logs start

    try {
        // 1. Simulate API Connection
        addLog('[SYSTEM] Initializing secure connection to Training Cluster (us-east-1)...');
        await new Promise(r => setTimeout(r, 600));

        // 2. Simulate API Request
        addLog(`[NETWORK] POST https://api.ai-codex.dev/v1/train`);
        addLog(`[PAYLOAD] { model: "${selectedModel}", dataset: "heart_vitals_v2", hyperparams: "auto" }`);
        
        // Network Latency Simulation
        await new Promise(r => setTimeout(r, 1200));

        // 3. Simulate Server-Side Streamed Logs
        const serverLogs = [
            '[BACKEND] Request received. Allocating GPU instance...',
            '[BACKEND] Dataset loaded: heart_vitals.csv (Size: 45KB, Rows: 303)',
            '[BACKEND] Preprocessing: StandardScaler applied to numerical features.',
            '[BACKEND] Preprocessing: OneHotEncoder applied to categorical features.',
            '[BACKEND] Train/Test Split: 80/20 with random_state=42.',
            `[BACKEND] Initializing ${selectedModel} estimator...`,
            '[BACKEND] Fitting model to training set...',
            '[BACKEND] Optimizing loss function...',
            '[BACKEND] Validating model performance...',
            '[SUCCESS] Training converged. Generating metrics report.'
        ];

        for (const log of serverLogs) {
            // Variable delay to simulate real processing time
            await new Promise(r => setTimeout(r, 300 + Math.random() * 500));
            addLog(log);
        }

        // 4. "Fetch" the result
        addLog('[NETWORK] Receiving response payload (200 OK)...');
        await new Promise(r => setTimeout(r, 500));

        // Update State
        setCurrentMetrics(MEDICAL_MODEL_DATA[selectedModel]);
        setShowSuccess(true);
        setTimeout(() => setShowSuccess(false), 3000);

    } catch (error) {
        addLog('[ERROR] Connection timeout. Training failed.');
    } finally {
        setIsTraining(false);
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="pb-24 space-y-16"
    >
      <header className="relative">
        <motion.h1 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="text-4xl md:text-6xl font-serif font-bold text-slate-900 dark:text-white flex flex-col md:flex-row items-start md:items-center gap-4 md:gap-6"
        >
          <span className="bg-indigo-600 px-5 py-2 rounded-2xl text-lg md:text-xl shadow-2xl shadow-indigo-600/30 text-white self-start">Lab</span>
          Medical Case Study
        </motion.h1>
        <motion.p 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-slate-500 dark:text-slate-400 mt-6 text-lg md:text-xl font-light leading-relaxed max-w-3xl"
        >
          Simulating a clinical heart disease diagnostic tool. Select algorithms and analyze metrics to find the most reliable predictor.
        </motion.p>
      </header>

      <motion.div
        initial="hidden"
        animate="show"
        variants={{
          hidden: { opacity: 0 },
          show: {
            opacity: 1,
            transition: {
              staggerChildren: 0.1
            }
          }
        }}
        className="flex bg-slate-200 dark:bg-slate-900/50 backdrop-blur-md p-1.5 rounded-2xl border border-slate-300 dark:border-slate-800/50 max-w-md transition-colors duration-300"
        role="tablist"
        aria-label="Lab Views"
      >
         {[
           { id: 'eda', icon: <BarChart3 size={16} />, label: 'Data EDA' },
           { id: 'performance', icon: <TrendingUp size={16} />, label: 'Benchmarks' },
           { id: 'code', icon: <Code size={16} />, label: 'Notebook' }
         ].map(tab => (
           <button 
             key={tab.id}
             onClick={() => setActiveTab(tab.id as any)}
             role="tab"
             aria-selected={activeTab === tab.id}
             aria-controls={`panel-${tab.id}`}
             id={`tab-${tab.id}`}
             className={`flex-1 py-3 text-[11px] font-black uppercase tracking-widest flex items-center justify-center gap-2 rounded-xl transition-all duration-300 ${activeTab === tab.id ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30' : 'text-slate-600 dark:text-slate-500 hover:text-slate-900 dark:hover:text-slate-300'}`}
           >
             {tab.icon} {tab.label}
           </button>
         ))}
      </motion.div>

      <AnimatePresence mode="wait">
        {activeTab === 'eda' && (
             <motion.div
                key="eda"
                role="tabpanel"
                id="panel-eda"
                aria-labelledby="tab-eda"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="space-y-8"
             >
                <div className="p-8 rounded-3xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/50 shadow-xl transition-colors duration-300">
                    <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-8 flex items-center gap-3 transition-colors duration-300">
                        <BarChart3 className="text-indigo-500" /> Feature Correlation Matrix
                    </h3>
                    <div className="flex flex-col lg:flex-row items-center gap-16">
                        <CorrelationHeatmap />
                        <div className="flex-1 space-y-6">
                            <div className="space-y-2">
                                <h4 className="text-xs font-black text-slate-500 uppercase tracking-widest">Analysis</h4>
                                <p className="text-slate-600 dark:text-slate-300 leading-relaxed font-light transition-colors duration-300">
                                    The heatmap reveals critical dependencies. Values close to <span className="text-emerald-500 dark:text-emerald-400 font-bold">1.0</span> indicate strong positive correlation, while <span className="text-rose-500 dark:text-rose-400 font-bold">-1.0</span> indicates strong negative correlation.
                                </p>
                            </div>
                            
                            <div className="space-y-3">
                                <h4 className="text-xs font-black text-slate-500 uppercase tracking-widest">Key Insights</h4>
                                <ul className="space-y-4">
                                    <li className="flex gap-4 text-slate-500 dark:text-slate-400 text-sm bg-slate-50 dark:bg-slate-950/50 p-4 rounded-xl border border-slate-200 dark:border-slate-800/50 transition-colors duration-300">
                                        <div className="w-1.5 h-1.5 rounded-full bg-rose-500 mt-2 shrink-0" />
                                        <span>
                                            <strong className="text-rose-500 dark:text-rose-400 block mb-1">Age vs. HR (-0.4)</strong>
                                            Maximum heart rate clearly decreases as patient age increases, a standard physiological trend.
                                        </span>
                                    </li>
                                    <li className="flex gap-4 text-slate-500 dark:text-slate-400 text-sm bg-slate-50 dark:bg-slate-950/50 p-4 rounded-xl border border-slate-200 dark:border-slate-800/50 transition-colors duration-300">
                                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-2 shrink-0" />
                                        <span>
                                            <strong className="text-emerald-500 dark:text-emerald-400 block mb-1">BP vs. Target (0.4)</strong>
                                            Elevated blood pressure shows a significant positive correlation with the heart disease target variable.
                                        </span>
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
             </motion.div>
        )}

        {activeTab === 'performance' && (
          <motion.div 
            key="perf"
            role="tabpanel"
            id="panel-performance"
            aria-labelledby="tab-performance"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-12"
          >
            <div className="glass-card p-10 rounded-3xl flex flex-col lg:flex-row gap-12 items-center justify-between relative bg-white dark:bg-slate-900/30 border border-slate-200 dark:border-slate-800 shadow-xl transition-colors duration-300">
              <AnimatePresence>
                {showSuccess && (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 bg-indigo-50/90 dark:bg-indigo-600/10 backdrop-blur-sm z-20 flex items-center justify-center rounded-3xl border-2 border-indigo-500/50"
                  >
                    <div className="text-center">
                       <CheckCircle size={48} className="text-indigo-500 dark:text-indigo-400 mx-auto mb-4" />
                       <h3 className="text-xl font-bold text-indigo-900 dark:text-white">Analysis Complete</h3>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="w-full lg:w-1/3 space-y-6">
                <div>
                  <label className="block text-[10px] font-black text-indigo-600 dark:text-indigo-400 uppercase tracking-[0.2em] mb-4">Select Algorithm</label>
                  <select 
                    value={selectedModel} 
                    onChange={(e) => setSelectedModel(e.target.value as MLModelType)} 
                    disabled={isTraining} 
                    className="w-full bg-slate-50 dark:bg-slate-950/80 text-slate-900 dark:text-white border-2 border-slate-200 dark:border-slate-800 rounded-2xl p-4 font-black text-sm cursor-pointer hover:border-indigo-500/50 transition-all duration-300 focus:outline-none"
                  >
                    {Object.values(MLModelType).map(m => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>
                <motion.button 
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={trainModel}
                  disabled={isTraining}
                  className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-black py-4 rounded-2xl shadow-xl shadow-indigo-600/20 transition-all duration-300 flex items-center justify-center gap-3"
                >
                  {isTraining ? <Loader2 className="animate-spin" size={18} /> : <Play size={18} fill="white" />}
                  {isTraining ? 'Training...' : 'Fit & Evaluate'}
                </motion.button>
              </div>

              <div className="flex-1 w-full grid grid-cols-1 md:grid-cols-3 gap-6">
                {[
                  { label: 'Accuracy', key: 'accuracy', color: 'emerald' },
                  { label: 'Precision', key: 'precision', color: 'indigo' },
                  { label: 'Recall', key: 'recall', color: 'rose' }
                ].map((metric) => (
                  <div key={metric.label} className="bg-slate-50 dark:bg-slate-950/50 p-8 rounded-3xl border border-slate-200 dark:border-slate-800 text-center relative group overflow-hidden transition-colors duration-300">
                    <div className="text-[9px] text-slate-500 uppercase font-black tracking-widest mb-2">{metric.label}</div>
                    <div className={`text-4xl font-mono font-black text-${metric.color}-600 dark:text-${metric.color}-400 transition-colors duration-300`}>
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
                className="bg-slate-900 rounded-2xl border border-slate-800 p-6 overflow-hidden shadow-2xl"
              >
                <div className="flex items-center gap-2 mb-4 pb-4 border-b border-slate-800">
                   <Terminal size={16} className="text-indigo-400" />
                   <span className="text-[10px] font-black uppercase tracking-widest text-slate-500">Training_Log_Stream.txt</span>
                </div>
                <div 
                  ref={terminalRef}
                  className="h-48 overflow-y-auto font-mono text-[11px] leading-relaxed text-slate-400"
                >
                  {logs.map((log, i) => (
                    <div key={i} className="mb-1 flex gap-4">
                       <span className="text-slate-600">[{new Date().toLocaleTimeString()}]</span>
                       <span>{log}</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>
        )}

        {activeTab === 'code' && (
             <motion.div
                key="code"
                role="tabpanel"
                id="panel-code"
                aria-labelledby="tab-code"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center justify-center min-h-[400px] border border-dashed border-slate-200 dark:border-slate-800 rounded-3xl bg-slate-50 dark:bg-slate-900/20 transition-colors duration-300"
             >
                <Code size={48} className="text-slate-400 dark:text-slate-700 mb-4" />
                <h3 className="text-xl font-bold text-slate-500 dark:text-slate-400">Notebook View</h3>
                <p className="text-slate-400 dark:text-slate-600 mt-2">To access the full code environment, please start the Lab from the main topic page.</p>
             </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};
